"""Train the corrected BuildingNet point-segmentation baseline."""
import argparse, csv, random, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import BuildingNetDataset, NUM_PART_CLASSES, IGNORE_INDEX
from model import GeometryAwarePCT

def mean_iou(pred, target):
    scores=[]
    for cls in range(NUM_PART_CLASSES):
        p,t=pred==cls,target==cls; union=(p|t).sum().item()
        if union: scores.append((p&t).sum().item()/union)
    return float(np.mean(scores)) if scores else 0.0

def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    training=optimizer is not None; model.train(training); total=correct=0; loss_sum=0.; ps=[]; ts=[]
    for batch in tqdm(loader, desc="train" if training else "val", leave=False):
        x,y=batch["features"].to(device,non_blocking=True),batch["label"].to(device,non_blocking=True)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type=="cuda"):
            logits=model(x); loss=loss_fn(logits.reshape(-1,NUM_PART_CLASSES),y.reshape(-1))
        if training: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        pred=logits.argmax(-1); valid=y!=IGNORE_INDEX; n=valid.sum().item()
        total+=n; correct+=(pred[valid]==y[valid]).sum().item(); loss_sum+=loss.item()*n
        ps.append(pred[valid].detach().cpu().flatten()); ts.append(y[valid].detach().cpu().flatten())
    return loss_sum/max(total,1), correct/max(total,1), mean_iou(torch.cat(ps),torch.cat(ts))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=Path(__file__).parent)
    parser.add_argument("--points",type=int,default=1024)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--loss",choices=("ce","weighted_ce"),default="ce")
    args=parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds=BuildingNetDataset(args.data_dir,"train",args.points,True,args.seed)
    val_ds  =BuildingNetDataset(args.data_dir,"val",  args.points,False,args.seed)
    pin=device.type=="cuda"
    g=torch.Generator(); g.manual_seed(args.seed)
    train_loader=DataLoader(train_ds,args.batch_size,shuffle=True, generator=g,num_workers=0,pin_memory=pin)
    val_loader  =DataLoader(val_ds,  args.batch_size,shuffle=False,              num_workers=0,pin_memory=pin)

    model=GeometryAwarePCT().to(device)
    weights=None
    if args.loss=="weighted_ce":
        weights=torch.from_numpy(np.load(args.data_dir/"class_weights.npy",allow_pickle=False)).float().to(device)
    loss_fn=nn.CrossEntropyLoss(weight=weights,ignore_index=IGNORE_INDEX)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-6)
    scaler=torch.amp.GradScaler(device.type,enabled=device.type=="cuda")

    ckpt_dir=args.data_dir/"checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    log_dir =args.data_dir/"logs";        log_dir.mkdir(exist_ok=True)
    run_name=f"{args.loss}_seed{args.seed}"
    ckpt_path=ckpt_dir/f"best_{run_name}.pth"
    log_path =log_dir /f"{run_name}.csv"

    best=-1.
    print(f"device={device}  train={len(train_ds)}  val={len(val_ds)}  run={run_name}")
    with open(log_path,"w",newline="",encoding="utf-8") as fh:
        writer=csv.writer(fh)
        writer.writerow(["epoch","train_loss","train_acc","train_miou","val_loss","val_acc","val_miou"])
        for step in range(1,args.epochs+1):
            tr=run_epoch(model,train_loader,loss_fn,device,optimizer,scaler)
            with torch.no_grad(): va=run_epoch(model,val_loader,loss_fn,device)
            scheduler.step()
            print(f"epoch={step:03d}  train=({tr[0]:.4f},{tr[1]:.4f},{tr[2]:.4f})  val=({va[0]:.4f},{va[1]:.4f},{va[2]:.4f})")
            writer.writerow([step,*tr,*va]); fh.flush()
            if va[2]>best:
                best=va[2]
                torch.save({"model":model.state_dict(),"epoch":step,"val_miou":best,
                            "num_classes":NUM_PART_CLASSES,"in_channels":9,"run":run_name},ckpt_path)
                print(f"  → checkpoint saved (val_miou={best:.4f})")
    print(f"Done. Best val Part IoU = {best:.4f}  →  {ckpt_path}")

if __name__=="__main__": main()

