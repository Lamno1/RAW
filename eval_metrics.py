"""Evaluate a frozen segmentation checkpoint once on the official test split."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import BuildingNetDataset, NUM_PART_CLASSES, PART_LABELS, IGNORE_INDEX
from model import GeometryAwarePCT

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=Path(__file__).parent)
    p.add_argument("--checkpoint",type=Path); p.add_argument("--points",type=int,default=1024); p.add_argument("--batch-size",type=int,default=8); a=p.parse_args()
    checkpoint=a.checkpoint or a.data_dir/"checkpoints"/"best_segmentation_model.pth"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved=torch.load(checkpoint,map_location=device,weights_only=True); model=GeometryAwarePCT().to(device); model.load_state_dict(saved["model"]); model.eval()
    data=BuildingNetDataset(a.data_dir,"test",a.points,False); loader=DataLoader(data,a.batch_size,shuffle=False,num_workers=0)
    intersections=np.zeros(NUM_PART_CLASSES,dtype=np.int64); unions=np.zeros_like(intersections); correct=total=0; shape_ious=[]
    with torch.no_grad():
        for batch in loader:
            x=batch["features"].to(device); y=batch["label"].to(device); pred=model(x).argmax(-1)
            valid=y!=IGNORE_INDEX; correct+=(pred[valid]==y[valid]).sum().item(); total+=valid.sum().item()
            for sample in range(y.shape[0]):
                sample_scores=[]; sample_valid=valid[sample]
                for cls in range(NUM_PART_CLASSES):
                    pm=(pred[sample]==cls)&sample_valid; ym=(y[sample]==cls)&sample_valid
                    inter=(pm&ym).sum().item(); union=(pm|ym).sum().item()
                    intersections[cls]+=inter; unions[cls]+=union
                    if union: sample_scores.append(inter/union)
                shape_ious.append(float(np.mean(sample_scores)))
    iou=np.divide(intersections,unions,out=np.full(NUM_PART_CLASSES,np.nan),where=unions>0)
    print(f"class_accuracy={correct/total:.6f} part_IoU={np.nanmean(iou):.6f} shape_IoU={np.mean(shape_ious):.6f}")
    for cls,value in enumerate(iou): print(f"{cls+1:02d}_{PART_LABELS[cls]}_iou={value:.6f}")
if __name__=="__main__": main()
