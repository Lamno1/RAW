"""Create resumable, non-pickle BuildingNet caches and train-only statistics."""
from __future__ import annotations
import argparse, json, os, tempfile, zipfile
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dataset import load_ply, NUM_PART_CLASSES, PART_LABELS

def load_raw_labels(archive, name):
    with archive.open(f"point_labels/{name}_label.json") as stream: mapping=json.load(stream)
    labels=np.empty(len(mapping),dtype=np.uint8)
    for index,value in mapping.items(): labels[int(index)]=int(value)
    return labels

def atomic_npz(path, features, labels):
    path.parent.mkdir(parents=True,exist_ok=True)
    handle,temp=tempfile.mkstemp(prefix=path.stem+".",suffix=".npz",dir=path.parent); os.close(handle)
    try:
        np.savez(temp,features=features.astype(np.float32,copy=False),labels=labels)
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--data-dir",type=Path,default=Path(__file__).parent)
    parser.add_argument("--limit",type=int,help="Smoke-test only; do not use its weights for training")
    parser.add_argument("--overwrite",action="store_true"); args=parser.parse_args(); root=args.data_dir
    archive_path=root/"model_data"/"point_cloud"/"point_labels.zip"; counts=np.zeros(NUM_PART_CLASSES,dtype=np.int64)
    with zipfile.ZipFile(archive_path) as archive:
        for split in ("train","val","test"):
            names=[x.strip() for x in (root/"splits"/f"{split}_split.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
            if args.limit: names=names[:args.limit]
            for name in tqdm(names,desc=f"cache {split}"):
                target=root/"cache_npz"/split/f"{name}.npz"
                labels=load_raw_labels(archive,name)
                if split=="train":
                    valid=labels[(labels>=1)&(labels<=NUM_PART_CLASSES)]
                    counts+=np.bincount(valid-1,minlength=NUM_PART_CLASSES)
                if args.overwrite or not target.exists():
                    features=load_ply(root/"POINT_CLOUDS"/f"{name}.ply")
                    if len(features)!=len(labels): raise ValueError(f"Lệch point/label: {name}")
                    atomic_npz(target,features,labels)
    if args.limit:
        print("Smoke test complete; limited-data weights were not written."); return
    total=counts.sum(); frequencies=counts/total
    weights=np.zeros(NUM_PART_CLASSES,dtype=np.float32); present=counts>0
    weights[present]=1.0/np.log(1.2+frequencies[present]); weights*=NUM_PART_CLASSES/weights.sum()
    np.save(root/"class_counts.npy",counts,allow_pickle=False); np.save(root/"class_weights.npy",weights,allow_pickle=False)
    report={str(i+1):{"name":PART_LABELS[i],"count":int(counts[i]),"frequency":float(frequencies[i]),"weight":float(weights[i])} for i in range(NUM_PART_CLASSES)}
    (root/"class_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Cached all splits; valid train points={total:,}; label 0 is excluded.")
if __name__=="__main__": main()
