"""Independent five-seed full-cloud paired analysis (registered extension)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel

ROOT=Path(__file__).resolve().parent; RUN=ROOT/"runs"/"full_cloud_validation"; SEEDS=(42,123,2026,4409,5513)
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def sd(a): return float(np.std(np.asarray(a,dtype=float),ddof=1))
def main():
 rec={"s5":[],"s2":[]}
 for arm in rec:
  for seed in SEEDS:
   p=RUN/f"{arm}_seed{seed}.json"; x=json.loads(p.read_text()); ck=Path(x["checkpoint"])
   assert x["architecture"]==arm and x["training_seed"]==seed and x["split"]=="validation"
   assert x["dry_run"] is False and x["test_split_used"] is False and x["protocol_verification_passed"] is True
   assert x["vote_seeds"]==[1103,2207,3301] and sha(ck)==x["checkpoint_sha256"]
   m=x["metrics"]; rec[arm].append({"seed":seed,"part_iou":float(m["part_iou"]),"accuracy":float(m["classification_accuracy"]),"head_iou":float(m["group_iou"]["head"]),"mid_iou":float(m["group_iou"]["mid"]),"tail_iou":float(m["group_iou"]["tail"]),"checkpoint":str(ck),"checkpoint_sha256":sha(ck)})
 def stat(key):
  a=np.array([r[key] for r in rec['s5']]); b=np.array([r[key] for r in rec['s2']]); d=a-b; t=ttest_rel(a,b)
  return {"s5_values":a.tolist(),"s2_values":b.tolist(),"delta_values":d.tolist(),"s5_mean":float(a.mean()),"s5_sd":sd(a),"s2_mean":float(b.mean()),"s2_sd":sd(b),"delta_mean":float(d.mean()),"delta_sd":sd(d),"paired_t":float(t.statistic),"paired_p_two_sided":float(t.pvalue)}
 out={"schema_version":"1.0-5seed","seeds":list(SEEDS),"arms":rec,"paired_statistics":{"part_iou":stat('part_iou'),"classification_accuracy":stat('accuracy'),"head_iou":stat('head_iou'),"mid_iou":stat('mid_iou'),"tail_iou":stat('tail_iou')},"all_artifact_checks_passed":True,"notes":["Five-seed full-cloud validation; S5 row-softmax versus paired S2 Lovasz."]}
 p=RUN/"paired_summary_5seed.json"; p.write_text(json.dumps(out,indent=2)+"\n",encoding='utf-8'); print(json.dumps(out["paired_statistics"],indent=2)); print(p)
if __name__=='__main__': main()
