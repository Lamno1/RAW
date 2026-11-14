# S7–S8 latency profile: measured conclusion

Date: 2026-09-30. Status: post-run interpretation of the frozen validation-only profile. This document does not change the S8 promotion decision or authorize test-set access.

## Provenance and scope

- Preregistration: `protocol/s8/latency_profile_preregistration.json`, SHA-256 `526e977678c63e0c21f9dd3c6969f392cd2aa3104595ddc4102744f38c0b905d`.
- Authorization: `protocol/s8/latency_profile_authorization.json`, SHA-256 `a8c3021d98ce3e06958c5e3b1ae385e912967bbe6a0b45a45ea0ead455810386`.
- Raw result: `runs/s8_latency_profile/latency_profile.json`, SHA-256 `a2f44e1b1a57341c8dd6eb1737045c3634dffebd0f9cb6affa25737542751431`. Its `status.json` says `COMPLETE` and names that same result hash.
- The profiler checked 24 locked dependency hashes before data access. It used the frozen seed-42 S7 and S8 checkpoints, eight fixed validation building indices, one fixed vote per building, batch sizes 1 and 4, and two repetitions with reversed condition order. It did not use segmentation labels or the test split.
- There are **8 distinct buildings**. Each condition has 16 building-runs (8 buildings × 2 repetitions), 400 chunks, and either 400 forward calls at batch 1 or 112 at batch 4. Matched conditions use the same chunk indices.

## Timing results

All times below are pooled over the 16 building-runs and 400 chunks in each condition. The `topology` timer includes CPU chunk selection/normalization and topology construction; it does not separate anchors, support, kNN, and row-softmax transport.

| Condition | Total (s) | CPU chunk/topology (s) | GPU forward (s) | Forward calls |
|---|---:|---:|---:|---:|
| S7, batch 1 | 53.062 | 33.401 | 18.139 | 400 |
| S7, batch 4 | 41.538 | 34.074 | 6.606 | 112 |
| S8, batch 1 | 160.119 | 122.846 | 35.621 | 400 |
| S8, batch 4 | 144.441 | 129.722 | 13.680 | 112 |

- At matched batch 4, S8/S7 total time is **3.477×**. The two counterbalanced repetitions yield 3.487× and 3.468×. At matched batch 1, the pooled ratio is 3.018×.
- At batch 4, CPU chunk/topology time differs by 95.649 s and total time by 102.902 s. The measured topology-stage difference therefore accounts for **92.95% of this subset's total-time gap**. GPU forward accounts for 6.87% of the gap. This is a stage-time decomposition, not a suboperation-level causal attribution.
- Changing S8 from batch 1 to batch 4 lowers forward time from 35.621 s to 13.680 s, while total time falls from 160.119 s to 144.441 s (**9.79%**). Its measured topology time is 122.846 s versus 129.722 s; the variation does not imply batching makes topology intrinsically slower, because each condition builds the same 400 chunks and run conditions can vary.
- The maximum absolute difference between a per-record wall timer and the sum of its timed stages is 0.0034 s. All 64 records passed one-vote coverage.

## Interpretation and decision boundary

On this controlled eight-building subset, **CPU chunk/topology construction is the largest measured contributor** to S8's slowdown. Batch size matters for the forward stage but has a modest effect on total S8 time because topology is built once per chunk in either batch configuration. The present profile does not isolate which topology suboperation consumes the time.

The original 4.5075× full-validation ratio used all 187 buildings, three votes, and the preregistered S7 batch-4 versus S8 batch-1 evaluators in separate runs. The 3.477× matched-batch subset result does not replace that full-validation measurement or reopen its latency gate. The separately measured S8 accuracy regression also remains unchanged. The S8 internal promotion gate therefore **remains failed**, with no test-set evaluation or headline architecture claim authorized.

For project delivery, retain S7 as the validated backbone. If S8 latency remains a research priority, the next prospective measurement should split the CPU chunk/topology timer into feature selection/normalization, each anchor selection, each support build, each kNN build, and each row-softmax transport call on fixed chunks. Any optimization must preserve the topology and output contracts, be registered before a new validation run, and still cannot address the independent accuracy failure by itself.
