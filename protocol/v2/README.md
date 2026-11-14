# SROT protocol v2 mutation

This directory is a prospective, non-retroactive branch. It does not modify
the S0/v1 freeze and does not erase the failed 1.25x latency result.

The new question is whether detached SROT fits an independently grounded
absolute hardware budget: at least 50,000 source points/s for the complete
forward + backward + optimizer model step, no more than 3,200 MiB allocated
VRAM, and no more than 216 projected GPU-hours for three 50-epoch seeds.
The 50,000 points/s requirement predates S5 and therefore is not fitted to the
observed SROT timings.

No SROT v2 segmentation training is authorized by this mutation alone. The
mathematical and hardware gates in `srot_latency_mutation.json` must first pass
and be captured in a new immutable benchmark artifact. Row-softmax S5 is an
independent main branch and is unaffected by this document.
