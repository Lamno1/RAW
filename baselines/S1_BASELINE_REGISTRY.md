# S1 baseline registry

This registry separates public reference numbers from locally reproduced,
budget-matched results. They must never be merged into one experimental column.

| Baseline | Source | Status under 4 GB / Windows | Permitted use |
|---|---|---|---|
| Historical GeometryAwarePCT | local `model.py` | runner implemented; confirmatory seeds pending | corrected bridge control |
| CAMO P1 local operator | local `camo_build` | reproduced | local sparse point control |
| BuildingNet MinkowskiNet Res16UNet34C | official BuildingNet repository | official setup used 24 GB A5000 and MinkowskiEngine/Linux; not locally reproducible yet | public reference only |
| PTv3/serialization control | official implementation to be selected | dependency/config audit pending | cannot be claimed yet |

The official BuildingNet repository reports its point-cloud metrics separately
and documents a 24 GB training setup. A reduced configuration is a new local
experiment, not a reproduction of the published number.

S1 is not scientifically complete until three corrected bridge seeds finish and
at least one modern budget-matched control is either reproduced or explicitly
abandoned with a documented compatibility/hardware failure.
