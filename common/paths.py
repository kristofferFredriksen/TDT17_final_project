import os
from pathlib import Path

# Priority: $DATA_ROOT, then a local copy in <repo>/data (gitignored), then the IDUN course dir.
_LOCAL = Path(__file__).resolve().parent.parent / "data"
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or (_LOCAL if _LOCAL.is_dir() else "/cluster/projects/vc/courses/TDT17/2026/1_infra"))

POTHOLE_MIX = DATA_ROOT / "Pothole-Mix-v2/kfth5g2xk3-2/Pothole-Mix/pothole-mix"
NRD = DATA_ROOT / "NorwegianRoadDamage"
RDD2022 = DATA_ROOT / "RDD2022/RDD2022"
