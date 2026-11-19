import os
from pathlib import Path

# Override with DATA_ROOT=/some/local/copy when running off IDUN.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/cluster/projects/vc/courses/TDT17/2026/1_infra"))

POTHOLE_MIX = DATA_ROOT / "Pothole-Mix-v2/kfth5g2xk3-2/Pothole-Mix/pothole-mix"
NRD = DATA_ROOT / "NorwegianRoadDamage"
RDD2022 = DATA_ROOT / "RDD2022/RDD2022"
