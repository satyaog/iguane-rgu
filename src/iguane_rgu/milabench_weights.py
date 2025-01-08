from pathlib import Path
import yaml

with Path(__file__).with_suffix(".yaml").open() as _f:
    WEIGHTS = yaml.safe_load(_f)

for bench in list(WEIGHTS.keys()):
    if "weight" not in WEIGHTS[bench] or not WEIGHTS[bench]["enabled"]:
        del WEIGHTS[bench]
        continue
    WEIGHTS[bench] = WEIGHTS[bench]["weight"]
