import pandas as pd

try:
    import tomllib as _
except ModuleNotFoundError:
    # iguane imports tomli from pip which might not be installed. To avoid
    # import exceptions, add our custom tomllib which falls back to directly
    # import tomli
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).parent / "faketomllib"))
    import tomllib as _

from iguane.fom import RAWDATA, FIELDS

MAPPING_IGUANE_MILA = {
    "A100-SXM4-40GB": "NVIDIA-A100-SXM4-40GB",
    "A100-SXM4-80GB": "NVIDIA-A100-SXM4-80GB",
    "H100-SXM5-80GB": "NVIDIA-H100-80GB-HBM3",
    "L40S": "NVIDIA-L40S",
    "RTX8000": "Quadro-RTX-8000",
    "V100-SXM2-32GB": "Tesla-V100-SXM2-32GB",
}

GPU_DATA = {
    MAPPING_IGUANE_MILA[gpu]: RAWDATA[gpu]
    for gpu in sorted(RAWDATA)
    if gpu in MAPPING_IGUANE_MILA
}

# Filter out non-numerical fields
GPU_DATA = pd.DataFrame(GPU_DATA).T[FIELDS]

GPU_ALIASES = {
    "NVIDIA-A100-SXM4-40GB": [],
    "NVIDIA-A100-SXM4-80GB": [],
    "NVIDIA-H100-80GB-HBM3": [],
    "NVIDIA-L40S": [],
    "Quadro-RTX-8000": [],
    "Tesla-V100-SXM2-32GB": ["Tesla-V100-SXM2-32GB-LS"],
}

REF_GPU = "NVIDIA-A100-SXM4-80GB"

NORMALIZED_GPU_DATA = GPU_DATA / GPU_DATA.loc[REF_GPU]
