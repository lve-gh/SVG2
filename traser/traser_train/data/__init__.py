"""Dataset registry.

Annotation files are resolved relative to the repository root (``<repo>/data``) so no
absolute paths are baked into the code. Place (or symlink) the annotation JSONs there;
see ``docs/DATA.md`` for the data preparation instructions.
"""

import os
import re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_ROOT = os.path.join(_REPO_ROOT, "data")


def _annotation(filename):
    return {
        "annotation_path": os.path.join(_DATA_ROOT, filename),
        "data_path": "",
    }


data_dict = {
    # SVG2 (objects, attributes, timestamped relationships), converted from
    # https://huggingface.co/datasets/UWGZQ/Synthetic_Visual_Genome2
    # by data/prepare_svg2.py. sav = SA-V videos, pvd = PE-Video videos.
    "svg2_sav": _annotation("svg2_sav.json"),
    "svg2_pvd": _annotation("svg2_pvd.json"),
    # Objects-only and objects+relationships datasets from public video benchmarks.
    "vipseg": _annotation("vipseg.json"),
    "vidor": _annotation("vidor.json"),
    "vidvrd": _annotation("vidvrd.json"),
    "lvvis": _annotation("lvvis.json"),
    "ovis": _annotation("ovis.json"),
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list
