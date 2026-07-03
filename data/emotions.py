"""Canonical 7-emotion label space shared by FER2013 and RAF-DB.

We adopt the FER2013 ordering as canonical and remap RAF-DB into it so a single
detector head (nc=7) serves both datasets, as the manuscript's Table VIII
requires (same variant evaluated on FER2013 and RAF-DB).
"""

# Canonical order (index -> name).
CANON = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NAME_TO_ID = {n: i for i, n in enumerate(CANON)}

# FER2013 native ids already match CANON:
#   0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral
FER2013_TO_CANON = {i: i for i in range(7)}

# RAF-DB "basic" labels are 1..7 with a *different* order:
#   1 Surprise, 2 Fear, 3 Disgust, 4 Happiness, 5 Sadness, 6 Anger, 7 Neutral
RAFDB_TO_CANON = {
    1: NAME_TO_ID["surprise"],
    2: NAME_TO_ID["fear"],
    3: NAME_TO_ID["disgust"],
    4: NAME_TO_ID["happy"],
    5: NAME_TO_ID["sad"],
    6: NAME_TO_ID["angry"],
    7: NAME_TO_ID["neutral"],
}


def data_yaml_dict(root: str):
    """Return the ultralytics data.yaml content for a prepared dataset root."""
    return {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CANON),
        "names": {i: n for i, n in enumerate(CANON)},
    }
