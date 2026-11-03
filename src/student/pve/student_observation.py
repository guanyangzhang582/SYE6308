"""Student-editable image input. Default: four 192 x 168 grayscale frames.

Only transform the supplied RGB image. Never operate the game or change rewards.
The coordinate branch still uses the six original self-first position values.
SB3 divides these uint8 images by 255 once, inside its policy preprocessor.
"""
import cv2
import numpy as np

INPUT_CONFIG = {'color': 'gray', 'width': 192, 'height': 168, 'stack_frames': 4, 'raw_shape': [224, 320, 3], 'crop_xyxy': [0, 0, 256, 224], 'verify_right_padding': True, 'black_threshold': 5, 'max_nonblack_fraction': 0.005}


def preprocess(rgb):
    """One RGB uint8 frame -> (channels, height, width) uint8 array."""
    a = np.asarray(rgb)
    c = INPUT_CONFIG
    if a.dtype != np.uint8 or tuple(a.shape) != tuple(c["raw_shape"]):
        raise ValueError(f"Unexpected raw image: {a.shape} {a.dtype}; expected {c['raw_shape']} uint8.")
    x0, y0, x1, y1 = c["crop_xyxy"]
    if c["verify_right_padding"] and x1 < a.shape[1]:
        excluded = a[:, x1:, :]
        fraction = float(np.mean(np.max(excluded, axis=2) > c["black_threshold"]))
        if fraction > c["max_nonblack_fraction"]:
            raise ValueError(
                f"CROP_GUARD_FAILED: right-side nonblack fraction={fraction:.6f}. "
                "The fixed viewport is not valid for this observed frame. "
                "Stop and inspect the raw image; do not train with a silently changing crop."
            )
    view = a[y0:y1, x0:x1, :]
    if c["color"] == "gray":
        gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
        result = cv2.resize(gray, (c["width"], c["height"]), interpolation=cv2.INTER_AREA)[None, :, :]
    else:
        result = cv2.resize(view, (c["width"], c["height"]), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    return np.ascontiguousarray(result, dtype=np.uint8)
