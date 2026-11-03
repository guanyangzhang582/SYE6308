"""Versioned, per-agent image preprocessing plus the original six positions.
Each exported agent carries its own exact student_observation.py source and config.
No control-state input, frame differences or extra geometry features.
"""
from collections import deque
from pathlib import Path
import hashlib
import json
import sys
import types
import numpy as np

_MODULES = {}


def _validate_config(c):
    required = {"color", "width", "height", "stack_frames", "raw_shape", "crop_xyxy",
                "verify_right_padding", "black_threshold", "max_nonblack_fraction"}
    if not isinstance(c, dict) or set(c) != required:
        raise ValueError("INPUT_CONFIG has missing or unsupported fields.")
    if c['color'] not in ('gray', 'rgb') or c['stack_frames'] != 4:
        raise ValueError("This upgrade supports four gray or four RGB frames.")
    if any(type(c[k]) is not int or not 64 <= c[k] <= 512 for k in ('height', 'width')):
        raise ValueError("Input height/width must be integer values between 64 and 512.")
    if c['raw_shape'] != [224, 320, 3]:
        raise ValueError("This game integration expects a 224 x 320 RGB emulator buffer.")
    box = c['crop_xyxy']
    if not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box):
        raise ValueError("crop_xyxy must contain four integers.")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 320 and 0 <= y0 < y1 <= 224):
        raise ValueError("Crop is outside the raw image.")
    if type(c['verify_right_padding']) is not bool:
        raise ValueError("verify_right_padding must be a bool.")
    if type(c['black_threshold']) is not int or not 0 <= c['black_threshold'] <= 255:
        raise ValueError("Invalid black threshold.")
    v = c['max_nonblack_fraction']
    if not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 0.01:
        raise ValueError("Invalid padding tolerance (must be 0..0.01).")
    return json.loads(json.dumps(c, allow_nan=False))


def _load_source(text):
    if not isinstance(text, str) or not 0 < len(text.encode('utf-8')) <= 256 * 1024:
        raise ValueError("Invalid preprocessing source size.")
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if digest not in _MODULES:
        name = 'sf2_observation_source_' + digest
        module = types.ModuleType(name)
        module.__file__ = '<student_observation:' + digest + '>'
        # Student files and external agent packages must already be trusted.
        # Content hashes detect drift; they do not sandbox arbitrary Python.
        exec(compile(text, module.__file__, 'exec'), module.__dict__)
        config = _validate_config(module.INPUT_CONFIG)
        if not callable(getattr(module, 'preprocess', None)):
            raise TypeError("student_observation.py must define preprocess(rgb).")
        _MODULES[digest] = (module, config)
    return digest, _MODULES[digest]


def build_descriptor(source_path):
    text = Path(source_path).read_text(encoding='utf-8')
    digest, (_, config) = _load_source(text)
    result = {'schema': 'sf2-student-observation-v1', 'source_sha256': digest,
              'config': config, 'source': text}
    validate_descriptor(result)
    return result


def validate_descriptor(d):
    if not isinstance(d, dict) or set(d) != {'schema', 'source_sha256', 'config', 'source'}:
        raise ValueError("Missing versioned observation descriptor; old input packages are not converted.")
    if d['schema'] != 'sf2-student-observation-v1':
        raise ValueError("Unsupported observation descriptor.")
    digest, (module, config) = _load_source(d['source'])
    if digest != d['source_sha256'] or config != d['config']:
        raise ValueError("Observation source/config/hash do not agree.")
    return module, config


def image_shape(spec):
    _, c = validate_descriptor(spec['observation'])
    channels = 1 if c['color'] == 'gray' else 3
    return (c['stack_frames'] * channels, c['height'], c['width'])


def description(spec):
    _, c = validate_descriptor(spec['observation'])
    return {'protocol': 'sf2-configurable-image4-position6-v3',
            'source_sha256': spec['observation']['source_sha256'],
            'config': c, 'image_shape': list(image_shape(spec)),
            'dtype': 'uint8', 'normalization': 'SB3 divides by 255 once',
            'channel_order': 'oldest to latest, then R/G/B within each RGB frame',
            'decision_frames': 6, 'position_count': 6 if spec.get('use_position', True) else 0}


def position(raw, player, scale=1024.0):
    if player not in (1, 2) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid player or position scale.')
    values = np.array([raw[f'p{player}_x'], raw[f'p{player}_y'],
                       raw[f'p{3-player}_x'], raw[f'p{3-player}_y']], np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 65535).any():
        raise ValueError('Invalid coordinate readback.')
    x, y, ox, oy = values
    return (np.array([x, y, ox, oy, ox-x, oy-y])/scale).astype(np.float32)


class Side:
    def __init__(self, spec, player, scale=1024.0):
        self.spec, self.player, self.scale = spec, player, scale
        self.module, self.config = validate_descriptor(spec['observation'])
        self.channels = 1 if self.config['color'] == 'gray' else 3
        self.frames = deque(maxlen=4)
        self.frame_ids = deque(maxlen=4)
        self.observed_frames_checked = 0

    def _frame(self, rgb):
        a = np.asarray(self.module.preprocess(rgb))
        expected = (self.channels, self.config['height'], self.config['width'])
        if a.dtype != np.uint8 or tuple(a.shape) != expected:
            raise ValueError(f"Preprocessor returned {a.shape} {a.dtype}; expected {expected} uint8.")
        self.observed_frames_checked += 1
        return np.ascontiguousarray(a).copy()

    def reset(self, rgb, frame_id=0):
        self.frames.clear(); self.frame_ids.clear()
        a = self._frame(rgb)
        for _ in range(4):
            self.frames.append(a.copy()); self.frame_ids.append(int(frame_id))

    def update(self, rgb, frame_id=None):
        if not self.frames:
            raise RuntimeError("Initialize the frame history before updating it.")
        n = self.frame_ids[-1]+6 if frame_id is None else int(frame_id)
        if n <= self.frame_ids[-1]:
            raise ValueError("Frame IDs must increase within an episode.")
        self.frames.append(self._frame(rgb)); self.frame_ids.append(n)

    def observe(self, raw):
        if len(self.frames) != 4:
            raise RuntimeError("Expected a complete four-frame history.")
        image = np.concatenate(tuple(self.frames), axis=0)
        if not self.spec.get('use_position', True):
            return image
        return {'image': image, 'position': position(raw, self.player, self.scale)}


def observation_space(use_position=True, scale=1024.0, *, spec):
    from gymnasium import spaces
    if bool(spec.get('use_position', True)) != bool(use_position):
        raise ValueError("Position flag disagrees with agent spec.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid coordinate scale.")
    image = spaces.Box(0, 255, image_shape(spec), np.uint8)
    if not use_position:
        return image
    bound = np.float32(65535/scale)
    return spaces.Dict({'image': image, 'position': spaces.Box(
        np.array([0, 0, 0, 0, -bound, -bound], np.float32),
        np.full(6, bound, np.float32), dtype=np.float32)})


def display_frames(image, spec):
    if tuple(image.shape) != image_shape(spec) or image.dtype != np.uint8:
        raise ValueError("Image tensor differs from its exported observation contract.")
    channels = 1 if spec['observation']['config']['color'] == 'gray' else 3
    return [image[i*channels:(i+1)*channels][0] if channels == 1
            else image[i*channels:(i+1)*channels].transpose(1, 2, 0) for i in range(4)]
