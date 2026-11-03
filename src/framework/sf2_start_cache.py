"""Verified match-start caching for the existing SF2 Student Lab.

Only emulator state is restored. Python/NumPy/Torch RNGs, model parameters,
optimizers and the learner's RNG are NOT serialized or reset by this module.
The classroom environment applies a fresh neutral offset and resets image
history, reward totals and terminal bookkeeping. No long-action queue is used.

A cache is an automatically generated emulator savestate, not a new CPU seed.
Same complete start + same input sequence may replay exactly the same fight.
No addresses are guessed or written to randomize the native CPU.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import zipfile

import numpy as np
from sf2_boot import ColdBootMatch, GAME, ROM_SHA1, RAM_FIELDS, bcd_seconds, validate_character_ids

CACHE_VERSION = 'sf2-classroom-verified-start-v2'
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
_MEMO = {}
_FAMILY = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def array_digest(array):
    a = np.ascontiguousarray(array)
    return digest(str((a.shape, a.dtype.str)).encode() + a.tobytes())


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def _json(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def _write_json(path, data):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def enabled():
    return True  # Integrated runtime; no notebook monkey-patch or enable flag required.


def cache_root():
    configured = os.environ.get('SF2_START_CACHE_DIR')
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / 'cache' / 'match_starts_v2'


def _family(backend):
    """Invalidate for changed emulator binaries, relevant source or ABI."""
    key = (str(Path(backend.__file__).resolve()), CACHE_VERSION)
    if key not in _FAMILY:
        root = Path(backend.__file__).resolve().parent
        paths = {Path(backend.__file__).resolve()}
        for p in root.rglob('*'):
            if not p.is_file():
                continue
            if p.suffix.lower() in ('.so', '.pyd', '.dll', '.dylib'):
                # Stable-retro's Python binding and the Genesis core.
                if p.parent == root or 'genesis' in p.name.lower():
                    paths.add(p)
        version = getattr(backend, '__version__', None)
        try:
            version = importlib.metadata.version('stable-retro')
        except importlib.metadata.PackageNotFoundError:
            pass
        if not version:
            raise RuntimeError('Cannot identify stable-retro version; refusing an unversioned state cache.')
        src = Path(__file__).resolve().parent
        _FAMILY[key] = {
            'cache_version': CACHE_VERSION,
            'game': GAME, 'rom_sha1': ROM_SHA1,
            'backend_version': str(version),
            'python': platform.python_version(), 'machine': platform.machine(),
            'backend_binaries': {str(p.relative_to(root)): file_digest(p) for p in sorted(paths)},
            'source_hashes': {name: file_digest(src / name) for name in
                              ('sf2_boot.py', 'sf2_menu.py', 'sf2_start_cache.py')},
        }
    return _FAMILY[key]


def descriptor(boot):
    # Deliberately not based on the learned network or reward function.
    # Include the full bootstrap configuration so future setup fields cannot
    # accidentally reuse an incompatible cached match.
    return {'family': _family(boot.backend), 'match_config': asdict(boot.config)}


def _cache_key(desc):
    return digest(_json(desc).encode())


@contextmanager
def _lock(path, timeout=180):
    """Kaggle/Linux interprocess lock; no global ML random-number consumption."""
    import fcntl
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a+b') as f:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Timed out waiting for cache builder: {p}')
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read_definitions(boot):
    definitions = {name: {'address': addr, 'type': dtype}
                   for name, (addr, dtype) in RAM_FIELDS.items()}
    calibrated = boot.report.get('options_fix', {})
    addr = calibrated.get('corrected_address')
    if addr is not None:
        definitions['difficulty_menu'] = {
            'address': int(addr, 0) if isinstance(addr, str) else int(addr),
            'type': calibrated.get('dtype', '|u1'),
        }
    return definitions


def _snapshot(boot, desc):
    """Capture at exactly the existing start() return point, BEFORE jitter."""
    if not boot.battle_started or boot.stage_write_active or boot.phase != 'battle':
        raise RuntimeError('Cache capture requires the verified post-initialization battle point.')
    if boot.trace and (boot.trace[-1]['p1_buttons'] or boot.trace[-1]['p2_buttons']):
        raise RuntimeError('Cached starts must have released controller buttons.')
    if getattr(boot.engine, 'movie', None):
        raise RuntimeError('BK2 recording is not supported while validating a start cache.')
    state = bytes(boot.engine.em.get_state())
    if not state or len(state) > MAX_PAYLOAD_BYTES:
        raise RuntimeError('Invalid emulator snapshot size.')
    rgb = np.asarray(boot.rgb, dtype=np.uint8).copy()
    ram = np.asarray(boot.engine.get_ram(), dtype=np.uint8).copy()
    meta = {
        'descriptor': desc,
        'capture_point': 'existing ColdBootMatch.start return, before neutral offset',
        'state_sha256': digest(state), 'rgb_sha256': array_digest(rgb),
        'ram_sha256': array_digest(ram), 'readings': dict(boot.values),
        'read_definitions': _read_definitions(boot),
        'frame': int(boot.frame), 'fps': float(boot.fps), 'buttons': list(boot.buttons),
        'source_output_dir': str(boot.root),
        'source_bootstrap_status': boot.report.get('status'),
        'source_options_check': boot.report.get('options_fix'),
        'native_cpu_fresh_random_seed_claimed': False,
        'rng_reset_scope': 'emulator snapshot only; NOT Python/NumPy/Torch/model RNG',
    }
    _validate_start(boot)
    return {'state': state, 'rgb': rgb, 'ram': ram, 'meta': meta}


def _validate_start(boot):
    raw = boot.values
    validate_character_ids(raw, boot.config.p1_character, boot.config.p2_character)
    seconds = bcd_seconds(raw['timer_bcd'])
    if seconds is None or not 0 < seconds < 99:
        raise RuntimeError('Cached start must be at the original running-countdown checkpoint.')
    if any(not 0 <= raw[f'p{p}_hp'] <= 176 for p in (1, 2)):
        raise RuntimeError('Cached start contains invalid/KO health.')
    if raw['difficulty_menu'] != boot.config.difficulty_stars - 1:
        raise RuntimeError('Cached menu difficulty mismatch.')
    if boot.config.mode == 'pve' and raw['difficulty_fight'] != boot.config.difficulty_stars - 1:
        raise RuntimeError('Cached CPU difficulty mismatch.')
    if any(raw[f'p{p}_wins'] != 0 for p in (1, 2)):
        raise RuntimeError('Not the original first-round starting checkpoint.')


def _restore_exact(boot, payload):
    """Restore WITHOUT RetroEnv.reset's extra step and without resetting any ML RNG.

    The first image is the actual RGB captured with the snapshot: framebuffer
    contents are not necessarily serialized by every emulator core. Subsequent
    images come from actual engine.step calls; deterministic replay is verified.
    """
    m = payload['meta']; engine = boot.engine
    if abs(float(boot.fps) - float(m['fps'])) > 1e-6:
        raise RuntimeError('Emulator frame rate differs from the cached start.')
    if list(boot.buttons) != m['buttons']:
        raise RuntimeError('Controller layout differs from cached layout.')
    if getattr(engine, 'movie', None):
        raise RuntimeError('Cannot restore while recording a BK2 movie.')
    engine.em.set_state(payload['state'])
    for p in range(engine.players):
        engine.em.set_button_mask(np.zeros(engine.num_buttons, dtype=np.uint8), p)
    engine.data.reset()
    for name, spec in m['read_definitions'].items():
        engine.data.set_variable('boot_' + name, dict(spec))  # Definitions only, not RAM values.
    engine.data.update_ram()
    got_ram = np.asarray(engine.get_ram(), dtype=np.uint8)
    if not np.array_equal(got_ram, payload['ram']):
        raise RuntimeError('Full RAM readback differs after restoring match-start snapshot.')
    boot.values = boot.read()
    if boot.values != m['readings']:
        raise RuntimeError('Cached character/position/HP/timer readback mismatch.')
    boot.rgb = payload['rgb'].copy()
    engine.img = boot.rgb.copy()
    engine.ram = None
    boot.frame = int(m['frame'])
    boot.phase = 'battle'; boot.battle_started = True; boot.stage_write_active = False
    boot.trace.clear()
    _validate_start(boot)


def _probe_trace(boot, frames=96):
    """No model involved. Exact same nonrandom controller sequence for replay QC."""
    old_callback = boot.frame_callback; boot.frame_callback = None
    h = hashlib.sha256()
    phase1 = ((), ('RIGHT',), ('DOWN', 'B'), (), ('UP', 'LEFT'), (), ('C',), ())
    phase2 = ((), ('LEFT',), ('Y',), (), ('DOWN',), (), ('C',), ())
    try:
        for i in range(frames):
            slot = (i // 6) % len(phase1)
            k2 = phase2[slot] if boot.config.mode == 'pvp' else ()
            raw = boot.frame_step(phase1[slot], k2)
            h.update(_json(raw).encode())
            h.update(np.ascontiguousarray(boot.engine.get_ram()).tobytes())
            h.update(np.ascontiguousarray(boot.rgb).tobytes())
    finally:
        boot.frame_callback = old_callback
    return h.hexdigest()


def _check_payload(p, desc):
    m = p['meta']
    if m.get('descriptor') != desc:
        raise RuntimeError('Cache identity/configuration mismatch.')
    if digest(p['state']) != m.get('state_sha256'):
        raise RuntimeError('State cache checksum mismatch.')
    if array_digest(p['rgb']) != m.get('rgb_sha256') or array_digest(p['ram']) != m.get('ram_sha256'):
        raise RuntimeError('Cached observation/RAM checksum mismatch.')
    if not m.get('roundtrip_verified') or not m.get('probe_sha256'):
        raise RuntimeError('Cache was not verified by the builder.')


def _save(path, p):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.stem + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, state=np.frombuffer(p['state'], dtype=np.uint8),
                                rgb=p['rgb'], ram=p['ram'],
                                metadata=np.frombuffer(_json(p['meta']).encode('utf-8'), dtype=np.uint8))
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def _load(path, desc):
    if path.stat().st_size > MAX_PAYLOAD_BYTES:
        raise RuntimeError('Cache file exceeds safety size limit.')
    with zipfile.ZipFile(path) as z:
        if sum(x.file_size for x in z.infolist()) > MAX_PAYLOAD_BYTES:
            raise RuntimeError('Uncompressed state cache exceeds safety size limit.')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'state', 'rgb', 'ram', 'metadata'}:
            raise RuntimeError('Unexpected cache archive fields.')
        if any(data[k].dtype != np.uint8 for k in data.files):
            raise RuntimeError('Unexpected state-cache array type.')
        p = {'state': data['state'].tobytes(), 'rgb': data['rgb'].copy(),
             'ram': data['ram'].copy(), 'meta': json.loads(data['metadata'].tobytes().decode())}
    _check_payload(p, desc)
    return p


class CachedColdBootMatch(ColdBootMatch):
    """Same public start interface; only replaces redundant native initialization."""
    def start(self):
        if not enabled():
            t = time.perf_counter()
            result = super().start()
            self.cache_info = {'status': 'DISABLED_COLD_BOOT', 'seconds': time.perf_counter() - t}
            return result
        t = time.perf_counter(); status = 'ERROR'; path = None
        try:
            desc = descriptor(self); key = _cache_key(desc)
            path = cache_root() / (key + '.npz')
            cache_root().mkdir(parents=True, exist_ok=True)
            # Cached starts use the same implementation even for independent
            # evaluation processes. Every new emulator verifies its replay.
            with _lock(path.with_suffix('.lock')):
                payload = _MEMO.get(str(path))
                if payload is None and path.exists():
                    payload = _load(path, desc)
                if payload is None:
                    super().start()
                    payload = _snapshot(self, desc)
                    reference = _probe_trace(self)
                    _restore_exact(self, payload)
                    replay = _probe_trace(self)
                    if replay != reference:
                        raise RuntimeError('Snapshot restoration changed the 96-frame reference trajectory. Cache not saved.')
                    _restore_exact(self, payload)
                    payload['meta'].update(roundtrip_verified=True, probe_frames=96,
                                           probe_sha256=reference)
                    _save(path, payload)
                    status = 'BUILT_AND_REPLAY_VERIFIED'
                else:
                    _check_payload(payload, desc)
                    self._create()  # Creates emulator and definitions, but skips the menu route.
                    _restore_exact(self, payload)
                    replay = _probe_trace(self, payload['meta']['probe_frames'])
                    if replay != payload['meta']['probe_sha256']:
                        raise RuntimeError('Cached start does not replay identically in this emulator. Stop; disable cache to inspect.')
                    _restore_exact(self, payload)
                    status = 'CACHE_HIT_REPLAY_VERIFIED'
                _MEMO[str(path)] = payload
            self.cache_info = {
                'status': status, 'key': key, 'path': str(path),
                'seconds': time.perf_counter() - t,
                'initial_readings': dict(self.values),
                'capture_point': payload['meta']['capture_point'],
                'snapshot_rng_restored': 'game state only, including any serialized native PRNG state',
                'learner_rng_reseeded': False, 'new_native_cpu_seed_claimed': False,
                'first_image_source': 'actual RGB captured at original start checkpoint',
                'post_restore_replay_frames': payload['meta']['probe_frames'],
            }
            # Do not claim that a hit navigated menus or did no state restore.
            self.report.update(status='CACHED_START_CHECKS_PASSED',
                               external_state_files_used=True, in_memory_snapshots_used=True,
                               automatic_match_start_cache=True, cache=self.cache_info,
                               all_setup_writes_stopped=True,
                               source_bootstrap=payload['meta']['source_output_dir'],
                               note='Cache restores a game start, not a new independently seeded CPU.')
            _write_json(self.root / 'cache_reset.json', self.cache_info)
            print('[START CACHE]', status, f'{self.cache_info["seconds"]:.3f}s', flush=True)
            return self.rgb.copy(), dict(self.values)
        except Exception as exc:
            self.stage_write_active = False
            self.report.update(status='ERROR', error=repr(exc), cache_file=str(path),
                               note='No silent fallback. Run the explicit environment check or inspect this cache before continuing.')
            _write_json(self.root / 'cache_error.json', self.report)
            raise
        finally:
            self.flush()
