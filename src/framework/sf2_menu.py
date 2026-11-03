
"""Repair OPTIONS detection in the existing no-state SF2 experiment.

This module uses a tiny text-mask and star-row layout measured from the user's
actual 2026-09-26 startup video. It does not contain a ROM or game-state snapshot.
No OCR, no new dependency, no RAM value writes, no weakened readiness checks.
Integrated menu helper. Existing character, CPU and round guards are retained.
Rule-menu recognition below is new and requires a real Kaggle regression run.

The source of the two address candidates is the original 0xFFFE45 definition
and its neighbouring byte. Neither is declared correct before observing
reversible menu changes AND agreement with the visible difficulty stars.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import numpy as np

VERSION = "options-visual-readback-fix-1"
# 102 x 16 binary mask, x=76..177, y=12..27 of the native 320x224 image.
# Captured from the visible OPTION MODE text, not a complete game image.
_HEADER_B64 = (
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AB4+Px4eM4Aw3j4+AADMbLQwzGwAZsxsbAADMbDAwzGwAZsxsZAADMbDAwzHwAfsxsYAADMeDAwzHw"
    "AfsxseAADMYDAwzHwAZsxsYAADMYDAwzGwAZsxsZAADMYDAwzGwAZsxsbAAB48Hh4eOYA53j4+AAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
)
_HEADER = np.unpackbits(np.frombuffer(base64.b64decode(_HEADER_B64), dtype=np.uint8))[:1632].reshape(16,102).astype(bool)
_CANDIDATES = {
    "byte_FFFE44": {"address": 0xFFFE44, "type": "|u1"},
    "byte_FFFE45": {"address": 0xFFFE45, "type": "|u1"},
}


def _colour_mask(rgb):
    a = np.asarray(rgb, dtype=np.int16)
    return (a.max(axis=-1) >= 105) & (a.max(axis=-1)-a.min(axis=-1) >= 55)


def visible_options_stars(rgb):
    """Return 1..8 only when this exact OPTIONS header and star row are legible.

    This is limited to the verified 320x224 menu layout. Unknown layouts,
    fades, ambiguous star colours and different screens return None.
    """
    rgb = np.asarray(rgb)
    if rgb.shape != (224,320,3):
        return None
    mask = _colour_mask(rgb[12:28,76:178])
    union = int(np.count_nonzero(mask | _HEADER))
    score = np.count_nonzero(mask & _HEADER) / max(1, union)
    if score < 0.88:
        return None
    counts = [int(_colour_mask(rgb[48:57,136+8*i:144+8*i]).sum()) for i in range(8)]
    if any(5 < x < 12 for x in counts):
        return None
    active = [x >= 12 for x in counts]
    n = sum(active)
    if not 1 <= n <= 8 or active != [True]*n+[False]*(8-n):
        return None
    return n


def _register_read_candidates(boot):
    before = np.asarray(boot.engine.get_ram()).copy()
    for name, spec in _CANDIDATES.items():
        boot.engine.data.set_variable("options_fix_"+name, dict(spec))
    if not np.array_equal(before, np.asarray(boot.engine.get_ram())):
        raise RuntimeError("Registering read definitions unexpectedly changed RAM.")


def _wait_for_options(boot, max_frames=360):
    """Wait for a visible, stable menu; NEVER use START as a retry here."""
    previous = None
    consecutive = 0
    for i in range(max_frames+1):
        n = visible_options_stars(boot.rgb)
        consecutive = consecutive+1 if n is not None and n == previous else (1 if n is not None else 0)
        previous = n
        if consecutive >= 4:
            return n
        if i < max_frames:
            boot.advance(1)
    boot.shot("options_fix_no_legible_settings_page")
    raise RuntimeError("OPTIONS header/stars not recognized. Kept the real screen; no blind START retry.")


def _sample(boot, label):
    stars = _wait_for_options(boot)
    candidates = {name: int(boot.engine.data.lookup_value("options_fix_"+name)) for name in _CANDIDATES}
    raw = np.asarray(boot.engine.get_ram()).reshape(-1)
    row = {
        "label": label, "frame": int(boot.frame), "visible_stars": int(stars),
        "expected_menu_value": int(stars-1), "old_or_current_menu_read": int(boot.values["difficulty_menu"]),
        **candidates,
        # Flat-array indices are recorded separately; never silently equated
        # to GameData's bus addresses or used to select a RAM writer.
        "flat_ram_length": int(raw.size),
        "flat_index_FE44": int(raw[0xFE44]) if raw.size > 0xFE44 else None,
        "flat_index_FE45": int(raw[0xFE45]) if raw.size > 0xFE45 else None,
    }
    boot.report["options_fix"]["readback_samples"].append(row)
    Path(boot.root, "options_fix_readbacks.json").write_text(
        json.dumps(boot.report["options_fix"], indent=2), encoding="utf-8")
    boot.shot("options_fix_"+label)
    print(f"[OPTIONS] {label}: visible={stars} stars | menu_read={row['old_or_current_menu_read']} | "
          f"FFFE44={candidates['byte_FFFE44']} FFFE45={candidates['byte_FFFE45']}", flush=True)
    return row


def _tap_and_check(boot, key, before, label):
    expected = max(1, before-1) if key == "LEFT" else min(8, before+1)
    boot.tap(1, key, hold=8, release=48)
    row = _sample(boot, label)
    if row["visible_stars"] != expected:
        raise RuntimeError(f"Visible difficulty did not respond to {key}: expected {expected} stars, "
                           f"observed {row['visible_stars']}. No automatic acceptance.")
    return row


def set_difficulty_via_menu_fixed(boot):
    boot.phase = "boot_and_options"
    boot.report["options_fix"] = {"version": VERSION, "readback_samples": [],
        "status": "STARTED", "template_source": "user startup video, OPTION MODE around 35.13 seconds",
        "ram_value_writes": 0, "corrected_address": None}
    # The user's video confirms this navigation reaches OPTIONS. The old
    # 68-frame wait started testing during a fade, before the settings appeared.
    boot.advance(boot.config.boot_idle_frames)
    boot.shot("title_after_boot")
    boot.tap(1,"START",release=128)
    boot.tap(1,"DOWN")
    boot.tap(1,"DOWN")
    boot.tap(1,"START",release=240)
    _wait_for_options(boot)
    _register_read_candidates(boot)
    first = _sample(boot,"baseline")
    samples = [first]
    n = first["visible_stars"]
    # Reversible menu experiment; all actual changes come from gamepad keys.
    for i,key in enumerate(("LEFT","RIGHT","RIGHT","LEFT"),1):
        row = _tap_and_check(boot,key,n,f"calibration_{i}_{key.lower()}")
        samples.append(row)
        n = row["visible_stars"]
    boot.advance(60)
    idle = _sample(boot,"idle_stability")
    samples.append(idle)
    if idle["visible_stars"] != n or len({r["visible_stars"] for r in samples}) < 2:
        raise RuntimeError("Difficulty not stable or no reversible variation was recorded.")
    matches = [name for name in _CANDIDATES
               if all(r[name] == r["visible_stars"]-1 for r in samples)]
    if len(matches) != 1:
        boot.report["options_fix"]["matching_candidates"] = matches
        raise RuntimeError(f"Difficulty read mapping is ambiguous or absent: {matches}. "
                           "See options_fix_readbacks.json; no guess was installed.")
    chosen = matches[0]
    spec = _CANDIDATES[chosen]
    # Rebind HOW the variable is read. Do not modify any RAM contents.
    before = np.asarray(boot.engine.get_ram()).copy()
    boot.engine.data.set_variable("boot_difficulty_menu",dict(spec))
    if not np.array_equal(before,np.asarray(boot.engine.get_ram())):
        raise RuntimeError("Read-definition rebinding unexpectedly changed RAM.")
    boot.values = boot.read()
    boot.report["options_fix"].update(corrected_address=hex(spec["address"]),dtype=spec["type"],
                                     chosen_candidate=chosen)
    target = boot.config.difficulty_stars
    for i in range(8):
        if n == target:
            break
        key = "LEFT" if n > target else "RIGHT"
        row = _tap_and_check(boot,key,n,f"set_target_{i+1}")
        n = row["visible_stars"]
        if boot.values["difficulty_menu"] != n-1:
            raise RuntimeError("Selected read mapping stopped matching the visible stars.")
    boot.advance(60)
    final = _sample(boot,"target_set")
    if final["visible_stars"] != target or boot.values["difficulty_menu"] != target-1:
        raise RuntimeError("Final difficulty differs between the screen, readback and requested setting.")
    boot.report["difficulty_set_via_menu"] = boot.values["difficulty_menu"]
    boot.report["options_fix"]["status"] = "MENU_DIFFICULTY_VISUAL_AND_READBACK_MATCH"
    Path(boot.root,"options_fix_readbacks.json").write_text(
        json.dumps(boot.report["options_fix"],indent=2),encoding="utf-8")
    boot.shot("options_difficulty_set")
    print(f"OPTIONS FIX CHECK PASSED: {target} visible stars; read mapping {hex(spec['address'])}.",flush=True)
    # Exit only after successful visual AND numeric agreement. This is the
    # first START sent after entering the settings screen.
    boot.tap(1,"START",release=120)



# Text masks measured from the user's real startup video (30.0s / 41.0s).
MENU_TEXT_B64 = {'rules': 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHz/GMN8PHzHAAAMZmGGZmGMZmAAAMZmPH5mGMZ2AAAMB+JH5mGMZ+AAAMZmfmZ8GMZuAAAMZmQmZgGMZmAAAHz/5+fwPHzjAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/nfHx8AAAQEBAGZmZmJmAAAQEBAGY8ZmBmAAD+/v4H4YZnh8AAB8fHwGYYfGBmAAA4ODgGYYYGJmAAB8fHwP888Pz2AABEREQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHx8fjx8x3wAAAAMZmmRjGZsYAAAAMZmGBjGdsAAAAAMZmGBjGfnwAAAAMZ8GBjGbgYAAAAMZgGBjGZsYAAAAHzwPDx843wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', 'game': 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHwa03wAfH44fH4MYaZmIAxpkYZpkN48fmAAwBg9ZngMgkfngAfRhkfBgM5+ZuAABhh+ZhgNZCZmIAxtlCZhgH7n5/wAfD3n9nwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAE4AfAAAfBh+fvBMQAxgAAZpiZmWBGwAwAAAZjwYeGBCgAfAAAfKQYGGBLgABgAAZn4a2GBFAwxjAAZkIYGGJBAwfDAA/Oc8PPzICAAIAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH18fO98AH2Yfn7MZmxmZmAGaYmZlN5mxmZmAGY8GBhNh9xuZmAPwkGBhM5mxm58AGZ+GppMZmxmZgAGZCGBhH72fDzwAPznfHzAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'}

_MENU_MASKS = {k:np.unpackbits(np.frombuffer(base64.b64decode(v),dtype=np.uint8))[:48*84].reshape(48,84).astype(bool) for k,v in MENU_TEXT_B64.items()}


def menu_kind(rgb):
    if np.asarray(rgb).shape != (224,320,3):
        return None
    actual = _colour_mask(rgb[164:212,86:170])
    scores={k:np.count_nonzero(actual & v)/max(1,np.count_nonzero(actual | v)) for k,v in _MENU_MASKS.items()}
    best=max(scores,key=scores.get)
    # Exclude the HYPER star area for rule matching; speed is checked separately.
    rule_a=actual[:,:56];rule_t=_MENU_MASKS['rules'][:,:56]
    rule_score=np.count_nonzero(rule_a & rule_t)/max(1,np.count_nonzero(rule_a | rule_t))
    if rule_score>.82:
        return 'rules'
    return 'game' if scores['game']>.82 else None


def cursor_row(rgb):
    counts=[int(_colour_mask(rgb[164+16*i:180+16*i,67:84]).sum()) for i in range(3)]
    on=[i for i,n in enumerate(counts) if n>=12]
    return on[0] if len(on)==1 else None


def visible_speed_stars(rgb):
    if menu_kind(rgb)!='rules':
        return None
    counts=[int(_colour_mask(rgb[184:193,144+8*i:152+8*i]).sum()) for i in range(10)]
    if any(5<n<12 for n in counts):
        return None
    bits=[n>=12 for n in counts];n=sum(bits)
    return n if bits==[True]*n+[False]*(10-n) else None


def _seek_menu(boot, wanted, allow_start, budget=2200):
    for i in range(budget):
        if menu_kind(boot.rgb)==wanted and cursor_row(boot.rgb) is not None:
            boot.advance(4)
            if menu_kind(boot.rgb)==wanted and cursor_row(boot.rgb) is not None:
                return
        # START only while seeking the title/menu from intro; NEVER use directions on unknown pages.
        if allow_start and i and i%240==0:
            boot.tap(1,'START',release=60)
        else:
            boot.advance(1)
    boot.shot('unrecognized_'+wanted+'_menu')
    raise RuntimeError('Expected '+wanted+' menu not recognized. Inspect actual startup screenshots.')


def _select_row(boot,wanted,row):
    for _ in range(4):
        if menu_kind(boot.rgb)!=wanted:
            raise RuntimeError('Menu changed before row selection.')
        current=cursor_row(boot.rgb)
        if current==row:return
        if current is None:
            boot.advance(8);continue
        boot.tap(1,'DOWN' if current<row else 'UP',release=32)
    raise RuntimeError('Cursor did not reach requested menu row.')


def select_rules_and_game_start(boot):
    """Explicitly choose rule set/speed; no character-probe keys on mode pages.

    Menu templates were replay-tested against the user's old video. The full
    new navigation path is NOT yet certified on a real emulator here.
    """
    boot.phase='select_rules_and_speed'
    _seek_menu(boot,'rules',allow_start=True)
    wanted=0 if boot.config.ruleset=='champion' else 1
    _select_row(boot,'rules',wanted)
    if wanted==1:
        for _ in range(8):
            n=visible_speed_stars(boot.rgb)
            if n is None:boot.advance(8);continue
            if n==boot.config.speed_stars:break
            boot.tap(1,'LEFT' if n>boot.config.speed_stars else 'RIGHT',release=32)
        if visible_speed_stars(boot.rgb)!=boot.config.speed_stars:
            boot.shot('speed_not_verified')
            raise RuntimeError('HYPER speed stars not equal to requested setting.')
    boot.shot('rules_and_speed_selected')
    boot.report['ruleset_selection']={'ruleset':boot.config.ruleset,'speed_stars':boot.config.speed_stars if wanted else None,'method':'actual rule-menu text + cursor + stars','visual_menu_checked':True,'combat_ruleset_ram_certified':False}
    boot.tap(1,'START',release=90)
    _seek_menu(boot,'game',allow_start=False,budget=720)
    _select_row(boot,'game',0)
    boot.shot('native_game_start_selected')
    boot.tap(1,'START',release=120)
