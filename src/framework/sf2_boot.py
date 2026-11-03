"""Cold-boot SF2 setup. No external or hidden emulator save states.

Native CPU source: GAME START (single-player arcade), never a Python bot.
Targeted PvE is EXPERIMENTAL: an initialization-only stage selector at
0xFF96AA (>u2) requests the opponent's stage. We DO NOT write p1/p2 character
IDs to make the identity checks pass. Stop with evidence if the native game
loads a different opponent. The user observed ChunLi vs CPU ChunLi with the prior
menu patch; other combinations and this edition's new rule-menu route need testing.

Menu timing/difficulty observations and character read mappings are adapted
from LEIA-qro/street_fighter (MIT), commit ea7545305f30a62dceb7c7b77b9d51466029bf7e.
Stage selector: Rune / BigBossman, GameHacking.org game 15969.
See SOURCES.md and THIRD_PARTY_LICENSES.txt. No ROM or model is distributed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections import deque
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import re
import time
from typing import Callable

import numpy as np

GAME = "StreetFighterIISpecialChampionEdition-Genesis"
ROM_SHA1 = "a5aad1d108046d9388e33247610dafb4c6516e0b"
CHARACTERS = ("Ryu", "EHonda", "Blanka", "Guile", "Ken", "ChunLi",
              "Zangief", "Dhalsim", "MBison", "Sagat", "Balrog", "Vega")
CHAR_IDS = {name: i for i, name in enumerate(CHARACTERS)}
# US character-select layout, as visible in the user's actual menu.
GRID = ((0, 1, 2, 3, 10, 11), (4, 5, 6, 7, 9, 8))
CELLS = {cid: (row, col) for row, values in enumerate(GRID) for col, cid in enumerate(values)}
RAM_FIELDS = {
    "p1_x": (0xFF8006, ">u2"), "p1_y": (0xFF800A, ">u2"),
    "p2_x": (0xFF8286, ">u2"), "p2_y": (0xFF828A, ">u2"),
    "p1_hp": (0xFF8042, ">i2"), "p2_hp": (0xFF82C2, ">i2"),
    "p1_char": (0xFF81DB, "|u1"), "p2_char": (0xFF845B, "|u1"),
    "p1_wins": (0xFF81DA, "|u1"), "p2_wins": (0xFF845A, "|u1"),
    "timer_bcd": (0xFF972A, "|u1"),
    "difficulty_menu": (0xFFFE44, "|u1"),
    "difficulty_fight": (0xFF97B3, "|u1"),
    "stage_selector": (0xFF96AA, ">u2"),
    # Read-only corroboration, not treated as definitive CPU proof.
    "p1_game_input": (0xFF81E2, "|u1"),
    "p2_game_input": (0xFF845E, "|u1"),
}


def normalize_character(value: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", str(value).lower())
    for name in CHARACTERS:
        if re.sub(r"[^a-z0-9]", "", name.lower()) == key:
            return name
    raise ValueError(f"Unknown character {value!r}. Choose one of {CHARACTERS}.")


def backend_module():
    try:
        return importlib.import_module("stable_retro")
    except ModuleNotFoundError as exc:
        if exc.name != "stable_retro":
            raise
        return importlib.import_module("retro")


def resolve_installed_game(backend) -> tuple[str, Path]:
    """Resolve the actual backend ID without altering public rules or ROM bytes."""
    mismatches = []
    for game_id in (GAME, GAME + "-v0"):
        for method in ("get_original_romfile_path", "get_romfile_path"):
            fn = getattr(backend.data, method, None)
            if fn is not None:
                try:
                    path = Path(fn(game_id))
                    if path.is_file():
                        checksum = hashlib.sha1(path.read_bytes()).hexdigest()
                        if checksum == ROM_SHA1:
                            return game_id, path
                        mismatches.append(f"{game_id}: {checksum}")
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass
    if mismatches:
        raise RuntimeError("Wrong ROM SHA1. Expected " + ROM_SHA1 + "; found "
                           + "; ".join(mismatches))
    raise FileNotFoundError("Matching ROM is not installed under " + GAME + " or "
                            + GAME + "-v0. Run the ROM setup cell, not a .state upload.")


def find_installed_rom(backend) -> Path:
    # Preserve the existing path-only API for other callers.
    return resolve_installed_game(backend)[1]


def pack_buttons(buttons: list[str], p1=(), p2=()) -> np.ndarray:
    result = np.zeros(2 * len(buttons), dtype=np.int8)
    for port, pressed in enumerate((p1, p2)):
        if isinstance(pressed, str):
            pressed = (pressed,)
        for name in pressed:
            if name not in buttons:
                raise ValueError(f"Unknown button {name!r}; actual buttons: {buttons}")
            result[port * len(buttons) + buttons.index(name)] = 1
    return result


def route_to_character(current_id: int, target_id: int) -> list[str]:
    """No wrap-around is assumed. A rectangular menu is verified at runtime."""
    if current_id not in CELLS or target_id not in CELLS:
        raise ValueError("Character ID is outside the sourced US roster.")
    r, c = CELLS[current_id]
    tr, tc = CELLS[target_id]
    vertical = ["DOWN" if tr > r else "UP"] * abs(tr-r)
    horizontal = ["RIGHT" if tc > c else "LEFT"] * abs(tc-c)
    return vertical + horizontal


def bcd_seconds(value: int) -> int | None:
    value = int(value)
    if value < 0 or value > 255 or (value & 15) > 9 or (value >> 4) > 9:
        return None
    return (value >> 4) * 10 + (value & 15)


def round_loaded(v: dict) -> bool:
    """Read-only signature from the native first-round initialization."""
    return (v["p1_hp"] == 176 and v["p2_hp"] == 176
            and v["timer_bcd"] == 0x99
            and v["p1_wins"] == 0 and v["p2_wins"] == 0
            and v["p1_char"] in CELLS and v["p2_char"] in CELLS
            and 0 < v["p1_x"] < 4096 and 0 < v["p2_x"] < 4096)


def validate_character_ids(v: dict, p1: str, p2: str | None) -> None:
    if v["p1_char"] != CHAR_IDS[p1]:
        raise RuntimeError(f"P1 identity mismatch: requested {p1}, raw ID={v['p1_char']}. Inspect screenshots.")
    if p2 is not None and v["p2_char"] != CHAR_IDS[p2]:
        raise RuntimeError(
            f"CPU/P2 identity mismatch: requested {p2}, raw ID={v['p2_char']}. "
            "The initialization method did NOT produce the requested opponent; no fake fallback is used.")


@dataclass
class MatchConfig:
    ruleset: str = "hyper"
    speed_stars: int = 3
    mode: str = "pve"
    p1_character: str = "ChunLi"
    p2_character: str = "ChunLi"
    difficulty_stars: int = 4
    target_cpu: bool = True
    allow_initialization_stage_write: bool = True
    video_seconds: float = 30
    seed: int = 42
    p1_model: str = ""
    p2_model: str = ""
    decision_frames: int = 6
    position_scale: float = 1024.0
    boot_idle_frames: int = 1740
    menu_budget: int = 8
    fight_load_budget: int = 1800
    live_wait_budget: int = 900
    confirm_p1: str = "A"
    confirm_p2: str = "START"
    match_cap_seconds: float = 180

    def validate(self):
        if self.ruleset not in ("champion", "hyper"):
            raise ValueError("ruleset must be champion or hyper")
        if type(self.speed_stars) is not int or not 0 <= self.speed_stars <= 4:
            raise ValueError("speed_stars must be 0..4; it is not CPU difficulty")
        if self.mode not in ("pve", "pvp"):
            raise ValueError("mode must be pve or pvp")
        self.p1_character = normalize_character(self.p1_character)
        self.p2_character = normalize_character(self.p2_character)
        if not 1 <= self.difficulty_stars <= 8:
            raise ValueError("difficulty_stars must be 1..8 (stored menu value 0..7)")
        if not 0 < self.video_seconds <= 300 or self.decision_frames != 6:
            raise ValueError("video_seconds must be 0..300 and decision_frames must remain 6")
        if not math.isfinite(self.position_scale) or self.position_scale <= 0:
            raise ValueError("position_scale must be positive and finite")
        if self.mode == "pve" and self.p2_model:
            raise ValueError("PvE uses the NATIVE CPU. Leave p2_model empty; use PvP for a model opponent.")
        if self.mode == "pve" and self.target_cpu and not self.allow_initialization_stage_write:
            raise ValueError("Targeted CPU selection requires the experimental init-stage write. "
                             "Set target_cpu=False only to test the natural arcade opponent.")
        if self.boot_idle_frames < 600 or self.menu_budget < 1:
            raise ValueError("Invalid startup timing")
        for key in (self.confirm_p1, self.confirm_p2):
            if key not in ("A", "B", "C", "START", "X", "Y", "Z"):
                raise ValueError("Unsupported character confirmation button")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class ColdBootMatch:
    """A single real emulator; reset means close + recreate + navigate.

    This intentionally does not use get_state(), set_state(), initial_state,
    downloadable snapshots, or opaque .state bytes hidden in the notebook.
    The cost is that every new match repeats the startup sequence.
    """
    def __init__(self, config: MatchConfig, output_dir: str | Path):
        config.validate()
        self.config = config
        self.root = Path(output_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend_module()
        self.engine = None
        self.rgb = None
        self.values = {}
        self.buttons = []
        self.frame = 0
        self.phase = "not_started"
        self.stage_write_active = False
        self.battle_started = False
        self.trace = deque(maxlen=6000)
        self.checkpoints = []
        self.frame_callback: Callable | None = None
        self.report = {
            "status": "STARTED", "config": asdict(config),
            "external_state_files_used": False, "in_memory_snapshots_used": False,
            "training_performed": False, "pve_python_opponent_created": False,
            "initialization_method": "native arcade menu + optional pre-fight stage selector",
            "targeted_pve_method_experimental": True,
            "sprite_identity_automatically_recognized": False,
            "native_cpu_control_automatically_proven": False,
            "ram_writes": [], "checkpoints": self.checkpoints,
        }

    def _create(self):
        game_id, path = resolve_installed_game(self.backend)
        checksum = hashlib.sha1(path.read_bytes()).hexdigest()
        if checksum != ROM_SHA1:
            raise RuntimeError(f"Wrong ROM SHA1 {checksum}; expected {ROM_SHA1}. Address map is version-specific.")
        self.report["rom_sha1"] = checksum
        self.report["backend_game_id"] = game_id
        scenario = self.root / "bootstrap_scenario.json"
        scenario.write_text("{}", encoding="utf-8")
        self.engine = self.backend.make(game=game_id, state=self.backend.State.NONE,
                                        players=2, use_restricted_actions=self.backend.Actions.ALL,
                                        scenario=str(scenario), render_mode="rgb_array")
        self.buttons = list(self.engine.buttons)
        self.rgb, _ = self.engine.reset()
        before = np.asarray(self.engine.get_ram()).copy()
        for name, (address, dtype) in RAM_FIELDS.items():
            self.engine.data.set_variable("boot_"+name, {"address": address, "type": dtype})
        if not np.array_equal(before, np.asarray(self.engine.get_ram())):
            raise RuntimeError("Registering read definitions unexpectedly changed RAM.")
        self.values = self.read()
        self.fps = float(self.engine.em.get_screen_rate())
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise RuntimeError("Invalid native frame rate")
        self.report.update(buttons=self.buttons, video_fps=self.fps,
                           native_image_shape=list(self.rgb.shape),
                           versions={})
        for package in ("stable-retro", "numpy"):
            try:
                self.report["versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass

    def read(self):
        return {name: int(self.engine.data.lookup_value("boot_"+name)) for name in RAM_FIELDS}

    def _write_stage_before_load(self):
        if self.battle_started:
            raise RuntimeError("Safety check: setup writes are forbidden after battle initialization.")
        value = CHAR_IDS[self.config.p2_character]
        # The only RAM value this implementation ever writes. Character IDs,
        # health, position, timer and difficulty-derived combat parameters are NOT written.
        before = self.values["stage_selector"]
        self.engine.data.set_value("boot_stage_selector", value)
        if int(self.engine.data.lookup_value("boot_stage_selector")) != value:
            raise RuntimeError("Stage-selector write did not read back correctly.")
        self.report["ram_writes"].append({"frame": self.frame, "phase": self.phase,
            "address": "0xFF96AA", "type": ">u2", "before": before, "value": value})

    def frame_step(self, p1=(), p2=()):
        if self.config.mode == "pve" and p2:
            raise RuntimeError("PvE safety: never send buttons to P2; CPU is native to GAME START.")
        if self.stage_write_active:
            self._write_stage_before_load()
        a = pack_buttons(self.buttons, p1, p2)
        self.rgb, _, terminated, truncated, _ = self.engine.step(a)
        self.frame += 1
        self.values = self.read()
        row = {"frame": self.frame, "phase": self.phase, "p1_buttons": list(p1),
               "p2_buttons": list(p2), **self.values}
        self.trace.append(row)
        if self.frame_callback is not None:
            self.frame_callback(self.rgb, row)
        if terminated or truncated:
            raise RuntimeError("Unexpected native terminal flag in bootstrap scenario.")
        return self.values

    def advance(self, count: int, p1=(), p2=()):
        if not 0 <= int(count) <= 20000:
            raise ValueError("Invalid native-frame advance length")
        for _ in range(int(count)):
            self.frame_step(p1, p2)

    def tap(self, player: int, key: str, hold=8, release=8):
        if player not in (1, 2):
            raise ValueError("player must be 1 or 2")
        self.advance(hold, (key,) if player == 1 else (), (key,) if player == 2 else ())
        self.advance(release)

    def shot(self, label: str):
        from PIL import Image
        name = f"{len(self.checkpoints):02d}_{label}.png"
        Image.fromarray(self.rgb).save(self.root/name)
        self.checkpoints.append({"file": name, "frame": self.frame, "phase": self.phase,
                                 "readings": dict(self.values)})
        print(f"[{self.phase}] {label}: char IDs {self.values['p1_char']}/{self.values['p2_char']}; "
              f"HP {self.values['p1_hp']}/{self.values['p2_hp']}; "
              f"timer raw {self.values['timer_bcd']}", flush=True)

    def _set_difficulty_via_menu(self):
        from sf2_menu import set_difficulty_via_menu_fixed
        return set_difficulty_via_menu_fixed(self)

    def _cursor_responds(self, player: int) -> bool:
        if round_loaded(self.values):
            self.shot("fight_reached_before_selection")
            raise RuntimeError("Reached a fight before choosing characters. Stop rather than patch a live fighter.")
        key = f"p{player}_char"
        before = self.values[key]
        if before not in CELLS:
            return False
        row, col = CELLS[before]
        direction = "RIGHT" if col < 5 else "LEFT"
        reverse = "LEFT" if direction == "RIGHT" else "RIGHT"
        expected = GRID[row][col+(1 if direction == "RIGHT" else -1)]
        self.tap(player, direction)
        after = self.values[key]
        self.tap(player, reverse)
        # A menu name alone is not detection. We check a reversible cursor movement.
        return after == expected and self.values[key] == before

    def _seek_player_select(self):
        from sf2_menu import select_rules_and_game_start
        select_rules_and_game_start(self)
        self.phase = 'seek_native_player_select'
        for attempt in range(self.config.menu_budget):
            if self._cursor_responds(1):
                self.shot('p1_character_cursor_responds')
                return
            self.advance(60)
        self.shot('character_cursor_not_verified')
        raise RuntimeError('Character cursor not verified after GAME START; inspect screenshot, no blind START retry.')

    def _choose(self, player: int, name: str):
        self.phase = f"choose_p{player}_{name}"
        target = CHAR_IDS[name]
        for key in route_to_character(self.values[f"p{player}_char"], target):
            before = self.values[f"p{player}_char"]
            r, c = CELLS[before]
            nr, nc = r + (key == "DOWN") - (key == "UP"), c + (key == "RIGHT") - (key == "LEFT")
            expected = GRID[nr][nc]
            self.tap(player, key)
            if self.values[f"p{player}_char"] != expected:
                self.shot("cursor_mismatch")
                raise RuntimeError("Actual cursor movement differs from expected grid. Recalibration required.")
        if self.values[f"p{player}_char"] != target:
            raise RuntimeError("Target character was not selected")
        self.shot(f"p{player}_{name}_highlighted")

    def _join_second_player(self):
        # Native arcade challenge is deliberately used, not the V.S. BATTLE
        # handicap/settings menu from the earlier calibration notebook.
        self.phase = "pvp_native_second_player_join"
        for _ in range(3):
            if self._cursor_responds(2) and self._cursor_responds(1):
                self.shot("both_player_cursors_respond")
                self.report["pvp_route"] = "native arcade two-controller challenge"
                return
            self.tap(2, "START", release=128)
        raise RuntimeError("P2 did not join as a selectable human player. No PvP success is claimed.")

    def _confirm_and_load(self):
        self.phase = "native_match_initialization"
        self.stage_write_active = self.config.mode == "pve" and self.config.target_cpu
        first = (self.config.confirm_p1,)
        second = (self.config.confirm_p2,) if self.config.mode == "pvp" else ()
        try:
            for index in range(self.config.fight_load_budget):
                # Menu confirmation held for 8 native frames; never pressed again in battle.
                self.frame_step(first if index < 8 else (), second if index < 8 else ())
                if round_loaded(self.values):
                    self.stage_write_active = False
                    self.battle_started = True
                    self.report["first_round_initialized_frame"] = self.frame
                    # Let native initialization settle; the source captures after eight
                    # frames inside the walk-in window. No setup writes occur here.
                    self.advance(8)
                    self.shot("native_round_initialized")
                    validate_character_ids(self.values, self.config.p1_character,
                        self.config.p2_character if self.config.mode == "pvp" or self.config.target_cpu else None)
                    if self.values["difficulty_menu"] != self.config.difficulty_stars-1:
                        raise RuntimeError("Difficulty changed during initialization")
                    self.report["difficulty_fight_readback"] = self.values["difficulty_fight"]
                    if self.config.mode == "pve" and self.values["difficulty_fight"] != self.config.difficulty_stars-1:
                        raise RuntimeError("Native fight difficulty copy does not match menu setting. Do not train on this setup.")
                    return
            self.shot("fight_initialization_timeout")
            raise RuntimeError("No native first-round signature. The game may still be in a menu; see images and trace.")
        finally:
            self.stage_write_active = False

    def _wait_until_live(self):
        self.phase = "wait_for_native_countdown"
        consecutive = 0
        for _ in range(self.config.live_wait_budget):
            self.frame_step()
            validate_character_ids(self.values, self.config.p1_character,
                self.config.p2_character if self.config.mode == "pvp" or self.config.target_cpu else None)
            seconds = bcd_seconds(self.values["timer_bcd"])
            if seconds is not None and 0 < seconds < 99:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= 3:
                if self.values["p1_hp"] < 0 or self.values["p2_hp"] < 0:
                    raise RuntimeError("A character was KO'd before the demo could start")
                self.phase = "battle"
                self.shot("battle_start")
                self.report.update(status="BOOTSTRAP_CHECKS_PASSED_VISUAL_CHECK_REQUIRED",
                    actual_character_ids={"p1":self.values["p1_char"],"p2":self.values["p2_char"]},
                    initial_battle_readings=dict(self.values),
                    all_setup_writes_stopped=True,
                    loaded_identifiers_checked_read_only=True)
                return
        self.shot("countdown_not_running")
        raise RuntimeError("Timer did not start. Do not mistake a menu/paused frame for a live fight.")

    def start(self):
        try:
            self._create()
            self._set_difficulty_via_menu()
            self._seek_player_select()
            if self.config.mode == "pvp":
                self._join_second_player()
            self._choose(1, self.config.p1_character)
            if self.config.mode == "pvp":
                self._choose(2, self.config.p2_character)
            self._confirm_and_load()
            self._wait_until_live()
            return self.rgb.copy(), dict(self.values)
        except Exception as exc:
            self.stage_write_active = False
            self.report.update(status="ERROR", error=repr(exc), failed_phase=self.phase)
            if self.rgb is not None:
                self.shot("error")
            raise
        finally:
            self.flush()

    def flush(self):
        write_json(self.root/"bootstrap_report.json", self.report)
        write_json(self.root/"bootstrap_trace.json", list(self.trace))

    def close(self):
        self.stage_write_active = False
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        gc.collect()
        self.flush()
