"""One reusable emulator with verified automatic start caches.
Same native menu/character/difficulty guards as the verified one-star environment.
Only resetting a round restores state; no live HP/position/difficulty changes.
"""
from __future__ import annotations
from collections import deque
from contextlib import redirect_stdout
from pathlib import Path
import time
from sf2_boot import MatchConfig,validate_character_ids
from sf2_start_cache import CachedColdBootMatch,_MEMO,_restore_exact
from sf2_common import round_result,rules_signature

class Game:
    def __init__(self,common,out):
        rules_signature(common)
        self.common=dict(common);self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.boot=None;self.key=None;self.payload=None;self.on_frame=None
        self.elapsed=0;self.last_boot_seconds=0.;self.resets=0;self.warm_resets=0
    @property
    def raw(self):return self.boot.values
    @property
    def rgb(self):return self.boot.rgb
    @property
    def fps(self):return self.boot.fps
    def open(self,p1,p2,mode,stars):
        start=time.perf_counter()
        key=(p1,p2,mode,int(stars),self.common['ruleset'],self.common['speed_stars'])
        self.mode=mode;self.on_frame=None
        if self.boot is not None and self.key==key and self.payload is not None:
            self.boot.frame_callback=None
            _restore_exact(self.boot,self.payload)
            self.warm_resets+=1
            self.last_cache_status='WARM_SAME_EMULATOR_RESTORE'
        else:
            self.close()
            cfg=MatchConfig(mode=mode,p1_character=p1,p2_character=p2,
                            difficulty_stars=int(stars),ruleset=self.common['ruleset'],
                            speed_stars=self.common['speed_stars'])
            folder=self.out/f'start_{self.resets:04d}'
            folder.mkdir(parents=True,exist_ok=True)
            self.boot=CachedColdBootMatch(cfg,folder)
            try:
                with (folder/'boot.log').open('w',encoding='utf-8') as log,redirect_stdout(log):
                    self.boot.start()
                self.payload=_MEMO.get(self.boot.cache_info['path'])
                if self.payload is None or not self.payload['meta'].get('roundtrip_verified'):
                    raise RuntimeError('No verified start payload was produced.')
                self.key=key;self.last_cache_status=self.boot.cache_info['status']
            except BaseException:
                self.close();raise
        self.boot.trace=deque(maxlen=120)  # Small failure ring; no long routine per-frame JSON dumps.
        self.identity=(p1,p2);self.elapsed=0
        self.wins=(self.raw['p1_wins'],self.raw['p2_wins'])
        self.write_count=len(self.boot.report['ram_writes'])
        self.last_boot_seconds=time.perf_counter()-start;self.resets+=1
        return self.rgb.copy(),dict(self.raw)
    def step_keys(self,k1=(),k2=()):
        if self.boot is None:raise RuntimeError('Open the match first.')
        before=dict(self.raw)
        if self.mode=='pve' and k2:raise RuntimeError('Native-CPU PvE must never send keys to P2.')
        if self.boot.stage_write_active or len(self.boot.report['ram_writes'])!=self.write_count:
            raise RuntimeError('Post-start RAM setup write detected.')
        raw=self.boot.frame_step(k1,k2);self.elapsed+=1
        validate_character_ids(raw,*self.identity)
        if self.mode=='pve' and (raw['difficulty_menu']!=self.boot.config.difficulty_stars-1 or
                               raw['difficulty_fight']!=self.boot.config.difficulty_stars-1):
            raise RuntimeError('CPU difficulty readback changed during battle.')
        winner,reason=round_result(raw,self.wins)
        if reason is None and any(raw[f'p{p}_hp']>before[f'p{p}_hp'] for p in (1,2)):
            raise RuntimeError('Unexpected health refill during a round.')
        if reason is None and self.elapsed>=int(self.common['max_round_frames']):reason='frame_cap'
        if self.on_frame is not None:self.on_frame(self.rgb,dict(raw),tuple(k1),tuple(k2),self.elapsed)
        return raw,winner,reason
    def close(self):
        if self.boot is not None:
            self.boot.close();self.boot=None
        self.key=None;self.payload=None
