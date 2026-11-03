"""Versioned contracts and isolated loading of student Python source."""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np

ENV_PROTOCOL='sf2-classroom-sixframe-v2'
OBS_PROTOCOL='sf2-configurable-image4-position6-v3'
BUNDLE_SCHEMA='sf2-plain-ppo-agent-v3'

def json_write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_module(path,kind):
    """Content-addressed names prevent two teams' custom model classes colliding."""
    p=Path(path).resolve();name='sf2_custom_'+kind+'_'+file_hash(p)[:24]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,p)
        if spec is None or spec.loader is None:raise ImportError(p)
        m=importlib.util.module_from_spec(spec);sys.modules[name]=m
        try:spec.loader.exec_module(m)
        except BaseException:
            sys.modules.pop(name,None);raise
    return sys.modules[name]

def parameter_hash(model):
    h=hashlib.sha256()
    for name,v in model.policy.state_dict().items():
        h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def compute_transition(before,after,player,winner,reason,frames):
    if player not in (1,2):raise ValueError('Invalid learner player')
    opp=3-player
    dealt=max(0,max(before[f'p{opp}_hp'],0)-max(after[f'p{opp}_hp'],0))
    taken=max(0,max(before[f'p{player}_hp'],0)-max(after[f'p{player}_hp'],0))
    result=('cutoff' if reason=='frame_cap' else 'draw' if winner=='draw' else 'win' if winner==f'p{player}' else 'loss' if winner else 'ongoing')
    return {'damage_dealt':dealt,'damage_taken':taken,'result':result,'end_reason':reason,
            'elapsed_native_frames':frames,'self_hp':after[f'p{player}_hp'],'opponent_hp':after[f'p{opp}_hp'],
            'self_player':player,
            'self_position':[after[f'p{player}_x'],after[f'p{player}_y']],
            'opponent_position':[after[f'p{opp}_x'],after[f'p{opp}_y']]}

def round_result(v,wins):
    h1,h2=v['p1_hp'],v['p2_hp']
    if min(h1,h2)<0:
        return ('draw' if h1<0 and h2<0 else 'p2' if h1<0 else 'p1'),'signed_hp_ko'
    a,b=v['p1_wins']>wins[0],v['p2_wins']>wins[1]
    if a or b:return ('draw' if a and b else 'p1' if a else 'p2'),'round_counter'
    from sf2_boot import bcd_seconds
    if bcd_seconds(v['timer_bcd'])==0:
        return ('p1' if h1>h2 else 'p2' if h2>h1 else 'draw'),'native_time_up'
    return None,None

def rules_signature(common):
    if common['ruleset'] not in ('champion','hyper') or type(common['speed_stars']) is not int or not 0<=common['speed_stars']<=4:
        raise ValueError('Ruleset must be champion/hyper and supported speed 0..4')
    if common['decision_frames']!=6 or not np.isfinite(common['position_scale']) or common['position_scale']<=0:
        raise ValueError('This protocol uses 6-frame decisions and a positive coordinate scale')
    if type(common['max_round_frames']) is not int or not 6<=common['max_round_frames']<=36000:
        raise ValueError('max_round_frames must be 6..36000')
    return {k:common[k] for k in ('ruleset','speed_stars','decision_frames','position_scale','max_round_frames')}


def runtime_versions():
    from importlib.metadata import version,PackageNotFoundError
    result={}
    for name in ('torch','numpy','gymnasium','stable-baselines3','stable-retro'):
        try:result[name]=version(name)
        except PackageNotFoundError:result[name]=None
    return result
