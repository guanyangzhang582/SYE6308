"""Public practice arena using the classroom game engine and isolated actor RPC.
No student reward is called. No learner is optimized. No arbitrary student game
wrapper is accepted. This is a reviewed-code practice runner, NOT a secure sandbox.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile
import numpy as np

from sf2_actions import action_index,compiled
from sf2_common import json_write,runtime_versions
from sf2_submission import inspect_submission,load_rules,rules_hash,_public_check,digest
from sf2_arena_rpc import ActorProcess


def prepare_frame(rgb,raw,frame):
    rgb=np.asarray(rgb)
    if rgb.shape!=(224,320,3) or rgb.dtype!=np.uint8:
        raise ValueError('Arena game must provide raw 224x320 RGB uint8.')
    # Do not expose RAM, opponent actions/weights or result functions to actor RPC.
    coordinates={k:int(raw[k]) for k in ('p1_x','p1_y','p2_x','p2_y')}
    return {'rgb_b64':base64.b64encode(rgb.tobytes()).decode('ascii'),
            'coordinates':coordinates,'frame':int(frame)}


def latency_summary(values):
    if not values:return {'requests':0}
    return {'requests':len(values),'mean_ms':float(np.mean(values))*1000,
            'p95_ms':float(np.percentile(values,95))*1000,'max_ms':float(max(values))*1000}


def play_matches(infos,rule,out,*,offsets,swap_sides,seed=42,record_video=True,
                 max_video_rounds=2,game_factory=None,actor_factory=ActorProcess):
    """Both policies receive the same frame before either set of buttons is applied."""
    if not offsets or len(offsets)>20 or any(type(x)is not int or not 0<=x<=60 for x in offsets):
        raise ValueError('Use 1..20 startup offsets, each an integer from 0 to 60.')
    if len(infos)!=2:raise ValueError('Exactly two submissions are required.')
    if game_factory is None:
        from sf2_game import Game
        game_factory=Game
    from sf2_video import Video
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    actors=[];game=None;rec=None;rows=[];videos=[];counts=[Counter(),Counter()]
    traces=out/'decisions.jsonl'
    commands=[compiled(x['spec']['actions']) for x in infos]
    initial_hashes=[]
    try:
        for i,info in enumerate(infos):
            a=actor_factory(info['agent_path'],out/f'actor_{"AB"[i]}',rule['inference'])
            actors.append(a);initial_hashes.append(a.ready['parameter_hash'])
            if a.ready.get('action_count')!=len(commands[i]):raise ValueError('Actor loaded wrong action count.')
        game=game_factory(rule['common'],out/'game')
        orders=[(0,1),(1,0)] if swap_sides else [(0,1)]
        with traces.open('w',encoding='utf-8') as trace:
            for order in orders:
                for index,offset in enumerate(offsets):
                    p1,p2=(infos[i] for i in order)
                    game.open(p1['spec']['character'],p2['spec']['character'],'pvp',1)
                    for _ in range(offset):
                        raw,winner,reason=game.step_keys((),())
                        if reason:raise RuntimeError('Round ended during the configured start offset.')
                    rid=len(rows);payload=prepare_frame(game.rgb,dict(game.raw),game.elapsed)
                    for port,idx in enumerate(order,1):
                        res=actors[idx].request('reset',player=port,seed=int(seed)+index*100+idx,**payload)
                        if res.get('status')!='reset' or res.get('player')!=port:
                            raise ValueError('Actor reset returned a wrong controller port.')
                    if record_video and index == 0 and len(videos)<max_video_rounds:
                        rec=Video(out/f'round_{rid:02d}.mp4',game.fps)
                        def on_frame(rgb,raw,k1,k2,frame):
                            rec.write(rgb,[f'P1 {p1["metadata"]["team_id"]} | P2 {p2["metadata"]["team_id"]}',
                                f'HP {raw["p1_hp"]}/{raw["p2_hp"]} | native frame={frame}',
                                'PRACTICE | CPU inference | NO TRAINING | six-frame decisions'],k1,k2)
                        game.on_frame=on_frame
                    winner=reason=None;decisions=0
                    while reason is None:
                        frame=game.elapsed
                        payload=prepare_frame(game.rgb,dict(game.raw),frame)
                        ids=[]
                        # Independent JSON processes; neither receives the other's chosen action.
                        for port,idx in enumerate(order,1):
                            res=actors[idx].request('act',round_id=rid,**payload)
                            if res.get('status')!='action' or res.get('frame')!=frame or res.get('player')!=port:
                                raise ValueError('Actor action has wrong frame/controller identity.')
                            a=action_index(res['action'],len(commands[idx]));ids.append(a);counts[idx][a]+=1
                        trace.write(json.dumps({'round':rid,'frame':frame,'offset':offset,
                            'p1_agent':'AB'[order[0]],'p2_agent':'AB'[order[1]],
                            'p1_action':ids[0],'p2_action':ids[1]},separators=(',',':'))+'\n')
                        for k1,k2 in zip(commands[order[0]][ids[0]],commands[order[1]][ids[1]]):
                            raw,winner,reason=game.step_keys(k1,k2)
                            if reason:break
                        decisions+=1
                    end_hashes=[]
                    for a in actors:
                        result=a.request('finish')
                        if result.get('status')!='unchanged':raise RuntimeError('Actor did not confirm unchanged model.')
                        end_hashes.append(result['parameter_hash'])
                    if end_hashes!=initial_hashes:raise RuntimeError('An actor changed policy parameters during inference.')
                    if reason=='frame_cap':outcome='cutoff'
                    elif winner in ('p1','p2'):outcome='AB'[order[int(winner[-1])-1]]
                    elif winner=='draw':outcome='draw'
                    else:raise RuntimeError('Unexpected winner/result from the public game engine.')
                    row={'round':rid,'offset':offset,'p1_agent':'AB'[order[0]],'p2_agent':'AB'[order[1]],
                         'winner':outcome,'end_reason':reason,'p1_hp':raw['p1_hp'],'p2_hp':raw['p2_hp'],
                         'native_frames':game.elapsed,'decisions':decisions,'weights_unchanged':True}
                    rows.append(row)
                    json_write(out/'rounds_partial.json',rows)
                    print(f'ARENA round={rid} P1={row["p1_agent"]} P2={row["p2_agent"]} offset={offset}: '
                          f'{outcome}, HP={row["p1_hp"]}/{row["p2_hp"]}, reason={reason}',flush=True)
                    if rec is not None:
                        videos.append(rec.finish());rec=None;game.on_frame=None
        totals=Counter(row['winner'] for row in rows)
        return {'rounds':rows,'videos':videos,'summary':{k:totals[k] for k in ('A','B','draw','cutoff')},
            'training_performed':False,'policy_weights_unchanged':True,
            'actor_initial_hashes':initial_hashes,
            'actor_runtime_versions':[a.ready.get('runtime_versions',{}) for a in actors],
            'latency':{name:latency_summary(actors[i].times) for i,name in enumerate('AB')},
            'action_counts':{name:{infos[i]['spec']['actions'][k]['name']:v for k,v in counts[i].items()} for i,name in enumerate('AB')},
            'same_package_interface_test':infos[0]['agent_sha256']==infos[1]['agent_sha256'],
            'warnings':['One or more rounds reached the safety frame cap; not scored as wins.'] if totals['cutoff'] else [],
            'coverage':'Only these submitted agents, character matchup, offsets and ports were tested.',
            'note':'Self-match is an interface check, not a skill benchmark. Rewards are never used for match scoring. Cutoffs are not wins.'}
    finally:
        if rec is not None:rec.close()
        if game is not None:game.close()
        for a in actors:a.close()


def framework_hashes(folder):
    return {p.name:digest(p.read_bytes()) for p in sorted(Path(folder).glob('*.py'))}


def run_job(job):
    out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    rule=load_rules(job['rules_path'])
    if job.get('trust_submissions') is not True:
        raise ValueError('Review custom Python/serialized weights, then set TRUST_SUBMISSIONS=True. Process separation is not a sandbox.')
    infos=[]
    for i,key in enumerate(('submission_a','submission_b')):
        infos.append(inspect_submission(job[key],rule,out/'inputs'/f'agent_{"AB"[i]}.zip'))
    for i,info in enumerate(infos):
        json_write(out/f'inspection_{"AB"[i]}.json',{k:v for k,v in info.items() if k not in ('spec','manifest')})
    result=play_matches(infos,rule,out,offsets=job['offsets'],swap_sides=job['swap_sides'],
                        seed=int(job.get('seed',42)),record_video=job.get('record_video',True),max_video_rounds=2)
    success='ARENA_SELF_CHECK_PASSED' if job['mode']=='self_check' else 'ARENA_MATCH_COMPLETE'
    result.update(status=success,rules_id=rule['rules_id'],rules_sha256=rules_hash(rule),
        submissions={name:{'team_id':x['metadata']['team_id'],'submission_sha256':x['submission_sha256'],
                          'agent_sha256':x['agent_sha256'],'character':x['spec']['character']}
                     for name,x in zip('AB',infos)},
        framework_hashes=framework_hashes(Path(__file__).parent),
        official_score=False,security_sandbox=False,
        rule_status=rule['status'],timing_scope='CPU actor RPC including preprocessing, inference and transport; early diagnostic snapshots may add cost.')
    # Receipt is OUTSIDE submission.zip: no circular checksum or post-check model mutation.
    json_write(out/'receipt.json',{k:v for k,v in result.items() if k not in ('videos','action_counts')})
    return result


def make_diagnostic_zip(out):
    out=Path(out);target=out/'arena_diagnostics.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            rel=p.relative_to(out)
            if not p.is_file() or p==target or rel.parts[0] in ('inputs','submission_inputs') or '__pycache__' in rel.parts:
                continue
            if p.suffix.lower() not in ('.json','.jsonl','.log','.png','.npz','.mp4'):continue
            z.write(p,rel.as_posix())
    return target


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args()
    job=json.loads(Path(args.job).read_text());out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();code=1
    try:report=run_job(job);code=0
    except BaseException as exc:
        report={'status':'ARENA_CHECK_FAILED','error':type(exc).__name__+': '+str(exc),
                'traceback':traceback.format_exc(),'official_score':False,'security_sandbox':False}
        print(report['traceback'],flush=True)
    report.update(wall_seconds=time.perf_counter()-start,runtime_versions=runtime_versions())
    json_write(out/'report.json',report)
    print('STATUS:',report['status'],flush=True);print('REPORT='+str(out/'report.json'),flush=True)
    print('DIAGNOSTICS='+str(make_diagnostic_zip(out)),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
