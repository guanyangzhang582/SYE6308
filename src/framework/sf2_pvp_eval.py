"""Saved-policy evaluation against an actual frozen agent, from learner ports.
No optimisers, no native-CPU evaluation fallback, no opponent replacement.
"""
from pathlib import Path
import numpy as np
from sf2_bundle import LoadedAgent
from sf2_common import parameter_hash, file_hash, rules_signature, source_module, json_write
from sf2_env import BattleEnv
from sf2_release import validate_frozen_config, validate_pvp_evaluation


def summarize_pvp(rows):
    n=len(rows)
    counts={name:sum(row['result']==name for row in rows)
            for name in ('win','loss','draw','cutoff')}
    return {**counts,'episodes':n,'win_fraction':counts['win']/n if n else None,
            'mean_hp_margin':float(np.mean([max(0,r['self_hp'])-max(0,r['opponent_hp'])
                                           for r in rows])) if n else None,
            'mean_episode_reward':float(np.mean([r['episode_reward'] for r in rows])) if n else None}


def evaluate_frozen(job, out):
    import torch
    from sf2_video import Video
    cfg=dict(job['evaluation'])
    eval_cfg=validate_pvp_evaluation({k:cfg[k] for k in
        ('offsets','seed','deterministic','record_video') if k in cfg})
    frozen=validate_frozen_config(job['frozen'])
    if not cfg.get('bundle'):raise ValueError('Select the learner package for PvP evaluation.')
    packages=[('trained',cfg['bundle'])]
    if cfg.get('compare_initial',False):
        if not cfg.get('initial_bundle'):raise ValueError('Set initial_bundle for a before/after comparison.')
        packages.insert(0,('initial',cfg['initial_bundle']))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    opponent_sha=file_hash(frozen['bundle'])
    rows=[];videos=[];unchanged=[]
    for label,package in packages:
        learner=opponent=env=rec=None
        try:
            learner=LoadedAgent(package,trusted=job.get('trust_packages',False),device='cpu')
            opponent=LoadedAgent(frozen['bundle'],trusted=job.get('trust_packages',False),device='cpu')
            for actor in (learner,opponent):
                if rules_signature(actor.common)!=rules_signature(job['common']):
                    raise ValueError('A PvP package uses different arena rules.')
            start_hash=parameter_hash(learner.model)
            reward=source_module(learner.directory/'reward.py','reward').compute_reward
            # difficulty_stars=1 is an internal native-menu initializer only.
            # In mode='pvp' both ports are controlled by models, never the CPU AI.
            pair={'character':opponent.spec['character'],'difficulty_stars':1,
                  'neutral_jitter_max':frozen['neutral_jitter_max']}
            env=BattleEnv(learner.spec,job['common'],[pair],out/label,reward,
                opponent_agent=opponent,learner_sides=tuple(frozen['learner_sides']),
                opponent_deterministic=frozen['opponent_deterministic'])
            for player in frozen['learner_sides']:
                env.learner_sides=(player,)
                for index,offset in enumerate(eval_cfg['offsets']):
                    # Repeated no-op offsets are scenarios, not independent CPU seeds.
                    # Seed only this separate evaluation process; training RNGs are untouched.
                    seed=eval_cfg['seed']+player*1000+index
                    torch.manual_seed(seed)
                    obs,reset=env.reset(seed=seed,options={'offset':offset})
                    if env.player!=player:raise RuntimeError('Wrong learner port in PvP evaluation.')
                    if eval_cfg['record_video'] and index==0:
                        rec=Video(out/f'{label}_P{player}.mp4',env.game.fps)
                        def frame(rgb,raw,k1,k2,number):
                            rec.write(rgb,[f'PVP EVALUATION | {label} learner=P{player} | frozen=P{3-player}',
                                f'P1 HP={raw["p1_hp"]} | P2 HP={raw["p2_hp"]} | offset={offset} | frame={number}',
                                'No learning. The frozen opponent is not the native CPU.'],k1,k2)
                        env.game.on_frame=frame
                    reward_sum=0.;count={};decisions=0
                    while True:
                        a=learner.act(env.sides[player],env.game.raw,
                                      deterministic=eval_cfg['deterministic'])
                        obs,r,t,tr,info=env.step(a)
                        reward_sum+=r;decisions+=1
                        name=info['action_name'];count[name]=count.get(name,0)+1
                        if env.sides[1].frame_ids[-1]!=env.sides[2].frame_ids[-1]:
                            raise RuntimeError('PvP actors observed different decision clocks.')
                        if t or tr:break
                    row={k:info[k] for k in ('result','end_reason','self_hp','opponent_hp','native_frames')}
                    row.update(controller=label,learner_player=player,opponent_player=3-player,
                        offset=offset,seed=seed,episode_reward=reward_sum,decisions=decisions,
                        learner_deterministic=eval_cfg['deterministic'],
                        opponent_deterministic=frozen['opponent_deterministic'],
                        initial_readings=reset['raw'],action_counts=count)
                    rows.append(row)
                    print(f'EVAL_PVP {label}: learner=P{player}, opponent=P{3-player}, offset={offset}, '
                          f'{row["result"]}, learner/opponent HP={row["self_hp"]}/{row["opponent_hp"]}, '
                          f'reward={reward_sum:+.4f}, reason={row["end_reason"]}',flush=True)
                    if rec:videos.append(rec.finish());rec=None
                    env.game.on_frame=None
            stats=env.performance_stats()
            if parameter_hash(learner.model)!=start_hash or not stats['frozen_opponent_unchanged']:
                raise RuntimeError('A policy changed during no-learning PvP evaluation.')
            unchanged.append({'controller':label,'learner_package_sha256':file_hash(package),
                              'learner_unchanged':True,'frozen_opponent_unchanged':True})
        finally:
            if rec:rec.close()
            if env:env.close()  # owns opponent
            elif opponent:opponent.close()
            if learner:learner.close()
    if file_hash(frozen['bundle'])!=opponent_sha:
        raise RuntimeError('Frozen package bytes changed during evaluation.')
    groups={label:summarize_pvp([r for r in rows if r['controller']==label]) for label,_ in packages}
    ports={label:{str(p):summarize_pvp([r for r in rows if r['controller']==label and r['learner_player']==p])
                  for p in frozen['learner_sides']} for label,_ in packages}
    print('PVP EVALUATION SUMMARY:',groups,flush=True)
    result={'status':'PVP_EVALUATION_COMPLETE','training_performed':False,
        'validation_target':'frozen_agent_pvp','episodes':rows,'summary':groups,'per_port':ports,
        'learner_sides':frozen['learner_sides'],'opponent_sha256':opponent_sha,
        'evaluation_settings':eval_cfg,'frozen_opponent_unchanged':True,'policy_checks':unchanged,
        'win_fraction':groups['trained']['win_fraction'],'mean_hp_margin':groups['trained']['mean_hp_margin'],
        'videos':videos,'note':'Every result is from the learner perspective. Fixed offsets are scenarios, not independent opponents. No general-strength claim.'}
    return result
