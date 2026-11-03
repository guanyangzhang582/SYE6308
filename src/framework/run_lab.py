"""Explicit task entry points. Ordinary PPO and standard SB3 vector environments.
Training, evaluation and PvP are separate tasks; no overloaded global MODE.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from sf2_common import (json_write,file_hash,source_module,parameter_hash,
                         rules_signature,runtime_versions)
from sf2_actions import library,compiled,action_index,QUANTUM
from sf2_bundle import make_spec,LoadedAgent,export_agent
from sf2_env import BattleEnv
from sf2_observation import description as input_description,image_shape,display_frames
from sf2_release import (initialise_policy, validate_frozen_config, validate_pvp_evaluation,
                         validation_job, validation_target)
from sf2_pvp_eval import evaluate_frozen


def student(job):
    folder=Path(job['sources'])
    actions=source_module(folder/'student_actions.py','actions')
    reward=source_module(folder/'student_reward.py','reward').compute_reward
    module=source_module(folder/'student_model.py','model')
    return make_spec(job['agent'],library(actions.CUSTOM_ACTIONS),folder/'student_observation.py'),reward,module


def schedule(settings):
    from sf2_boot import normalize_character
    chars=settings['opponent_characters'];levels=settings['difficulty_stars']
    if not chars or not levels:raise ValueError('Choose opponent characters and CPU difficulties.')
    if any(type(x) is not int or x not in range(1,9) for x in levels):
        raise ValueError('CPU difficulty must be 1..8 stars.')
    jitter=settings.get('neutral_jitter_max',0)
    if type(jitter) is not int or not 0<=jitter<=60:raise ValueError('neutral_jitter_max must be 0..60.')
    return [{'character':normalize_character(c),'difficulty_stars':n,'neutral_jitter_max':jitter}
             for c in chars for n in levels]


def show_input(folder,env,obs,prefix='actual'):
    from PIL import Image,ImageDraw
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    Image.fromarray(env.render()).save(folder/(prefix+'_game.png'))
    values=obs if isinstance(obs,dict) else {'image':obs}
    np.savez_compressed(folder/(prefix+'_input.npz'),**values)
    images=values['image'];frames=display_frames(images,env.spec)
    side=env.sides[env.player]
    h,w=frames[0].shape[:2];zoom=2;top=30
    board=Image.new('RGB',(4*w*zoom,h*zoom+top),'white');draw=ImageDraw.Draw(board)
    for i,img in enumerate(frames):
        im=Image.fromarray(img).convert('RGB').resize((w*zoom,h*zoom),Image.Resampling.NEAREST)
        board.paste(im,(i*w*zoom,top))
        draw.text((i*w*zoom+5,5),f'Game frame {side.frame_ids[i]}',fill='black')
    board.save(folder/(prefix+'_model_images.png'))
    meta={
        'keys':list(values),'image_shape':list(images.shape),'image_dtype':str(images.dtype),
        'position_order':['self_x','self_y','opponent_x','opponent_y','dx','dy'],
        'position':values.get('position',np.array([])).tolist(),
        'scale':env.common['position_scale'],'player':env.player,
        'raw':dict(env.game.raw),'image_mirrored':False,'control_input':False,
        'frame_ids':list(side.frame_ids),'observed_frames_checked':side.observed_frames_checked,
        'input_definition':input_description(env.spec),
        'source':'Actual env observation returned to the policy, not a candidate resize'}
    json_write(folder/(prefix+'_input.json'),meta)
    print('ACTUAL MODEL INPUT:',json.dumps({k:meta[k] for k in
          ('keys','image_shape','image_dtype','frame_ids')}),flush=True)
    return meta


def check(job,out):
    """One selected real match: cache roundtrip, fixed input replay and a real preview."""
    import gymnasium as gym
    import torch
    from sf2_observation import observation_space
    from sf2_video import Video
    spec,reward,module=student(job)
    # Installed ordinary PPO, synthetic shape contract only. This model is never trained/exported.
    shape_space=observation_space(spec['use_position'],job['common']['position_scale'],spec=spec)
    shape_actions=gym.spaces.Discrete(len(spec['actions']))
    class ShapeOnly(gym.Env):
        observation_space=shape_space
        action_space=shape_actions
    shape=ShapeOnly()
    model=module.build_model(shape,job['train'],job['agent']['seed'],'cpu')
    data={'image':np.zeros(image_shape(spec),np.uint8),'position':np.zeros(6,np.float32)}
    if not spec['use_position']:data=data['image']
    action,_=model.predict(data,deterministic=True)
    action_index(action,len(spec['actions']));del model
    pair=schedule(job['pve'])[0]
    env=BattleEnv(spec,job['common'],[pair],out/'game',reward)
    traces=[];resets=[];rec=None;motion_input=None
    try:
        for repeat in range(2):
            t=time.perf_counter();obs,info=env.reset(seed=int(job['agent']['seed']),options={'offset':0})
            resets.append(time.perf_counter()-t)
            if repeat==0:reset_input=show_input(out,env,obs)
            h=hashlib.sha256()
            for i in range(60):
                values=obs if isinstance(obs,dict) else {'image':obs}
                for key in sorted(values):h.update(key.encode()+np.ascontiguousarray(values[key]).tobytes())
                h.update(json.dumps(env.game.raw,sort_keys=True).encode())
                # Uses only the 15 known built-in actions, not student-provided scripts.
                obs,r,term,trunc,info=env.step((2,2,11,0,3,0,14,0)[i%8])
                h.update(json.dumps([r,term,trunc,info['result'],info['end_reason']]).encode())
                if repeat==0 and i==11:
                    motion_input=show_input(out,env,obs,prefix='motion')
                if term or trunc:break
            traces.append(h.hexdigest())
        if traces[0]!=traces[1]:raise RuntimeError('Same-start fixed-input replay differs after warm reset.')
        if motion_input is None:raise RuntimeError('Round ended before a stepped four-frame input could be checked.')
        obs,_=env.reset(seed=int(job['agent']['seed']),options={'offset':0})
        rec=Video(out/'preview.mp4',env.game.fps)
        env.game.on_frame=lambda rgb,raw,k1,k2,frame:rec.write(rgb,
            ['RANDOM PREVIEW | no model learning',
             f'P1 HP={raw["p1_hp"]} | P2 HP={raw["p2_hp"]} | frame={frame}',
             'Input display is not proof of an executed attack.'],k1,k2)
        rng=np.random.default_rng(int(job['agent']['seed']))
        for _ in range(120):
            obs,r,t,tr,info=env.step(int(rng.integers(env.action_space.n)))
            if t or tr:break
        video=rec.finish();rec=None
        print('ENVIRONMENT_CHECK_PASSED | ordinary PPO contract + one cached match replay',flush=True)
        print('Action count:',len(spec['actions']),'| input:',list(obs) if isinstance(obs,dict) else ['image'],flush=True)
        return {'status':'ENVIRONMENT_CHECK_PASSED','training_performed':False,
                'game_config':pair,'trace_sha256':traces,'reset_seconds':resets,
                'action_names':[a['name'] for a in spec['actions']],
                'input_keys':list(obs) if isinstance(obs,dict) else ['image'],
                'input_definition':input_description(spec),'actual_image_shape':list(image_shape(spec)),
                'motion_frame_ids':None if motion_input is None else motion_input['frame_ids'],
                'video':video,'scope':'First PvE condition; real reset and stepped history, fixed 60-decision replay, per-observed-frame padding guard. Not an all-match crop certification or a winning model.'}
    finally:
        if rec:rec.close()
        env.close()


def actions_preview(job,out):
    """C hold/release or a named action test, always actual emulator output."""
    from sf2_game import Game
    from sf2_video import Video
    spec,_,_=student(job);p=schedule(job['pve'])[0]
    game=Game(job['common'],out/'game');videos=[];rows=[];rec=None
    try:
        tests=job.get('actions',{}).get('names',['C_hold','C_tap','NO_OP'])
        cmds=compiled(spec['actions']);names=[a['name'] for a in spec['actions']]
        for name in tests:
            game.open(spec['character'],p['character'],'pve',p['difficulty_stars'])
            rec=Video(out/(name+'.mp4'),game.fps);trace=[];previous=();presses=0
            for frame in range(240):
                if name=='C_hold':keys=('C',)
                elif name=='C_tap':keys=('C',) if frame%30<6 else ()
                elif name in names:
                    keys=cmds[names.index(name)][frame%6] if frame%60<6 else ()
                else:raise ValueError('Unknown preview name: '+name)
                presses+=int('C' in keys and 'C' not in previous);previous=keys
                raw,winner,reason=game.step_keys(keys,())
                rec.write(game.rgb,[f'ACTION CHECK | {name} | frame={frame}',
                    f'P1 HP={raw["p1_hp"]} | P2 HP={raw["p2_hp"]} | C press edges={presses}',
                    'No model. Native CPU active. Buttons are not an animation detector.'],keys,())
                trace.append({'frame':frame,'keys':list(keys),'raw':dict(raw)})
                if reason:break
            videos.append(rec.finish());rec=None
            json_write(out/(name+'_trace.json'),trace)
            rows.append({'name':name,'frames':len(trace),'C_press_edges':presses})
        return {'status':'ACTIONS_RECORDED','training_performed':False,'videos':videos,'cases':rows}
    finally:
        if rec:rec.close()
        game.close()


def summary(rows):
    n=len(rows);count={k:sum(r['result']==k for r in rows) for k in ('win','loss','draw','cutoff')}
    return {**count,'episodes':n,'win_fraction':count['win']/n if n else None,
            'mean_hp_margin':float(np.mean([max(0,r['self_hp'])-max(0,r['opponent_hp']) for r in rows])) if n else None}


def evaluate(job,out):
    import torch
    from sf2_video import Video
    cfg=job['evaluation'];rows=[];videos=[]
    offsets=cfg.get('offsets',[1,7,13,19,25,31,37,43,49,55])
    if not offsets or any(type(x) is not int or not 0<=x<=60 for x in offsets):
        raise ValueError('Evaluation offsets must be a nonempty list of integers in 0..60.')
    actors=[]
    trained=LoadedAgent(cfg['bundle'],trusted=job.get('trust_packages',False),device='cpu')
    actors.append(('trained',trained,bool(cfg.get('deterministic',True))))
    try:
        if rules_signature(trained.common)!=rules_signature(job['common']):raise ValueError('Evaluation arena rules differ from the package.')
        if cfg.get('compare_sampling',False):actors.append(('trained_sampled',trained,False))
        if cfg.get('initial_bundle'):
            initial=LoadedAgent(cfg['initial_bundle'],trusted=job.get('trust_packages',False),device='cpu')
            if initial.spec!=trained.spec or rules_signature(initial.common)!=rules_signature(trained.common):
                initial.close();raise ValueError('Initial and trained packages differ in actions, character or arena rules.')
            actors.insert(0,('initial',initial,bool(cfg.get('deterministic',True))))
        if cfg.get('include_random',True):actors.insert(0,('random',None,False))
        for label,actor,det in actors:
            # A separate evaluation process: these seeds do not alter training RNGs.
            torch.manual_seed(int(job['agent']['seed']))
            rng=np.random.default_rng(int(job['agent']['seed']))
            spec=trained.spec if actor is None else actor.spec
            reward_module=source_module(trained.directory/'reward.py','reward')
            env=BattleEnv(spec,job['common'],schedule(job['pve']),out/label,reward_module.compute_reward)
            rec=None
            try:
                for ci,pair in enumerate(schedule(job['pve'])):
                    env.opponents=[pair]
                    for j,offset in enumerate(offsets):
                        obs,reset=env.reset(seed=int(job['agent']['seed'])+ci*1000+j,options={'offset':offset})
                        record=cfg.get('record_video',True) and ci==0 and j==0
                        if record:
                            rec=Video(out/f'{label}.mp4',env.game.fps)
                            def frame(rgb,raw,k1,k2,number):
                                rec.write(rgb,[f'{label} | CPU stars={pair["difficulty_stars"]} | offset={offset}',
                                    f'P1 HP={raw["p1_hp"]} | P2 HP={raw["p2_hp"]} | frame={number}',
                                    'NO LEARNING | input highlight does not guarantee an executed move.'],k1,k2)
                            env.game.on_frame=frame
                        counts={};total_reward=0.;first_damage=None;decision=0
                        while True:
                            if actor is None:a=int(rng.integers(env.action_space.n))
                            else:a=actor.act(env.sides[env.player],env.game.raw,deterministic=det)
                            obs,r,t,tr,info=env.step(a);total_reward+=r;decision+=1
                            name=info['action_name'];counts[name]=counts.get(name,0)+1
                            if first_damage is None and info['damage_dealt']>0:first_damage=decision
                            if t or tr:break
                        row={k:info[k] for k in ('result','end_reason','self_hp','opponent_hp','native_frames')}
                        row.update(controller=label,deterministic=det,condition=pair,offset=offset,
                                   initial_readings=reset['raw'],action_counts=counts,
                                   first_damage_decision=first_damage,episode_reward=total_reward)
                        rows.append(row)
                        print(f'EVAL {label}: offset={offset}, {row["result"]}, HP={row["self_hp"]}/{row["opponent_hp"]}, reward={total_reward:+.4f}, reason={row["end_reason"]}',flush=True)
                        if rec:videos.append(rec.finish());rec=None
                        env.game.on_frame=None
            finally:
                if rec:rec.close()
                env.close()
        groups={label:summary([r for r in rows if r['controller']==label]) for label,_,_ in actors}
        print('EVALUATION SUMMARY:',json.dumps(groups),flush=True)
        return {'status':'EVALUATION_COMPLETE','training_performed':False,'episodes':rows,
                'summary':groups,'videos':videos,'win_fraction':groups['trained']['win_fraction'],
                'mean_hp_margin':groups['trained']['mean_hp_margin'],
                'note':'Offsets are start-time scenarios, not independently seeded native CPUs. Selection validation is not a held-out task.'}
    finally:
        closed=set()
        for _,actor,_ in actors:
            if actor is not None and id(actor) not in closed:actor.close();closed.add(id(actor))


class WorkerFactory:
    """Pickleable factory for spawn, one emulator in each worker process."""
    def __init__(self,job,rank):self.job=job;self.rank=rank
    def __call__(self):
        import torch,cv2
        torch.set_num_threads(1);cv2.setNumThreads(1)
        spec,reward,_=student(self.job)
        frozen=None
        if self.job['kind']=='train_frozen':
            frozen=LoadedAgent(self.job['frozen']['bundle'],trusted=self.job.get('trust_packages',False),device='cpu')
            opponents=[{'character':frozen.spec['character'],'difficulty_stars':1,
                        'neutral_jitter_max':self.job['frozen']['neutral_jitter_max']}]
            sides=tuple(self.job['frozen']['learner_sides'])
        else:opponents=schedule(self.job['pve']);sides=(1,)
        return BattleEnv(spec,self.job['common'],opponents,
                         Path(self.job['out'])/f'worker_{self.rank:02d}',reward,
                         opponent_agent=frozen,learner_sides=sides,
                         opponent_deterministic=bool(self.job.get('frozen',{}).get('opponent_deterministic',True)))


def check_frozen(job, out):
    """Real two-controller smoke test. No learner update, no strength claim."""
    from sf2_bundle import LoadedAgent
    from sf2_video import Video
    from sf2_observation import position
    spec, reward, module = student(job)
    cfg = validate_frozen_config(job['frozen'])
    sides = list(cfg['learner_sides'])
    if not sides or any(type(p) is not int or p not in (1,2) for p in sides):
        raise ValueError('Use learner_sides=[1], [2], or [1,2].')
    rows=[]; videos=[]
    for player in sides:
        opponent=LoadedAgent(cfg['bundle'],trusted=job.get('trust_packages',False),device='cpu')
        pair={'character':opponent.spec['character'],'difficulty_stars':1,'neutral_jitter_max':cfg['neutral_jitter_max']}
        env=None;rec=None
        try:
            env=BattleEnv(spec,job['common'],[pair],out/f'learner_P{player}',reward,
                opponent_agent=opponent,learner_sides=(player,),
                opponent_deterministic=bool(cfg.get('opponent_deterministic',True)))
            obs,_=env.reset(seed=int(job['agent']['seed']),options={'offset':0})
            learner_start=show_input(out/f'learner_P{player}',env,obs)
            if env.player!=player:raise RuntimeError('Requested learner port was not selected.')
            learner=module.build_model(env,dict(job['train'],n_steps=8,batch_size=8,n_epochs=1),
                                       int(job['agent']['seed']),'cpu')
            initialization=initialise_policy(learner,spec,job['sources'],job['common'],
                job.get('initialization_bundle',''),trusted=job.get('trust_packages',False))
            before=parameter_hash(learner)
            rec=Video(out/f'frozen_check_learner_P{player}.mp4',env.game.fps)
            def on_frame(rgb,raw,k1,k2,number):
                rec.write(rgb,[f'FROZEN ENV CHECK | learner=P{player} | opponent=P{3-player}',
                    f'P1 HP={raw["p1_hp"]} | P2 HP={raw["p2_hp"]} | frame={number}',
                    'No optimiser updates. This is a wiring check, not a winning-model test.'],k1,k2)
            env.game.on_frame=on_frame
            transitions=[]
            for decision in range(60):
                pre_elapsed=env.game.elapsed
                pre_raw=dict(env.game.raw)
                if spec['use_position']:
                    np.testing.assert_array_equal(obs['position'],position(pre_raw,player,job['common']['position_scale']))
                action,_=learner.predict(obs,deterministic=True)
                obs,r,terminated,truncated,info=env.step(action)
                advanced=env.game.elapsed-pre_elapsed
                if not (1<=advanced<=6):raise RuntimeError('Frozen mode advanced outside the six-frame protocol.')
                if not(terminated or truncated) and advanced!=6:raise RuntimeError('Nonterminal step did not run six frames.')
                if env.sides[1].frame_ids[-1]!=env.sides[2].frame_ids[-1]:
                    raise RuntimeError('P1/P2 observations do not share the same game clock.')
                if decision==11:show_input(out/f'learner_P{player}',env,obs,prefix='motion')
                transitions.append({'decision':decision,'learner_player':player,'action':int(action),
                    'reward':r,'result':info['result'],'end_reason':info['end_reason'],
                    'frames':advanced,'raw_before':pre_raw,'raw_after':dict(env.game.raw)})
                if terminated or truncated:break
            stats=env.performance_stats()
            if not stats['frozen_opponent_unchanged'] or parameter_hash(learner)!=before:
                raise RuntimeError('A model changed during the no-learning frozen check.')
            videos.append(rec.finish());rec=None
            json_write(out/f'frozen_check_P{player}_transitions.json',transitions)
            rows.append({'learner_player':player,'decisions':len(transitions),
                'initialization':initialization,'input_shape':learner_start['image_shape'],
                'learner_unchanged':True,'frozen_opponent_unchanged':True})
        finally:
            if rec:rec.close()
            if env:env.close()
            else:opponent.close()
    return {'status':'FROZEN_ENVIRONMENT_CHECK_PASSED','training_performed':False,
            'cases':rows,'videos':videos,
            'scope':'Requested ports; 60 decisions each; self-first positions, shared frame clock, unchanged model weights. Not proof of learning quality.'}


def train(job,out):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    cfg=dict(job['train']);spec,reward,module=student(job)
    if job['kind']=='train_frozen':
        job=dict(job,frozen=validate_frozen_config(job['frozen']),
                 pvp_validation=validate_pvp_evaluation(job.get('pvp_validation',{})))
    n=int(cfg['n_envs']);steps=int(cfg['n_steps']);batch=int(cfg['batch_size'])
    if n not in (1,2,4,8):raise ValueError('n_envs must be 1,2,4 or 8.')
    if steps<2 or batch<2 or steps*n%batch:raise ValueError('n_envs*n_steps must divide evenly into batch_size.')
    if int(cfg['total_timesteps'])<=0:raise ValueError('total_timesteps must be positive.')
    if cfg.get('resume_from'):raise ValueError('Exact optimiser resume is not supported here. Use initialization_bundle for a new phase from saved policy weights.')
    device=cfg.get('device','auto')
    if device=='auto':device='cuda:0' if torch.cuda.is_available() else 'cpu'
    if str(device).startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable.')
    torch.set_num_threads(int(cfg.get('torch_threads',2)))
    vec=None;model=None;completed=False;episodes=[];rollouts=[]
    source=Path(job['sources']);start=time.perf_counter()
    try:
        # Game engines run in CPU processes; all learner parameters stay in this one model.
        t=time.perf_counter()
        vec=(SubprocVecEnv([WorkerFactory(job,i) for i in range(n)],start_method='spawn')
             if n>1 else DummyVecEnv([WorkerFactory(job,0)]))
        torch.set_num_threads(int(cfg.get('torch_threads',2)))
        worker_start=time.perf_counter()-t
        model=module.build_model(vec,cfg,int(job['agent']['seed']),device)
        if not isinstance(model,PPO):raise TypeError('This classroom bundle supports ordinary stable_baselines3.PPO.')
        model.set_logger(configure(str(out/'logs'),['stdout','csv']))
        devices=sorted({str(p.device) for p in model.policy.parameters()})
        print('ACTUAL TRAINING CONFIG:',json.dumps({'algorithm':'PPO','n_envs':model.n_envs,
            'n_steps_per_env':steps,'rollout_total':n*steps,'devices':devices,
            'image_shape':list(image_shape(spec)),'observation_sha256':spec['observation']['source_sha256'],
            'action_count':len(spec['actions']),'input':'image+position6' if spec['use_position'] else 'image_only',
            'requested_steps':cfg['total_timesteps'],'effective_step_budget':math.ceil(cfg['total_timesteps']/(n*steps))*n*steps,
            'learning_rate':cfg['learning_rate'],'gamma':cfg['gamma'],'clip_range':cfg['clip_range'],
            'batch_size':batch,'n_epochs':cfg['n_epochs'],
            'profile':job.get('profile'), 'validation_target':validation_target(job),
            'learner_sides':job.get('frozen',{}).get('learner_sides',[1])}),flush=True)
        initialization = initialise_policy(model, spec, source, job['common'],
            job.get('initialization_bundle', ''), trusted=job.get('trust_packages', False))
        print('TRAINING INITIALIZATION:', json.dumps(initialization), flush=True)
        before=parameter_hash(model)
        def pack(name,label):
            return export_agent(out/name,model,spec,source/'student_model.py',source/'student_reward.py',job['common'],
                {'label':label,'train':cfg,'initialization':initialization,'steps':int(model.num_timesteps),'sources':{p.name:file_hash(p) for p in source.glob('*.py')},
                 'profile':job.get('profile'), 'validation_target':validation_target(job),
                 'opponent_sha256':file_hash(job['frozen']['bundle']) if job['kind']=='train_frozen' else None,
                 'learner_sides':job.get('frozen',{}).get('learner_sides',[1])})
        initial=pack('initial_agent.zip','untrained_control' if initialization['mode']=='from_scratch' else 'phase_start_control')
        class Audit(BaseCallback):
            def __init__(self):
                super().__init__();self.end=None;self.begin=None;self.best=None;self.saved=0;self.evaluated=0
                self.sampling_seconds=0.;self.update_seconds=0.;self.evaluation_seconds=0.;self.counts={}
            def _on_step(self):
                for rank,info in enumerate(self.locals['infos']):
                    name=info.get('action_name','unknown');self.counts[name]=self.counts.get(name,0)+1
                    if 'episode' in info:
                        row={k:info.get(k) for k in ('result','end_reason','self_hp','opponent_hp','self_player','native_frames','neutral_offset_frames')}
                        row.update(steps=int(self.num_timesteps),worker=rank,episode=info['episode'])
                        episodes.append(row)
                        with (out/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                return True
            def update_timing(self):
                if self.end is not None:
                    if model.device.type=='cuda':torch.cuda.synchronize(model.device)
                    seconds=time.perf_counter()-self.end;self.update_seconds+=seconds
                    rollouts[-1]['update_and_logging_seconds']=seconds
                    print(f'PERF envs={n} sample={rollouts[-1]["sample_seconds"]:.3f}s update+log={seconds:.3f}s',flush=True)
                    json_write(out/'performance_rollouts.json',rollouts);self.end=None
            def validate(self,package):
                t=time.perf_counter();dest=out/f'validation_{int(model.num_timesteps)}'
                ej=validation_job(job,package,dest)
                path=out/'evaluation_job.json';json_write(path,ej)
                subprocess.run([sys.executable,'-u',__file__,'--job',str(path)],check=True)
                data=json.loads((dest/'report.json').read_text())
                self.evaluation_seconds+=time.perf_counter()-t
                with (out/'validation_history.jsonl').open('a',encoding='utf-8') as history:
                    history.write(json.dumps({'steps':int(model.num_timesteps),
                        'target':validation_target(job),'learner_sides':data.get('learner_sides',[1]),
                        'opponent_sha256':data.get('opponent_sha256'),
                        'per_port':data.get('per_port'), 'win_fraction':data['win_fraction'],
                        'mean_hp_margin':data['mean_hp_margin'], 'report':str(dest/'report.json')})+'\n')
                score=(data['win_fraction'],data['mean_hp_margin'])
                if self.best is None or score>self.best:
                    self.best=score;shutil.copy2(package,out/'best_agent.zip')
                    json_write(out/'best_selection.json',{'steps':int(model.num_timesteps),'score':score,
                        'target':validation_target(job),'learner_sides':data.get('learner_sides',[1]),
                        'opponent_sha256':data.get('opponent_sha256'),'validation':str(dest)})
            def _on_rollout_start(self):
                self.update_timing();now=int(model.num_timesteps);package=None
                if cfg.get('checkpoint_every',0) and now-self.saved>=cfg['checkpoint_every']:
                    package=pack(f'checkpoint_{now}_agent.zip','checkpoint');self.saved=now
                if cfg.get('evaluate_every',0) and now-self.evaluated>=cfg['evaluate_every']:
                    package=package or pack(f'evaluation_{now}_agent.zip','validation_checkpoint')
                    self.validate(package);self.evaluated=now
                if model.device.type=='cuda':torch.cuda.synchronize(model.device)
                self.begin=time.perf_counter()
            def _on_rollout_end(self):
                if model.device.type=='cuda':torch.cuda.synchronize(model.device)
                seconds=time.perf_counter()-self.begin;self.sampling_seconds+=seconds
                rollouts.append({'steps':int(model.num_timesteps),'sample_seconds':seconds})
                recent = episodes[-100:]
                if recent:
                    for outcome in ('win', 'loss', 'draw', 'cutoff'):
                        self.logger.record('audit/train_'+outcome+'_fraction_100',
                            sum(row['result']==outcome for row in recent)/len(recent))
                    self.logger.record('audit/completed_training_episodes', len(episodes))
                self.end=time.perf_counter()
        audit=Audit()
        # PvP can select the starting policy if subsequent training regresses.
        # The evaluation subprocess cannot advance the learner's RNG or emulator.
        if job['kind']=='train_frozen' and cfg.get('evaluate_initial',True):audit.validate(initial)
        model.learn(total_timesteps=int(cfg['total_timesteps']),callback=audit,progress_bar=False)
        audit.update_timing()
        if before==parameter_hash(model):raise RuntimeError('PPO returned without changing any policy parameters.')
        latest=pack('last_agent.zip','last_trained')
        if cfg.get('evaluate_final',True):audit.validate(latest)
        # Same-device export/reload check: no emulator is created by the agent loader.
        held=model._last_obs
        def probs(m):
            m.policy.set_training_mode(False)
            data,_=m.policy.obs_to_tensor(held)
            with torch.no_grad():return m.policy.get_distribution(data).distribution.probs.cpu().numpy()
        expected=probs(model);loaded=LoadedAgent(latest,trusted=True,device=device)
        try:error=float(np.max(np.abs(expected-probs(loaded.model))))
        finally:loaded.close()
        if error>1e-5:raise RuntimeError(f'Export/reload probability mismatch: {error}')
        active=audit.sampling_seconds+audit.update_seconds
        worker_stats = vec.env_method('performance_stats')
        if job['kind']=='train_frozen' and not all(row.get('frozen_opponent_unchanged') is True for row in worker_stats):
            raise RuntimeError('A frozen opponent changed parameters during training.')
        completed=True
        return {'status':'TRAINING_COMPLETE' ,'training_performed':True,'algorithm':'PPO',
                'n_envs':n,'devices':devices,'actual_timesteps':int(model.num_timesteps),
                'initial_agent':initial,'last_agent':latest,
                'best_agent':str(out/'best_agent.zip') if (out/'best_agent.zip').exists() else None,
                'sampling_seconds':audit.sampling_seconds,'update_and_logging_seconds':audit.update_seconds,
                'evaluation_seconds':audit.evaluation_seconds,'worker_start_seconds':worker_start,
                'active_steps_per_second':model.num_timesteps/max(active,1e-9),
                'worker_stats':worker_stats,'action_counts':audit.counts,
                'initialization':initialization,
                'validation_target':validation_target(job),
                'learner_sides':job.get('frozen',{}).get('learner_sides',[1]),
                'opponent_sha256':file_hash(job['frozen']['bundle']) if job['kind']=='train_frozen' else None,
                'learner_parameters_changed':True,
                'frozen_opponent_unchanged':(all(row.get('frozen_opponent_unchanged') is True for row in worker_stats) if job['kind']=='train_frozen' else None),
                'export_reload_probability_error':error,'episodes':episodes,
                'note':'PPO step budget rounds up to a full rollout. Speed excludes independent evaluation/startup/export. Best uses the explicitly reported target: CPU PvE or the fixed agent on the configured learner ports. A PvP best may remain at step zero. A complete run is not proof of a winning baseline.'}
    finally:
        if model is not None and not completed:
            try:model.save(out/'interrupted_policy.zip')
            except Exception:pass
        if vec is not None:vec.close()


def pvp(job,out):
    from sf2_game import Game
    from sf2_observation import Side
    from sf2_video import Video
    cfg=job['pvp'];actors=[];game=None;rec=None;rows=[];videos=[]
    try:
        for key in ('agent_a','agent_b'):
            actors.append(LoadedAgent(cfg[key],trusted=job.get('trust_packages',False),device='cpu'))
        for a in actors:
            if rules_signature(a.common)!=rules_signature(job['common']):raise ValueError('Both agents must use the same arena rules.')
        before=[parameter_hash(a.model) for a in actors]
        order=[(0,1),(1,0)] if cfg.get('swap_sides',True) else [(0,1)]
        rounds=int(cfg.get('rounds_per_side',1))
        if not 1<=rounds<=20:raise ValueError('rounds_per_side must be 1..20.')
        game=Game(job['common'],out/'game')
        for pair in order:
            a,b=(actors[i] for i in pair)
            for rep in range(rounds):
                game.open(a.spec['character'],b.spec['character'],'pvp',1)
                sides=[Side(a.spec,1,job['common']['position_scale']),Side(b.spec,2,job['common']['position_scale'])]
                for s in sides:s.reset(game.rgb,frame_id=game.elapsed)
                commands=[compiled(a.spec['actions']),compiled(b.spec['actions'])]
                rec=Video(out/f'pvp_{len(rows):02d}.mp4',game.fps)
                game.on_frame=lambda rgb,raw,k1,k2,frame:rec.write(rgb,
                    [f'PvP | P1={a.spec["team_name"]} | P2={b.spec["team_name"]}',
                     f'P1 HP={raw["p1_hp"]} | P2 HP={raw["p2_hp"]} | frame={frame}',
                     'NO LEARNING | synchronous decisions; no wall-time tournament deadline.'],k1,k2)
                winner=reason=None
                while reason is None:
                    # Both act on the same pre-step game state, even for identical characters.
                    raw=dict(game.raw)
                    ids=[a.act(sides[0],raw,bool(cfg.get('deterministic',True))),
                         b.act(sides[1],raw,bool(cfg.get('deterministic',True)))]
                    for k1,k2 in zip(commands[0][ids[0]],commands[1][ids[1]]):
                        raw,winner,reason=game.step_keys(k1,k2)
                        if reason:break
                    for s in sides:s.update(game.rgb,frame_id=game.elapsed)
                result=(('A' if pair[int(winner[-1])-1]==0 else 'B') if winner in ('p1','p2') else ('cutoff' if reason=='frame_cap' else winner))
                rows.append({'p1_agent':'A' if pair[0]==0 else 'B','p2_agent':'A' if pair[1]==0 else 'B',
                             'winner':result,'end_reason':reason,'p1_hp':raw['p1_hp'],'p2_hp':raw['p2_hp'],
                             'native_frames':game.elapsed})
                videos.append(rec.finish());rec=None;game.on_frame=None
        if before!=[parameter_hash(a.model) for a in actors]:raise RuntimeError('PvP altered policy parameters.')
        return {'status':'PVP_COMPLETE','training_performed':False,'rounds':rows,'videos':videos,
                'note':'Port reassignment is checked by the interface; P1 training does not guarantee P2 skill. No latency deadline is enforced.'}
    finally:
        if rec:rec.close()
        if game:game.close()
        for a in actors:a.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args()
    job=json.loads(Path(args.job).read_text());out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    json_write(out/'job.json',job);t=time.perf_counter();code=1
    try:
        import torch,cv2
        torch.set_num_threads(int(job['train'].get('torch_threads',2)));cv2.setNumThreads(1)
        fn={'check':check,'actions':actions_preview,'train':train,'train_frozen':train,
            'evaluate':evaluate,'pvp':pvp,'check_frozen':check_frozen,'evaluate_frozen':evaluate_frozen}[job['kind']]
        report=fn(job,out);code=0
    except BaseException as exc:
        report={'status':'ERROR','error':repr(exc),'traceback':traceback.format_exc()}
        print(report['traceback'],flush=True)
    report.update(wall_seconds=time.perf_counter()-t,runtime_versions=runtime_versions())
    json_write(out/'report.json',report);print('REPORT='+str(out/'report.json'),flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
