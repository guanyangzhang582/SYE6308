"""Installed-SB3 shape/export test on synthetic images, NOT a game or training run."""
from pathlib import Path
import argparse,json,tempfile
import numpy as np

def main():
    import gymnasium as gym
    import torch
    from sf2_common import source_module
    from sf2_bundle import make_spec,export_agent,LoadedAgent
    from sf2_actions import library
    from sf2_observation import Side,observation_space,image_shape,description
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args()
    job=json.loads(Path(args.job).read_text());source=Path(job['sources'])
    torch.set_num_threads(2)
    actions=source_module(source/'student_actions.py','actions')
    spec=make_spec(job['agent'],library(actions.CUSTOM_ACTIONS),source/'student_observation.py')
    cfg=dict(job['train'],n_steps=8,batch_size=8,n_epochs=1)
    common=job['common']
    obs_space=observation_space(spec['use_position'],common['position_scale'],spec=spec)
    act_space=gym.spaces.Discrete(len(spec['actions']))
    raw={'p1_x':200,'p1_y':192,'p2_x':390,'p2_y':180}
    rgb=np.zeros((224,320,3),np.uint8)
    x0,y0,x1,y1=spec['observation']['config']['crop_xyxy']
    rgb[y0:y1,x0:x1]=[120,160,200]
    side=Side(spec,1,common['position_scale']);side.reset(rgb)
    observed=side.observe(raw)
    if not obs_space.contains(observed):raise RuntimeError('Synthetic observation is outside declared space.')
    class ShapeOnly(gym.Env):
        observation_space=obs_space
        action_space=act_space
    module=source_module(source/'student_model.py','model')
    model=module.build_model(ShapeOnly(),cfg,int(job['agent']['seed']),'cpu')
    from stable_baselines3 import PPO
    if not isinstance(model,PPO):raise TypeError('Expected ordinary stable_baselines3.PPO.')
    model.policy.set_training_mode(False)
    def probs(m):
        obs,_=m.policy.obs_to_tensor(observed)
        with torch.no_grad():return m.policy.get_distribution(obs).distribution.probs.cpu().numpy()
    before=probs(model)
    with tempfile.TemporaryDirectory() as td:
        package=export_agent(Path(td)/'synthetic_shape_only.zip',model,spec,source/'student_model.py',source/'student_reward.py',common,{'scope':'Synthetic contract only; never trained'})
        loaded=LoadedAgent(package,trusted=True,device='cpu')
        try:
            error=float(np.max(np.abs(before-probs(loaded.model))))
            if error>1e-6:raise RuntimeError('CPU export/reload prediction mismatch.')
            index=loaded.act(side,raw,deterministic=True)
            if not 0<=index<len(spec['actions']):raise RuntimeError('Invalid action index.')
        finally:loaded.close()
    info={'status':'SYNTHETIC_INPUT_CONTRACT_PASSED','scope':'Installed PPO on synthetic input; no emulator, no learning, no useful model saved',
          'image_shape':list(image_shape(spec)),'source_sha256':spec['observation']['source_sha256'],
          'parameter_count':sum(p.numel() for p in model.policy.parameters()),'reload_probability_error':error,
          'input_definition':description(spec)}
    print(json.dumps(info,indent=2),flush=True)
    Path(job['report']).write_text(json.dumps(info,indent=2),encoding='utf-8')

if __name__=='__main__':main()
