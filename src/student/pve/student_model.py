"""Ordinary SB3 PPO + Nature-style CNN + a minimal optional position branch.
No MaskablePPO, control vector, frame-difference images or hand-crafted geometry.
With use_position=False, use the old standard CnnPolicy (NatureCNN) directly.
"""
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

MODEL_CONFIG = {'image_features':512, 'position_features':64}

class ImagePositionExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_space,image_features=512,position_features=64):
        super().__init__(observation_space,features_dim=image_features+position_features)
        # The three convolution sizes/strides match SB3 NatureCNN.
        cnn=nn.Sequential(
            nn.Conv2d(observation_space['image'].shape[0],32,8,4),nn.ReLU(),
            nn.Conv2d(32,64,4,2),nn.ReLU(),
            nn.Conv2d(64,64,3,1),nn.ReLU(),nn.Flatten())
        with torch.no_grad():
            flattened=cnn(torch.zeros(1,*observation_space['image'].shape)).shape[1]
        self.image_net=nn.Sequential(cnn,nn.Linear(flattened,image_features),nn.ReLU())
        self.position_net=nn.Sequential(nn.Linear(6,position_features),nn.ReLU())
    def forward(self,obs):
        # SB3 normalizes the uint8 image once; do not divide by 255 again here.
        return torch.cat((self.image_net(obs['image']),self.position_net(obs['position'])),dim=1)

def build_model(env,config,seed,device='auto'):
    has_position=hasattr(env.observation_space,'spaces')
    policy='MultiInputPolicy' if has_position else 'CnnPolicy'
    kwargs={}
    if has_position:
        kwargs={'features_extractor_class':ImagePositionExtractor,
                'features_extractor_kwargs':dict(MODEL_CONFIG),
                'net_arch':dict(pi=[],vf=[])}
    return PPO(policy,env,
        learning_rate=float(config['learning_rate']),n_steps=int(config['n_steps']),
        batch_size=int(config['batch_size']),n_epochs=int(config['n_epochs']),
        gamma=float(config['gamma']),gae_lambda=float(config.get('gae_lambda',0.95)),
        clip_range=float(config['clip_range']),ent_coef=float(config.get('ent_coef',0.0)),
        vf_coef=float(config.get('vf_coef',0.5)),max_grad_norm=0.5,
        target_kl=config.get('target_kl'),seed=int(seed),device=device,verbose=1,
        policy_kwargs=kwargs)
