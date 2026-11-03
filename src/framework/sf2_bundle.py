"""Portable agent = weights + exact actions + custom model source + protocol metadata.
Only load trusted packages: Python custom classes / SB3 checkpoints can execute code.
Checksums detect changes; they do NOT make untrusted Python safe.
"""
from __future__ import annotations
from pathlib import Path,PurePosixPath
import hashlib,json,shutil,tempfile,zipfile
import numpy as np
from sf2_common import BUNDLE_SCHEMA,ENV_PROTOCOL,OBS_PROTOCOL,json_write,file_hash,source_module,rules_signature,runtime_versions
from sf2_actions import signature,validate_library


def make_spec(agent,actions,observation_source):
    from sf2_observation import build_descriptor
    from sf2_boot import normalize_character
    return {'team_name':str(agent['team_name']),'character':normalize_character(agent['character']),
            'actions':validate_library(actions),'action_hash':signature(actions),
            'use_position':bool(agent.get('use_position',True)),
            'observation':build_descriptor(observation_source)}


def export_agent(path,model,spec,model_source,reward_source,common,training=None):
    from sf2_observation import observation_space,description
    meta=description(spec)
    if model.observation_space!=observation_space(spec.get('use_position',True),common['position_scale'],spec=spec):
        raise ValueError('Cannot export mismatched model/observation specification.')
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);model.save(d/'policy.zip')
        shutil.copy2(model_source,d/'model.py');shutil.copy2(reward_source,d/'reward.py')
        (d/'observation.py').write_text(spec['observation']['source'],encoding='utf-8')
        json_write(d/'spec.json',spec);json_write(d/'common.json',common);json_write(d/'training.json',training or {})
        manifest={'schema':BUNDLE_SCHEMA,'env_protocol':ENV_PROTOCOL,'observation_protocol':OBS_PROTOCOL,
                  'observation':meta,
                  'algorithm':'PPO','team_name':spec['team_name'],'character':spec['character'],
                  'action_count':len(spec['actions']),'action_hash':signature(spec['actions']),
                  'model_source_sha256':file_hash(d/'model.py'),'rules':rules_signature(common),
                  'input_keys':['image','position'] if spec.get('use_position',True) else ['image'],'image_mirrored':False,
                  'runtime_versions':runtime_versions(),
                  'security':'Trusted Python only. Checksums are not a sandbox.'}
        json_write(d/'manifest.json',manifest)
        json_write(d/'checksums.json',{p.name:file_hash(p) for p in sorted(d.iterdir()) if p.is_file()})
        temp=path.with_suffix('.building.zip')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(d.iterdir()):z.write(p,p.name)
        temp.replace(path)
    return str(path)


class LoadedAgent:
    def __init__(self,path,*,trusted=False,device='cpu'):
        if not trusted:raise ValueError('Read/review this package first, then set TRUST_AGENT_PACKAGES=True. Loading Python is not sandboxed.')
        self.path=Path(path).resolve();self.directory=Path(tempfile.mkdtemp(prefix='sf2_agent_'));self.model=None
        try:
            with zipfile.ZipFile(self.path) as z:
                infos=z.infolist()
                allowed={'policy.zip','model.py','reward.py','observation.py','spec.json','common.json','training.json','manifest.json','checksums.json'}
                if set(z.namelist())!=allowed or len(infos)!=len(allowed):raise ValueError('This input upgrade requires a classroom v3 package. Old v2/image-size checkpoints must use the old notebook; do not silently convert weights.')
                if sum(i.file_size for i in infos)>512*1024**2:raise ValueError('Agent package too large')
                for i in infos:
                    if PurePosixPath(i.filename).name!=i.filename or (i.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe archive member')
                    (self.directory/i.filename).write_bytes(z.read(i))
            checks=json.loads((self.directory/'checksums.json').read_text())
            if set(checks)!=allowed-{'checksums.json'}:raise ValueError('Missing package checksum')
            for n,h in checks.items():
                if file_hash(self.directory/n)!=h:raise ValueError('Package checksum mismatch: '+n)
            self.manifest=json.loads((self.directory/'manifest.json').read_text())
            self.spec=json.loads((self.directory/'spec.json').read_text());self.common=json.loads((self.directory/'common.json').read_text())
            m=self.manifest
            if m['schema']!=BUNDLE_SCHEMA or m['env_protocol']!=ENV_PROTOCOL or m['observation_protocol']!=OBS_PROTOCOL or m['algorithm']!='PPO':raise ValueError('Unsupported agent protocol')
            if signature(self.spec['actions'])!=m['action_hash'] or self.spec.get('action_hash')!=m['action_hash'] or len(self.spec['actions'])!=m['action_count'] or self.spec['character']!=m['character']:raise ValueError('Agent/action identity mismatch')
            if m['rules']!=rules_signature(self.common):raise ValueError('Rules metadata mismatch')
            if file_hash(self.directory/'model.py')!=m['model_source_sha256']:raise ValueError('Model source mismatch')
            from sf2_observation import description
            if m.get('observation')!=description(self.spec):raise ValueError('Observation metadata mismatch')
            if (self.directory/'observation.py').read_text(encoding='utf-8')!=self.spec['observation']['source']:
                raise ValueError('Preprocessing source differs from the spec.')
            source_module(self.directory/'model.py','model')
            from stable_baselines3 import PPO
            self.model=PPO.load(self.directory/'policy.zip',device=device)
            from sf2_observation import observation_space
            if self.manifest.get('input_keys') != (['image','position'] if self.spec.get('use_position',True) else ['image']):raise ValueError('Input metadata mismatch')
            if self.model.action_space.n!=len(self.spec['actions']):raise ValueError('Policy output does not match exported action table')
            if self.model.observation_space!=observation_space(self.spec.get('use_position',True),self.common['position_scale'],spec=self.spec):raise ValueError('Observation specification mismatch')
            self.model.policy.set_training_mode(False)
        except BaseException:
            self.close();raise
    def act(self,side,raw,deterministic=True):
        if side.spec['action_hash']!=self.spec['action_hash']:raise ValueError('Wrong action table for this model')
        if side.spec['observation']['source_sha256']!=self.spec['observation']['source_sha256'] or side.spec.get('use_position',True)!=self.spec.get('use_position',True):
            raise ValueError('Wrong observation definition for this policy.')
        if side.scale!=self.common['position_scale']:raise ValueError('Wrong position scale for this policy.')
        a,_=self.model.predict(side.observe(raw),deterministic=deterministic)
        return int(np.asarray(a).item())
    def close(self):
        self.model=None
        if getattr(self,'directory',None):shutil.rmtree(self.directory,ignore_errors=True)
