"""Small delivery helpers. No changes to the default PPO objective or game rules.

A saved learner can optionally initialise a NEW training phase. Only policy
weights are copied; the optimiser, rollout buffer and step count are new.
This is deliberately not advertised as an exact interrupted-run resume.
"""
from pathlib import Path
import hashlib
import json


def validate_frozen_config(frozen):
    """One fixed opponent and explicit learner ports; no native-CPU difficulty."""
    if not isinstance(frozen, dict):
        raise TypeError('PvP opponent settings must be a dictionary.')
    permitted={'bundle','learner_sides','opponent_deterministic','neutral_jitter_max'}
    if set(frozen)-permitted:
        raise ValueError('Unsupported PvP settings: '+str(sorted(set(frozen)-permitted)))
    bundle=frozen.get('bundle','')
    if not isinstance(bundle,str) or not bundle.strip():
        raise ValueError('Set the fixed opponent bundle path explicitly. It is not the learner initialisation path.')
    sides=frozen.get('learner_sides',[2])
    if (not isinstance(sides,(tuple,list)) or not sides
        or any(type(p) is not int or p not in (1,2) for p in sides)
        or len(set(sides))!=len(sides)):
        raise ValueError('learner_sides must be [1], [2], [1,2] or [2,1]. P2 is a port, not a permanent screen side.')
    deterministic=frozen.get('opponent_deterministic',True)
    if type(deterministic) is not bool:raise ValueError('opponent_deterministic must be True or False.')
    jitter=frozen.get('neutral_jitter_max',60)
    if type(jitter) is not int or not 0<=jitter<=60:raise ValueError('PvP neutral_jitter_max must be 0..60.')
    return {'bundle':str(Path(bundle).expanduser().resolve()),'learner_sides':list(sides),
            'opponent_deterministic':deterministic,'neutral_jitter_max':jitter}


def validate_pvp_evaluation(config):
    """Same requested ports/opponent as training; seed/offsets describe scenarios."""
    c=dict(config or {})
    allowed={'offsets','seed','deterministic','record_video'}
    if set(c)-allowed:raise ValueError('Unsupported PvP validation field: '+str(sorted(set(c)-allowed)))
    offsets=c.get('offsets',[0,12,24,36,48])
    if (not isinstance(offsets,list) or not offsets or len(offsets)>61
        or any(type(n) is not int or not 0<=n<=60 for n in offsets)
        or len(offsets)!=len(set(offsets))):
        raise ValueError('Use distinct integer PvP offsets in 0..60.')
    seed=c.get('seed',2025)
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('PvP evaluation seed must be an integer in 0..2**31-1.')
    for name in ('deterministic','record_video'):
        if name in c and type(c[name]) is not bool:raise ValueError(name+' must be True or False.')
    return {'offsets':offsets,'seed':seed,'deterministic':c.get('deterministic',True),
            'record_video':c.get('record_video',False)}


def frozen_check_fingerprint(root, agent, common, pve, frozen, *, initialization_bundle='', train=None):
    from lab_notebook import input_check_fingerprint
    c=validate_frozen_config(frozen)
    path=Path(c['bundle'])
    if not path.is_file():raise FileNotFoundError('Frozen opponent package does not exist: '+str(path))
    initial=Path(initialization_bundle).expanduser().resolve() if initialization_bundle else None
    if initial is not None and not initial.is_file():raise FileNotFoundError('Initial learner package does not exist: '+str(initial))
    return {'learner':input_check_fingerprint(root,agent,common,None,profile='pvp'),
            'opponent_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'initialization_sha256':hashlib.sha256(initial.read_bytes()).hexdigest() if initial else None,
            'learner_sides':c['learner_sides'],'opponent_deterministic':c['opponent_deterministic'],
            'neutral_jitter_max':c['neutral_jitter_max'], 'train':dict(train or {})}


def initialise_policy(model, spec, source, common, package, trusted=False):
    """Optional own-agent phase initialisation, with strict input/action checks."""
    from sf2_bundle import LoadedAgent
    from sf2_common import rules_signature, file_hash, parameter_hash
    if not package:
        return {'mode': 'from_scratch', 'initial_parameter_sha256': parameter_hash(model)}
    parent = LoadedAgent(package, trusted=trusted, device='cpu')
    try:
        # Team display names do not change the learned control task.
        a = {k: v for k, v in spec.items() if k != 'team_name'}
        b = {k: v for k, v in parent.spec.items() if k != 'team_name'}
        if a != b:
            raise ValueError('Initial learner differs in character, actions or exact input source. Start from scratch or use matching student files.')
        if rules_signature(common) != rules_signature(parent.common):
            raise ValueError('Initial learner uses different arena rules.')
        if file_hash(Path(source) / 'student_model.py') != file_hash(parent.directory / 'model.py'):
            raise ValueError('Initial learner model source differs. Do not silently reinterpret existing weights.')
        expected = parameter_hash(parent.model)
        model.policy.load_state_dict(parent.model.policy.state_dict(), strict=True)
        actual = parameter_hash(model)
        if actual != expected:
            raise RuntimeError('Policy initialisation did not copy parameters exactly.')
        return {
            'mode': 'saved_policy_new_optimizer',
            'package': str(Path(package).expanduser().resolve()),
            'package_sha256': file_hash(package),
            'source_steps': int(parent.model.num_timesteps),
            'initial_parameter_sha256': actual,
            'optimizer_restored': False,
            'note': 'This phase starts at step 0 with a new optimizer and new environment histories. It is not an exact resume.',
        }
    finally:
        parent.close()


def read_package_metadata(path):
    """Read plain JSON metadata without loading Python or deserialising weights."""
    import zipfile
    path = Path(path).expanduser().resolve()
    with zipfile.ZipFile(path) as z:
        result = {}
        for name in ('manifest.json', 'spec.json', 'training.json', 'common.json'):
            info = z.getinfo(name)
            if info.file_size > 2 * 1024 * 1024:
                raise ValueError('Metadata too large: ' + name)
            result[name] = json.loads(z.read(name))
    return result


def validation_target(job):
    return 'frozen_agent_pvp' if job['kind']=='train_frozen' else 'native_cpu_pve_P1'


def validation_job(job, package, destination):
    """Select the correct validation task; PvP can never fall back to CPU PvE."""
    import copy
    j=copy.deepcopy(job)
    j.update(out=str(destination),trust_packages=True)
    if job['kind']=='train_frozen':
        cfg=validate_pvp_evaluation(job.get('pvp_validation',{}))
        cfg['record_video']=False
        j.update(kind='evaluate_frozen',pve={},
                 evaluation={'bundle':str(package),'initial_bundle':'','compare_initial':False,**cfg})
        j['frozen']=validate_frozen_config(job['frozen'])
    else:
        j.update(kind='evaluate',evaluation={'bundle':str(package),
            'offsets':job.get('validation_offsets',[0,12,24,36,48]),
            'include_random':False,'record_video':False,'deterministic':True})
    return j


def resolve_pvp_setup(setup):
    """Explicit learner initialisation, never an implicit 'latest PvE' fallback."""
    required={'learner_start','learner_bundle','opponent_bundle','learner_sides',
              'opponent_deterministic','neutral_jitter_max'}
    if not isinstance(setup,dict) or set(setup)!=required:
        raise ValueError('PVP_SETUP fields do not match the notebook template.')
    mode=setup['learner_start'];bundle=setup['learner_bundle']
    if mode not in ('package','scratch'):raise ValueError('learner_start must be package or scratch.')
    if not isinstance(bundle,str):raise ValueError('learner_bundle must be a path string.')
    if mode=='package' and not bundle.strip():
        raise ValueError('Set learner_bundle to the model to continue learning, or explicitly choose learner_start="scratch".')
    if mode=='scratch' and bundle.strip():
        raise ValueError('learner_start="scratch" requires learner_bundle="". Do not leave a misleading old path.')
    initial=str(Path(bundle).expanduser().resolve()) if bundle else ''
    frozen=validate_frozen_config({'bundle':setup['opponent_bundle'],
        'learner_sides':setup['learner_sides'],'opponent_deterministic':setup['opponent_deterministic'],
        'neutral_jitter_max':setup['neutral_jitter_max']})
    for role,path in [('initial learner',initial),('fixed opponent',frozen['bundle'])]:
        if path and not Path(path).is_file():raise FileNotFoundError(role+' package not found: '+path)
    print('PvP learner:',initial if initial else 'NEW RANDOM MODEL',flush=True)
    print('PvP frozen opponent:',frozen['bundle'],flush=True)
    print('Learner ports:',frozen['learner_sides'],'| opponent ports:',[3-p for p in frozen['learner_sides']],flush=True)
    return initial,frozen
