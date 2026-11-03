"""Select one immutable v3 agent and wrap it for the public practice arena.

Static inspection NEVER imports student code or deserializes policy tensors.
A static pass is a packaging/protocol result, NOT a security or game-skill certificate.
"""
from __future__ import annotations
import ast
import datetime as dt
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile

from sf2_actions import signature, validate_library
from sf2_common import BUNDLE_SCHEMA, ENV_PROTOCOL, OBS_PROTOCOL, rules_signature
from sf2_observation import _validate_config

AGENT_FILES = frozenset({'policy.zip','model.py','reward.py','observation.py','spec.json',
                         'common.json','training.json','manifest.json','checksums.json'})
SUBMISSION_FILES = frozenset({'agent.zip','submission.json','static_check.json','README.txt'})
MAX_AGENT = 512 * 1024**2
MAX_OUTER = MAX_AGENT + 4 * 1024**2
JSON_LIMIT = 2 * 1024**2
SOURCE_LIMIT = 512 * 1024
SUBMISSION_SCHEMA = 'sye6308-submission-v1'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def json_bytes(data: bytes):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: '+str(key))
            result[key] = value
        return result
    def no_constant(value):
        raise ValueError('Non-finite JSON number: '+value)
    if len(data) > JSON_LIMIT:
        raise ValueError('JSON metadata exceeds the supported size.')
    return json.loads(data.decode('utf-8'), object_pairs_hook=no_duplicates, parse_constant=no_constant)


def _read_archive(data: bytes, allowed, total_limit: int, *, binary_names):
    if not data or len(data) > total_limit:
        raise ValueError('Archive is empty or exceeds the supported size.')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = z.infolist()
        names = [x.filename for x in infos]
        if len(names) != len(allowed) or set(names) != set(allowed):
            raise ValueError('Unexpected/missing/duplicate archive files. Expected: '+', '.join(sorted(allowed)))
        if sum(x.file_size for x in infos) > total_limit:
            raise ValueError('Uncompressed archive exceeds the supported size.')
        result = {}
        for x in infos:
            if (x.filename != PurePosixPath(x.filename).name or '\\' in x.filename
                    or x.is_dir() or stat.S_ISLNK(x.external_attr >> 16) or x.flag_bits & 1):
                raise ValueError('Unsafe or encrypted archive member: '+x.filename)
            limit = total_limit if x.filename in binary_names else SOURCE_LIMIT if x.filename.endswith('.py') else JSON_LIMIT
            if x.file_size > limit:
                raise ValueError('Archive member exceeds limit: '+x.filename)
            result[x.filename] = z.read(x)  # CRC checked by zipfile; never extract arbitrary paths.
        return result


def load_rules(path):
    r = json_bytes(Path(path).read_bytes())
    if r.get('schema') != 'sye6308-practice-rules-v1':
        raise ValueError('Unsupported arena rules schema.')
    rules_signature(r['common'])
    if r['inference']['device'] != 'cpu' or r['inference']['torch_threads'] != 1:
        raise ValueError('This practice build uses CPU inference with one torch thread per actor.')
    for key in ('load_timeout_seconds', 'decision_timeout_seconds'):
        if not 0 < float(r['inference'][key]) <= 600:
            raise ValueError('Invalid timeout rule.')
    if r['inference']['deterministic'] is not True:
        raise ValueError('This practice ruleset uses deterministic inference.')
    if not r['allowed_characters']:
        raise ValueError('The rules need an explicit character list.')
    return r


def rules_hash(rules):
    return digest(canonical(rules))


def inspect_agent_bytes(data: bytes, rules: dict):
    f = _read_archive(data, AGENT_FILES, MAX_AGENT, binary_names={'policy.zip'})
    hashes = json_bytes(f['checksums.json'])
    if set(hashes) != AGENT_FILES-{'checksums.json'}:
        raise ValueError('The agent checksum list is incomplete.')
    for name, expected in hashes.items():
        if digest(f[name]) != expected:
            raise ValueError('Agent checksum mismatch: '+name)
    for name in ('model.py','observation.py','reward.py'):
        ast.parse(f[name].decode('utf-8'), filename=name)  # Syntax only; do not execute.
    spec, common, manifest = (json_bytes(f[n]) for n in ('spec.json','common.json','manifest.json'))
    training = json_bytes(f['training.json'])
    if not isinstance(training, dict):
        raise ValueError('training.json must be an object.')
    if (manifest.get('schema') != BUNDLE_SCHEMA or manifest.get('env_protocol') != ENV_PROTOCOL
            or manifest.get('observation_protocol') != OBS_PROTOCOL or manifest.get('algorithm') != 'PPO'):
        raise ValueError('Only classroom v3 ordinary-PPO agents are supported; raw .pth/.zip weights are not a submission.')
    validate_library(spec['actions'])
    action_hash = signature(spec['actions'])
    if spec.get('action_hash') != action_hash or manifest.get('action_hash') != action_hash:
        raise ValueError('Action table/hash mismatch.')
    if manifest.get('action_count') != len(spec['actions']):
        raise ValueError('Action count mismatch.')
    if spec['character'] != manifest.get('character') or spec['character'] not in rules['allowed_characters']:
        raise ValueError('Character is not allowed by these rules or differs from the manifest.')
    if spec['team_name'] != manifest.get('team_name'):
        raise ValueError('Stored team name differs from the agent manifest.')
    if not isinstance(spec.get('use_position'), bool):
        raise ValueError('use_position must be a bool.')
    if rules_signature(common) != rules_signature(rules['common']) or manifest.get('rules') != rules_signature(common):
        raise ValueError('ARENA_RULE_MISMATCH: game speed/decision interval/coordinate scale/frame cap must match the public arena.')
    obs = spec['observation']
    if set(obs) != {'schema','source_sha256','config','source'} or obs['schema'] != 'sf2-student-observation-v1':
        raise ValueError('Unsupported observation descriptor.')
    c = _validate_config(obs['config'])
    if obs['source'].encode('utf-8') != f['observation.py'] or digest(f['observation.py']) != obs['source_sha256']:
        raise ValueError('Observation source/hash mismatch.')
    shape = [4 * (1 if c['color'] == 'gray' else 3), c['height'], c['width']]
    expected_obs = {'protocol':OBS_PROTOCOL, 'source_sha256':obs['source_sha256'],
        'config':c, 'image_shape':shape, 'dtype':'uint8', 'normalization':'SB3 divides by 255 once',
        'channel_order':'oldest to latest, then R/G/B within each RGB frame',
        'decision_frames':6, 'position_count':6 if spec['use_position'] else 0}
    if manifest.get('observation') != expected_obs:
        raise ValueError('Observation metadata mismatch.')
    if manifest.get('input_keys') != (['image','position'] if spec['use_position'] else ['image']):
        raise ValueError('Input key mismatch.')
    if manifest.get('image_mirrored') is not False:
        raise ValueError('Unexpected image mirroring protocol.')
    if digest(f['model.py']) != manifest.get('model_source_sha256'):
        raise ValueError('Model source/hash mismatch.')
    # Inspect nested SB3 container sizes/names without unpickling or torch.load.
    with zipfile.ZipFile(io.BytesIO(f['policy.zip'])) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if ('data' not in names or 'policy.pth' not in names or len(names) > 32
                or len(set(names)) != len(names) or sum(i.file_size for i in infos) > MAX_AGENT):
            raise ValueError('Malformed or oversized inner SB3 policy archive.')
        for i in infos:
            if i.filename != PurePosixPath(i.filename).name or '\\' in i.filename or i.flag_bits & 1 or stat.S_ISLNK(i.external_attr >> 16):
                raise ValueError('Unsafe inner policy archive member.')
    return {'status':'STATIC_PACKAGE_CHECK_PASSED', 'agent_sha256':digest(data),
        'agent_bytes':len(data), 'rules_id':rules['rules_id'], 'rules_sha256':rules_hash(rules),
        'team_name_in_model':spec['team_name'], 'character':spec['character'],
        'algorithm':'PPO', 'image_shape':shape, 'input_keys':manifest['input_keys'],
        'action_count':len(spec['actions']), 'action_hash':action_hash,
        'runtime_versions_at_training':manifest.get('runtime_versions',{}),
        'student_code_executed':False, 'policy_loaded':False,
        'scope':'Container integrity, declared contracts and syntax only. Not runtime/skill/security certification.',
        'spec':spec, 'common':common, 'manifest':manifest}


def inspect_agent(path, rules):
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.stat().st_size > MAX_AGENT:
        raise ValueError('Agent package too large.')
    return inspect_agent_bytes(p.read_bytes(), rules)


def _public_check(info):
    return {k:v for k,v in info.items() if k not in ('spec','common','manifest')}


def select_agent(root, source='pve', checkpoint='best', bundle=''):
    if source == 'path':
        if not bundle:
            raise ValueError('For source="path", fill the exact saved agent.zip path.')
        return Path(bundle).expanduser().resolve(), {'source':'path','checkpoint':'explicit'}
    if source not in ('pve','pvp') or checkpoint not in ('best','last'):
        raise ValueError('source must be pve/pvp/path; checkpoint must be best/last.')
    if bundle:
        raise ValueError('Use source="path" for an explicit path, or leave bundle empty.')
    p = Path(root)/f'last_training_{source}.json'
    if not p.is_file():
        raise FileNotFoundError('No completed '+source+' run here. Choose source="path" for an existing package.')
    report = json_bytes(p.read_bytes()) if p.stat().st_size<=JSON_LIMIT else json.loads(p.read_text())
    selected = report.get(checkpoint+'_agent')
    if not selected:
        raise ValueError('No '+checkpoint+' package in the selected run. Choose another checkpoint explicitly; no silent fallback.')
    return Path(selected).resolve(), {'source':source,'checkpoint':checkpoint}


def prepare_submission(root, *, rules_path, team_id, display_name='', source='pve',
                       checkpoint='best', bundle='', notes=''):
    """One click, no model execution: package ONLY the selected saved snapshot."""
    if not isinstance(team_id,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}',team_id):
        raise ValueError('Use a short team ID such as Group_05 (letters, digits, _ or -).')
    if not isinstance(display_name,str) or len(display_name)>80 or not isinstance(notes,str) or len(notes)>2000:
        raise ValueError('Display name/notes too long.')
    rules = load_rules(rules_path)
    selected, selection = select_agent(root,source,checkpoint,bundle)
    if not selected.is_file():
        raise FileNotFoundError(selected)
    if selected.stat().st_size>MAX_AGENT:
        raise ValueError('Agent package too large.')
    data = selected.read_bytes()  # One immutable byte snapshot. No reading current student/ files.
    info = inspect_agent_bytes(data, rules)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    metadata = {'schema':SUBMISSION_SCHEMA, 'created_utc':stamp, 'team_id':team_id,
        'display_name':display_name or team_id, 'notes':notes,
        'agent_file':'agent.zip', 'agent_sha256':info['agent_sha256'], 'character':info['character'],
        'algorithm':'PPO', 'selection':selection, 'rules_id':rules['rules_id'],
        'rules_sha256':rules_hash(rules),
        'notebook_sources_used':'Exact selected agent snapshot; mutable current student files are NOT repackaged.',
        'required_runtime_versions':info['runtime_versions_at_training'],
        'runtime_self_check':'NOT_RUN; run this exact submission in the public Arena notebook.',
        'security':'Contains executable custom Python and serialized PyTorch/SB3 objects. Review before loading.'}
    folder = Path(root).resolve()/'submissions';folder.mkdir(parents=True,exist_ok=True)
    target = folder/f'{team_id}_{stamp}_submission.zip'
    readme = ('SYE 6308 practice submission\n'
        'This ZIP contains one selected agent, not both best and last.\n'
        'Use the public Arena notebook to inspect and test the exact ZIP in a clean session.\n'
        'Do not run the training notebook to make this ZIP work.\n'
        'Do not replace agent.zip after self-check; the receipt binds its exact checksum.\n'
        'Technical report/course deliverables are separate. No extra ROM, .state cache, credential store or videos are collected; review your selected source for secrets.\n'
        'Static packaging is not model execution, a sandbox, or an official competition score.\n')
    payload={'agent.zip':data,'submission.json':canonical(metadata),
             'static_check.json':canonical(_public_check(info)),'README.txt':readme.encode()}
    with tempfile.NamedTemporaryFile(dir=folder,suffix='.building',delete=False) as f:
        temp=Path(f.name)
    try:
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
            for name,content in payload.items():
                z.writestr(name,content)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    sha=digest(target.read_bytes())
    target.with_suffix('.sha256').write_text(f'{sha}  {target.name}\n',encoding='utf-8')
    print('SUBMISSION_PACKAGED (static checks only; arena self-check still required)')
    print('Selected model:',selected)
    print('Team:',team_id,'| character:',info['character'],'| actions:',info['action_count'],'| input:',info['image_shape'])
    print('Submission:',target);print('Submission SHA256:',sha)
    return target


def inspect_submission(path, rules, extract_agent_to=None):
    p=Path(path).expanduser().resolve()
    if not p.is_file():raise FileNotFoundError(p)
    if p.stat().st_size>MAX_OUTER:raise ValueError('Submission ZIP too large.')
    data=p.read_bytes()
    f=_read_archive(data,SUBMISSION_FILES,MAX_OUTER,binary_names={'agent.zip'})
    m=json_bytes(f['submission.json'])
    if (m.get('schema')!=SUBMISSION_SCHEMA or m.get('agent_file')!='agent.zip'
            or m.get('rules_id')!=rules['rules_id'] or m.get('rules_sha256')!=rules_hash(rules)):
        raise ValueError('Submission schema/rules differ from the public arena release.')
    if not isinstance(m.get('team_id'),str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}',m['team_id']):
        raise ValueError('Invalid team ID.')
    info=inspect_agent_bytes(f['agent.zip'],rules)
    if m.get('agent_sha256')!=info['agent_sha256'] or m.get('character')!=info['character']:
        raise ValueError('Selected model does not match submission metadata.')
    if json_bytes(f['static_check.json'])!=_public_check(info):
        raise ValueError('Stored static check does not match the selected model; rebuild submission.')
    result={'status':'STATIC_SUBMISSION_CHECK_PASSED','submission_sha256':digest(data),
            'submission_bytes':len(data),'metadata':m,**info}
    result['status']='STATIC_SUBMISSION_CHECK_PASSED'
    if extract_agent_to is not None:
        target=Path(extract_agent_to);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(f['agent.zip'])
        result['agent_path']=str(target.resolve())
    return result


def candidate_models(root, limit=40):
    root=Path(root)
    candidates=[]
    for p in sorted(root.glob('runs/*/*/*_agent.zip'),key=lambda p:p.stat().st_mtime,reverse=True):
        if p.name in ('best_agent.zip','last_agent.zip'):
            candidates.append(p.resolve())
    return candidates[:limit]


def submission_widget(root, rules_path, on_packaged=None):
    """Optional one-click UI; the plain configuration cell is also provided."""
    import ipywidgets as w
    from IPython.display import display,FileLink
    if on_packaged is not None and not callable(on_packaged):
        raise TypeError("on_packaged must be callable or None.")
    candidates=candidate_models(root)
    labels=[(str(p.relative_to(Path(root).resolve())),str(p)) for p in candidates]
    select=w.Dropdown(options=[('Choose a saved model...', '')]+labels,description='Model:',layout=w.Layout(width='95%'))
    explicit=w.Text(value='',description='Or path:',placeholder='Exact v3 agent ZIP path',layout=w.Layout(width='95%'))
    team=w.Text(value='Group_01',description='Team ID:')
    name=w.Text(value='',description='Team name:')
    button=w.Button(description='Package selected model',button_style='primary',layout=w.Layout(width='220px'))
    output=w.Output()
    def on_click(_):
        output.clear_output();button.disabled=True
        with output:
            try:
                chosen=explicit.value.strip() or select.value
                if not chosen:raise ValueError('Select a model or enter its exact saved package path.')
                p=prepare_submission(root,rules_path=rules_path,team_id=team.value.strip(),
                    display_name=name.value.strip(),source='path',bundle=chosen)
                if on_packaged is not None:
                    on_packaged(p)
                display(FileLink(str(p)));display(FileLink(str(p.with_suffix('.sha256'))))
            except Exception as exc:
                print(type(exc).__name__+': '+str(exc))
            finally:button.disabled=False
    button.on_click(on_click)
    display(w.VBox([select,explicit,team,name,button,output]))
