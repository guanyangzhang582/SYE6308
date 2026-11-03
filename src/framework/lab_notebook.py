"""Notebook task launcher with separate PvE/PvP student sources and check gates.

Each run snapshots only the selected profile's four student files. Game-engine
code is shared. No task reads the other profile's mutable files or parameters.
External packages are copied once into the run, preventing mid-run replacement.
"""
from pathlib import Path
import copy, datetime, hashlib, json, os, shutil, signal, subprocess, sys

STUDENT_FILES = ('student_observation.py', 'student_actions.py',
                 'student_reward.py', 'student_model.py')


def profile_directory(root, profile):
    if profile not in ('pve', 'pvp'):
        raise ValueError('profile must be pve or pvp')
    return Path(root).resolve()/'student'/profile


def last_training(root, kind='pve'):
    if kind == 'frozen':
        kind = 'pvp'  # compatibility name, not a shared latest-run pointer
    if kind not in ('pve', 'pvp'):
        raise ValueError('Choose last_training(..., kind="pve" or "pvp").')
    p = Path(root)/('last_training_'+kind+'.json')
    if not p.is_file():
        raise FileNotFoundError(f'No completed {kind.upper()} training in this project. Supply an existing v3 package path explicitly.')
    return json.loads(p.read_text(encoding='utf-8'))


def input_check_fingerprint(root, agent, common, pve=None, profile='pve'):
    root = Path(root).resolve()
    source = profile_directory(root, profile)
    missing = [str(source/n) for n in STUDENT_FILES if not (source/n).is_file()]
    if missing:
        raise FileNotFoundError('Run the four student editing cells for '+profile+': '+', '.join(missing))
    return {
        'profile': profile,
        'source_hashes': {n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in STUDENT_FILES},
        'framework_hashes': {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((root/'code').glob('*.py'))},
        'agent': copy.deepcopy(agent), 'common':copy.deepcopy(common),
        # There is deliberately no dependency on CPU opponent/difficulty for PvP.
        'pve':copy.deepcopy(pve or {}) if profile == 'pve' else None,
    }


def assert_no_running_task(root):
    script = str(Path(root).resolve()/'code'/'run_lab.py').encode()
    procdir = Path('/proc')
    if not procdir.is_dir():
        return
    for p in procdir.iterdir():
        if p.name.isdigit() and int(p.name) != os.getpid():
            try:
                args=(p/'cmdline').read_bytes().split(b'\0')
            except (OSError, PermissionError):
                continue
            if script in args:
                raise RuntimeError('A task from this project is still running. Finish/stop it first; do not overwrite its sources.')


def _profile_for_task(kind):
    if kind in ('check_frozen', 'train_frozen', 'evaluate_frozen', 'pvp'):
        return 'pvp'
    if kind in ('check', 'actions', 'train', 'evaluate'):
        return 'pve'
    raise ValueError('Unknown task: '+str(kind))


def _marker(root, profile):
    return Path(root)/'checks'/(profile+'_check.json')


def _snapshot_packages(job, out):
    """Copy only package bytes, never execute a model here. Preserve exact identity."""
    refs=[]
    if job.get('initialization_bundle'):
        refs.append((job, 'initialization_bundle', 'initial_learner'))
    for section, keys in (('frozen', ('bundle',)),
                          ('evaluation', ('bundle','initial_bundle')),
                          ('pvp', ('agent_a','agent_b'))):
        for key in keys:
            if job.get(section, {}).get(key):
                refs.append((job[section],key,section+'_'+key))
    if refs and job.get('trust_packages') is not True:
        raise ValueError('Review these Python/model packages, then explicitly enable the trust switch.')
    records=[];copies={}
    folder=Path(out)/'package_inputs'
    for obj, key, label in refs:
        original=Path(obj[key]).expanduser().resolve()
        if not original.is_file():
            raise FileNotFoundError('Package does not exist: '+str(original))
        if original.stat().st_size > 512*1024**2:
            raise ValueError('Package exceeds the supported 512 MiB limit.')
        if str(original) not in copies:
            folder.mkdir(parents=True,exist_ok=True)
            target=folder/(label+'.zip')
            shutil.copy2(original,target)
            digest=hashlib.sha256(target.read_bytes()).hexdigest()
            copies[str(original)]=(str(target),digest)
        target,digest=copies[str(original)]
        records.append({'role':label,'original_path':str(original),'snapshot':target,'sha256':digest})
        obj[key]=target
    job['package_inputs']=records
    return records


def run_task(root, kind, agent, common, train, pve=None, **extra):
    root=Path(root).resolve();code=root/'code'
    assert_no_running_task(root)
    profile=_profile_for_task(kind)
    source=profile_directory(root,profile)
    fingerprint=None
    if kind in ('check','train'):
        fingerprint=input_check_fingerprint(root,agent,common,pve,profile='pve')
    elif kind in ('check_frozen','train_frozen'):
        from sf2_release import frozen_check_fingerprint
        fingerprint=frozen_check_fingerprint(root,agent,common,{},extra['frozen'],
            initialization_bundle=extra.get('initialization_bundle',''),train=train)
    if kind in ('train','train_frozen'):
        marker=_marker(root,profile)
        if not marker.is_file() or json.loads(marker.read_text()).get('fingerprint') != fingerprint:
            required='4 (PvE check)' if profile=='pve' else '9.3 (PvP check)'
            raise RuntimeError('Run Section '+required+' with the current '+profile.upper()+' files, packages and settings before training.')
    # Do not require mutable student files for a saved-package-only match/evaluation.
    uses_student=kind in ('check','actions','train','check_frozen','train_frozen')
    if uses_student:
        for name in STUDENT_FILES:
            p=source/name
            if not p.is_file():raise FileNotFoundError('Run the editing cell: '+str(p))
            compile(p.read_text(encoding='utf-8'),str(p),'exec')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    out=root/'runs'/profile/f'{stamp}_{kind}';out.mkdir(parents=True)
    sources=out/'sources';sources.mkdir()
    if uses_student:
        for name in STUDENT_FILES:shutil.copy2(source/name,sources/name)
    framework=out/'framework_sources';framework.mkdir()
    for p in code.glob('*.py'):shutil.copy2(p,framework/p.name)
    job={'kind':kind,'profile':profile,'out':str(out),'sources':str(sources),
         'agent':copy.deepcopy(agent),'common':copy.deepcopy(common),'train':copy.deepcopy(train),
         'pve':copy.deepcopy(pve or {}) if profile=='pve' else {}, **copy.deepcopy(extra)}
    if kind in ('check_frozen','train_frozen','evaluate_frozen'):
        from sf2_release import validate_frozen_config
        job['frozen']=validate_frozen_config(job['frozen'])
    package_inputs=_snapshot_packages(job,out)
    # Freeze known package bytes between the successful check and a training launch.
    if fingerprint is not None and profile=='pvp':
        expected={'frozen_bundle':fingerprint['opponent_sha256'],
                  'initial_learner':fingerprint.get('initialization_sha256')}
        for item in package_inputs:
            if item['role'] in expected and expected[item['role']] != item['sha256']:
                raise RuntimeError('A selected package changed while starting the task. Repeat the check.')
    (out/'job.json').write_text(json.dumps(job,indent=2),encoding='utf-8')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        SF2_START_CACHE_DIR=str(root/'cache'/'match_starts_v2'))
    print('Task:',kind,'| profile:',profile,'| output:',out,flush=True)
    with (out/'console.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(code/'run_lab.py'),'--job',str(out/'job.json')],
            cwd=code,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
            bufsize=1,start_new_session=True)
        try:
            for line in proc.stdout:print(line,end='');log.write(line);log.flush()
            proc.wait()
        except KeyboardInterrupt:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            print('Interrupted. Existing packages were not overwritten. Results:',out);raise
    p=out/'report.json'
    if not p.is_file():raise RuntimeError('No report. Inspect console.log in '+str(out))
    report=json.loads(p.read_text(encoding='utf-8'))
    if proc.returncode or report.get('status')=='ERROR':
        raise RuntimeError(report.get('error','Task failed')+' | '+str(out))
    print('STATUS:',report['status'])
    if kind in ('check','check_frozen'):
        expected='ENVIRONMENT_CHECK_PASSED' if kind=='check' else 'FROZEN_ENVIRONMENT_CHECK_PASSED'
        if report.get('status')==expected:
            marker=_marker(root,profile);marker.parent.mkdir(exist_ok=True)
            marker.write_text(json.dumps({'fingerprint':fingerprint,'directory':str(out)},indent=2),encoding='utf-8')
    if kind in ('train','train_frozen'):
        saved=json.dumps({'directory':str(out),**report},indent=2)
        (root/('last_training_'+profile+'.json')).write_text(saved,encoding='utf-8')
        print('BEST:',report.get('best_agent'));print('LAST:',report['last_agent'])
    return out,report


def run_pvp_task(root, kind, agent, common, train, opponent, **extra):
    """PvP does not accept or read PVE_CONFIG/PVE_TRAIN_CONFIG."""
    kinds={'check':'check_frozen','train':'train_frozen','evaluate':'evaluate_frozen'}
    if kind not in kinds:raise ValueError('PvP kind must be check, train or evaluate')
    return run_task(root,kinds[kind],agent,common,train,None,frozen=opponent,**extra)


def show_result(out,report=None):
    from IPython.display import display,Image,Video,FileLink
    out=Path(out);report=report or json.loads((out/'report.json').read_text())
    folders=[out]+sorted(out.glob('learner_P*'))
    for folder in folders:
        for name in ('actual_game.png','actual_model_images.png','motion_game.png','motion_model_images.png'):
            p=folder/name
            if p.is_file():
                print(folder.name,':','RESET input' if name.startswith('actual_') else 'STEPPED history')
                display(Image(filename=str(p)))
        for name in ('actual_input.json','motion_input.json'):
            p=folder/name
            if p.is_file():
                d=json.loads(p.read_text());print(name,': shape=',d['image_shape'],
                    'learner_port=',d['player'],'frames=',d['frame_ids'])
    videos=list(report.get('videos',[]))
    if report.get('video'):videos=[report['video']]+videos
    for video in videos:
        if video and Path(video).is_file():display(Video(video,embed=True,width=800))
    if report.get('summary'):print(json.dumps(report['summary'],indent=2))
    if report.get('validation_target'):print('Validation target:',report['validation_target'])
    print('Full results:',out);display(FileLink(str(out/'report.json')))
