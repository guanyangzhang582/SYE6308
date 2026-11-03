"""Shared launcher used by training-notebook self-check and standalone arena."""
from pathlib import Path
import datetime as dt
import json
import os
import signal
import shutil
import subprocess
import sys

from sf2_submission import inspect_submission,load_rules,digest,rules_hash


def inspect_for_display(path,rules_path):
    info=inspect_submission(path,load_rules(rules_path))
    print('STATIC_SUBMISSION_CHECK_PASSED (no model code executed)')
    print('Team:',info['metadata']['team_id'],'| Character:',info['character'])
    print('Input:',info['image_shape'],'| Actions:',info['action_count'])
    print('Submission SHA256:',info['submission_sha256'])
    print('Agent SHA256:',info['agent_sha256'])
    print('Training versions:',info['runtime_versions_at_training'])
    return {k:v for k,v in info.items() if k not in ('spec','manifest')}


def run_arena(root,*,submission_a,submission_b='',mode='self_check',trust_submissions=False,
              rules_path=None,offsets=None,record_video=True):
    if mode not in ('self_check','match'):raise ValueError('mode must be self_check or match.')
    if trust_submissions is not True:
        raise ValueError('Review all packages, then explicitly set TRUST_SUBMISSIONS=True. Checksums/processes are not a sandbox.')
    root=Path(root).resolve();code=root/'code'
    rules_path=Path(rules_path or root/'arena_rules_v1.json').resolve();rule=load_rules(rules_path)
    if mode=='self_check':
        if submission_b and Path(submission_b).resolve()!=Path(submission_a).resolve():
            raise ValueError('Self-check uses A twice. Leave submission_b empty, or choose mode="match".')
        submission_b=submission_a
        if offsets is not None and list(offsets)!=rule['selfcheck_offsets']:
            raise ValueError('Self-check offsets must match the published practice rules.')
        offsets=rule['selfcheck_offsets']
    elif not submission_b:raise ValueError('Fill submission_b for a two-team match.')
    if offsets is None:offsets=rule['selfcheck_offsets']
    # Fail before creating children if either container is invalid.
    checks=[inspect_submission(p,rule) for p in (submission_a,submission_b)]
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    out=root/'arena_runs'/f'{stamp}_{mode}';out.mkdir(parents=True)
    inputs=out/'submission_inputs';inputs.mkdir()
    paths=[]
    for name,p,info in zip('AB',(submission_a,submission_b),checks):
        target=inputs/f'{name}_submission.zip';shutil.copy2(p,target)
        if digest(target.read_bytes())!=info['submission_sha256']:
            raise RuntimeError('Submission changed while starting the check; retry with a fixed saved ZIP.')
        paths.append(str(target))
    snapshot=out/'rules.json';snapshot.write_text(json.dumps(rule,indent=2))
    job={'mode':mode,'out':str(out),'submission_a':paths[0],'submission_b':paths[1],
         'rules_path':str(snapshot),'offsets':list(offsets),'swap_sides':True,'seed':42,
         'record_video':bool(record_video),'trust_submissions':True}
    (out/'job.json').write_text(json.dumps(job,indent=2))
    env=os.environ.copy();env.update(SF2_START_CACHE_DIR=str(root/'cache'/'match_starts_v2'),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    print('Arena:',mode,'| output:',out,flush=True)
    with (out/'console.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(code/'sf2_arena.py'),'--job',str(out/'job.json')],
            cwd=code,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        try:
            for line in proc.stdout:print(line,end='');log.write(line);log.flush()
            proc.wait()
        except KeyboardInterrupt:
            # Actor processes have their own cleanup in the arena driver's finally block.
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    path=out/'report.json'
    if not path.is_file():
        report={'status':'ARENA_CHECK_FAILED','error':'Driver exited without report. See console.log.'}
        path.write_text(json.dumps(report,indent=2))
    else:report=json.loads(path.read_text())
    # Repack AFTER console.log is closed so diagnostics include the complete tail.
    from sf2_arena import make_diagnostic_zip
    make_diagnostic_zip(out)
    if proc.returncode or report.get('status')=='ARENA_CHECK_FAILED':
        print('FAILED: keep arena_diagnostics.zip with the unchanged submission ZIP.')
    return out,report


def show_arena_result(out,report):
    from IPython.display import display,FileLink,Video,Image
    out=Path(out);print('STATUS:',report['status'])
    if report.get('summary'):print('Round results:',report['summary'])
    if report.get('latency'):print('Latency:',json.dumps(report['latency'],indent=2))
    if report.get('error'):print(report['error'])
    for p in report.get('videos',[]):
        if p and Path(p).is_file():display(Video(p,embed=True,width=800))
    for name in ('report.json','receipt.json','arena_diagnostics.zip'):
        if (out/name).is_file():display(FileLink(str(out/name)))
    print('No reward-based scoring, no training, no security certification. Self-check is not proof of skill.')
