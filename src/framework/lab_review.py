"""Read-only run review and handoff files. Does not load or train a policy."""
from pathlib import Path
import json
import zipfile


def show_training_review(run_directory):
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    run = Path(run_directory)
    if not run.is_dir():
        raise FileNotFoundError(run)
    validation = run/'validation_history.jsonl'
    if validation.is_file():
        records = [json.loads(s) for s in validation.read_text().splitlines() if s.strip()]
    else:
        records = []
        for path in sorted(run.glob('validation_*/report.json')):
            data = json.loads(path.read_text())
            step = path.parent.name.rsplit('_', 1)[-1]
            if step.isdigit():
                records.append({'steps': int(step), 'win_fraction': data['win_fraction'],
                    'mean_hp_margin': data['mean_hp_margin'], 'report': str(path)})
    if records:
        table = pd.DataFrame(records).sort_values('steps').drop_duplicates('steps', keep='last')
        display(table[[c for c in ('steps','target','learner_sides','win_fraction','mean_hp_margin') if c in table]])
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(table['steps'], table['win_fraction'], marker='o')
        ax.set(xlabel='Total transitions in this phase', ylabel='Validation win fraction', ylim=(-0.03, 1.03))
        target=str(table['target'].iloc[-1]) if 'target' in table else 'recorded validation'
        ax.set_title('Validation target: '+target)
        fig.tight_layout(); fig.savefig(run/'validation_curve.png', dpi=150); plt.show()
    path = run/'logs/progress.csv'
    if path.is_file():
        logs = pd.read_csv(path)
        if {'time/total_timesteps', 'rollout/ep_rew_mean'} <= set(logs.columns):
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(logs['time/total_timesteps'], logs['rollout/ep_rew_mean'])
            ax.set(xlabel='Total transitions in this phase', ylabel='Mean training episode return')
            ax.set_title('Training return is not a validation win rate')
            fig.tight_layout(); fig.savefig(run/'reward_curve.png', dpi=150); plt.show()
    selection = run/'best_selection.json'
    if selection.is_file():
        print('Best checkpoint selection:', json.loads(selection.read_text()))
    path = run/'episodes.jsonl'
    if path.is_file():
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()][-100:]
        groups = []
        for result in ('win', 'loss', 'draw', 'cutoff'):
            values = [float(r['episode']['r']) for r in rows if r['result'] == result]
            if values:
                groups.append({'result': result, 'episodes': len(values),
                    'mean_episode_return': sum(values)/len(values),
                    'positive_return_episodes': sum(v>0 for v in values)})
        if groups:
            print('Most recent completed training episodes (not the validation games):')
            display(pd.DataFrame(groups))
        ports=[]
        for player in (1,2):
            subset=[r for r in rows if r.get('self_player')==player]
            if subset:ports.append({'learner_player':player,'episodes':len(subset),
                'win_fraction':sum(r['result']=='win' for r in subset)/len(subset)})
        if ports:display(pd.DataFrame(ports))


def make_handoff(run_directory, destination=None):
    """Archive model packages and reports, not ROMs, caches or emulator states."""
    run = Path(run_directory).resolve()
    if not (run/'report.json').is_file():
        raise FileNotFoundError('A completed run report is required.')
    dest = Path(destination) if destination else run.parent/(run.name+'_handoff.zip')
    paths = []
    for name in ('best_agent.zip', 'last_agent.zip', 'initial_agent.zip', 'report.json',
                 'job.json', 'best_selection.json', 'episodes.jsonl', 'console.log',
                 'validation_history.jsonl', 'performance_rollouts.json',
                 'validation_curve.png', 'reward_curve.png'):
        p = run/name
        if p.is_file(): paths.append(p)
    for folder in ('sources', 'framework_sources', 'logs'):
        d = run/folder
        if d.is_dir(): paths.extend(p for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(run.glob('validation_*/report.json'))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(paths)):
            z.write(p, p.relative_to(run))
        z.writestr('HANDOFF_README.txt',
            'Agent files are trusted-Python packages, not raw .pth weights.\n'
            'Use best_agent.zip for the validation-selected model; last_agent.zip is the final checkpoint.\n'
            'A completed run does not certify a winning model. Check the recorded evaluation results.\n'
            'No ROM, match-start state, game cache, credentials or pretrained external weights are included.\n')
    print('Handoff archive:', dest)
    return dest
