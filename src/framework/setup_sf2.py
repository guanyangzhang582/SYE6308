"""Install only missing dependencies and locate/import the original course ROM.
No external state download. Match-start snapshots are generated at runtime.
ROM is not included in this package.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
import subprocess
import sys
import urllib.request

ROM_SHA1='a5aad1d108046d9388e33247610dafb4c6516e0b'
GAME='StreetFighterIISpecialChampionEdition-Genesis'
# Same external ROM URL that was already present in the user's original v8.0.
LEGACY_URL='https://archive.org/download/street-fighter-ii-special-champion-edition-usa/Street%20Fighter%20II%27%20-%20Special%20Champion%20Edition%20%28USA%29.md'


def retro_requirement_for_python(version):
    """Select a published wheel recipe; never alter the notebook interpreter.

    Preserve the original 0.9.5 recipe on older runtimes. Version 1.0.1
    publishes CPython 3.13/3.14 wheels; 0.9.5 explicitly excludes Python 3.13.
    This selects an installer, not a certificate of game/model compatibility.
    """
    version = tuple(version[:2])
    if (3, 8) <= version < (3, 13):
        return 'stable-retro==0.9.5'
    if (3, 13) <= version < (3, 15):
        return 'stable-retro==1.0.1'
    raise RuntimeError(
        'No installation recipe is defined for Python ' + '.'.join(map(str, version))
        + '. Keep the full version report; do not remove this check or replace system Python.'
    )


def setup(directory, rom_file='', allow_legacy_fetch=False, install_sb3=False):
    folder=Path(directory).resolve();folder.mkdir(parents=True,exist_ok=True)
    print('Setup Python:', sys.version, '| executable:', sys.executable, flush=True)
    if not (importlib.util.find_spec('retro') or importlib.util.find_spec('stable_retro')):
        requirement = retro_requirement_for_python(sys.version_info)
        print('Installing emulator dependency:', requirement, flush=True)
        # Require a published emulator wheel instead of silently building C++ from source.
        # Use the SAME interpreter that runs the notebook and its worker processes.
        subprocess.run([sys.executable, '-m', 'pip', 'install',
                        '--only-binary=stable-retro', requirement], check=True)
        importlib.invalidate_caches()
    for module,package in [('cv2','opencv-python-headless'),('PIL','pillow')]:
        if importlib.util.find_spec(module) is None:
            subprocess.run([sys.executable,'-m','pip','install',package],check=True)
    if install_sb3 and importlib.util.find_spec('stable_baselines3') is None:
        import gymnasium
        from packaging.version import Version
        version=Version(gymnasium.__version__)
        if version < Version('0.29'):
            raise RuntimeError('Gymnasium is older than 0.29; inspect compatibility before installing SB3.')
        package='stable-baselines3==2.3.2' if version < Version('1.0') else 'stable-baselines3==2.7.0'
        subprocess.run([sys.executable,'-m','pip','install',package],check=True)
    try:
        import stable_retro as backend
        module_name='stable_retro'
    except ModuleNotFoundError as exc:
        if exc.name!='stable_retro':raise
        import retro as backend
        module_name='retro'
    print('Emulator import:', module_name, '| module:', getattr(backend, '__file__', 'unknown'), flush=True)
    try:
        print('stable-retro version:', importlib.metadata.version('stable-retro'), flush=True)
    except importlib.metadata.PackageNotFoundError:
        print('WARNING: stable-retro distribution metadata was not found. '
              'Review the backend before starting any training or arena test.', flush=True)
    def installed():
        # The course identity stays unchanged; only the backend integration ID
        # differs. Check these two known aliases, never an arbitrary game/version.
        for game_id in (GAME, GAME+'-v0'):
            for method in ('get_original_romfile_path','get_romfile_path'):
                fn=getattr(backend.data,method,None)
                if fn:
                    try:
                        p=Path(fn(game_id))
                        if p.is_file() and hashlib.sha1(p.read_bytes()).hexdigest()==ROM_SHA1:
                            print('Resolved game ID:', game_id, flush=True)
                            return p
                    except (OSError,RuntimeError,ValueError,TypeError):pass
        return None
    path=installed()
    if path is None:
        if rom_file:
            candidate=Path(rom_file).expanduser().resolve()
            if not candidate.is_file():raise FileNotFoundError(candidate)
        else:
            candidate=None
            for p in Path('/kaggle/input').rglob('*'):
                if p.is_file() and p.suffix.lower() in ('.md','.bin','.gen') and p.stat().st_size<8*1024**2:
                    if hashlib.sha1(p.read_bytes()).hexdigest()==ROM_SHA1:
                        candidate=p;break
            if candidate is None and allow_legacy_fetch:
                target=folder/'StreetFighterIISpecialChampionEdition-Genesis.md'
                print('Trying the original notebook ROM source; no state download.',flush=True)
                try:
                    req=urllib.request.Request(LEGACY_URL,headers={'User-Agent':'Mozilla/5.0'})
                    with urllib.request.urlopen(req,timeout=90) as response:
                        data=response.read(8*1024**2+1)
                    if len(data)>8*1024**2 or hashlib.sha1(data).hexdigest()!=ROM_SHA1:
                        raise RuntimeError('Downloaded bytes are not the expected ROM; file not imported.')
                    target.write_bytes(data);candidate=target
                except Exception as exc:
                    raise RuntimeError('Original ROM source unavailable. Attach your matching game ROM and set ROM_FILE. '
                                       'A .state file is not required and cannot substitute for the ROM.') from exc
            if candidate is None:
                raise FileNotFoundError('Provide ROM_FILE or attach your matching game ROM; no .state is needed.')
        if hashlib.sha1(candidate.read_bytes()).hexdigest()!=ROM_SHA1:
            raise RuntimeError('Provided ROM does not match the required SHA1.')
        # Dedicated folder avoids importing every file from the user's input directory.
        import shutil
        import_dir=folder/'rom_import';import_dir.mkdir(exist_ok=True)
        local=import_dir/'StreetFighterIISpecialChampionEdition-Genesis.md'
        if candidate.resolve()!=local.resolve():shutil.copy2(candidate,local)
        subprocess.run([sys.executable,'-m',module_name+'.import',str(import_dir)],check=True)
        path=installed()
    if path is None:raise RuntimeError('ROM import finished, but neither '+GAME+' nor '+GAME+'-v0 resolves to the required SHA1. Keep the full setup log.')
    print('ROM verified:',path,flush=True)
    print('SHA1:',ROM_SHA1,flush=True)
    print('Resources ready. Match-start caches will be generated separately; no PPO training yet.',flush=True)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True)
    p.add_argument('--rom-file',default='');p.add_argument('--legacy-fetch',action='store_true')
    p.add_argument('--sb3',action='store_true');a=p.parse_args()
    setup(a.directory,a.rom_file,a.legacy_fetch,a.sb3)
