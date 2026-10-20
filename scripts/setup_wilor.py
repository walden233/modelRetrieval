"""Prepare isolated WiLoR inference code, dependencies and official demo assets."""

import argparse
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'fcb911312a38fa8badd30d9656a167485d61b8f9'
ASSETS = {
    'pretrained_models/wilor_final.ckpt': ('wilor_final.ckpt', 2564989533),
    'pretrained_models/detector.pt': ('detector.pt', 53582271),
    'pretrained_models/model_config.yaml': ('model_config.yaml', 2233),
    'mano_data/mano/MANO_RIGHT.pkl': ('mano/MANO_RIGHT.pkl', 3821356),
    'mano_data/mano_mean_params.npz': ('mano_mean_params.npz', 1178),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-only', action='store_true')
    args = parser.parse_args()
    repo, environment = ROOT/'artifacts/tools/WiLoR', ROOT/'artifacts/tools/wilor-env'
    if not args.assets_only:
        if not repo.exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--no-checkout',
                            'https://github.com/rolpotamias/WiLoR.git', str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', REVISION], check=True)
        current = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if current != REVISION:
            raise ValueError(f'WiLoR revision differs: {current}; expected {REVISION}')
        python = environment/'bin/python'
        if not python.exists():
            subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(environment)], check=True)
            subprocess.run([str(python), '-m', 'pip', 'install', '--no-build-isolation',
                            '-r', str(ROOT/'configs/hand/requirements.txt')], check=True)
        print('WiLoR environment:', python, flush=True)
    base = 'https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/'
    for source, (relative, expected_size) in ASSETS.items():
        target = ROOT/'artifacts/models/hand/wilor'/relative
        if target.exists() and target.stat().st_size == expected_size:
            print('EXISTS', relative, flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix+'.part')
        with requests.get(base+source+'?download=true', stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            downloaded, next_report = 0, 128*1024**2
            with temporary.open('wb') as file:
                for chunk in response.iter_content(4*1024**2):
                    file.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        print(relative, f'{downloaded/expected_size:.0%}', flush=True)
                        next_report += 128*1024**2
        if temporary.stat().st_size != expected_size:
            raise ValueError(f'Incomplete asset: {temporary}')
        temporary.replace(target)
        print('DOWNLOADED', relative, flush=True)


if __name__ == '__main__':
    main()
