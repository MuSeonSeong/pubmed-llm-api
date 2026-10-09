"""Prepare pinned original weights; never depend on external cache symlinks."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors import safe_open
from app.catalog import ModelSpec, load_catalog
from app.settings import Settings

PATTERNS = ['*.json', '*.safetensors', '*.jinja', 'merges.txt', 'vocab.txt', 'tokenizer.model', 'README.md', 'LICENSE*']


def validate_files(path: Path) -> None:
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json']:
        if not (path / name).is_file():
            raise ValueError(f'Missing {name} in {path}')
    index = path / 'model.safetensors.index.json'
    weights = sorted(path.glob('*.safetensors'))
    if index.exists():
        names = set(json.loads(index.read_text())['weight_map'].values())
        if not all((path / name).is_file() for name in names):
            raise ValueError(f'Missing weight shards in {path}')
    if not weights:
        raise ValueError(f'Missing model weights in {path}')
    for file in weights:
        with safe_open(file, framework='numpy') as data:
            if not list(data.keys()):
                raise ValueError(f'Empty weights: {file}')


def prepare_model(spec: ModelSpec, destination: Path) -> Path:
    marker = destination / '.model-ready.json'
    if marker.exists():
        info = json.loads(marker.read_text())
        if info.get('revision') == spec.revision and info.get('repo_id') == spec.repo_id:
            if all((destination / name).is_file() and (destination / name).stat().st_size == size
                   for name, size in info.get('files', {}).items()):
                validate_files(destination)
                return destination
    options = dict(repo_id=spec.repo_id, revision=spec.revision, allow_patterns=PATTERNS)
    try:
        source = Path(snapshot_download(**options, local_files_only=True))
        validate_files(source)
    except (LocalEntryNotFoundError, ValueError):
        source = Path(snapshot_download(**options))
        validate_files(source)
    destination.mkdir(parents=True, exist_ok=True)
    for file in source.rglob('*'):
        if file.is_file():
            target = destination / file.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + '.partial')
            shutil.copyfile(file, temp)
            temp.replace(target)
    validate_files(destination)
    files = {str(file.relative_to(destination)): file.stat().st_size for file in destination.rglob('*')
             if file.is_file() and file != marker and not file.name.endswith('.partial')}
    marker.write_text(json.dumps({'repo_id': spec.repo_id, 'revision': spec.revision, 'files': files}, indent=2))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(load_catalog()))
    args = parser.parse_args()
    settings = Settings()
    for spec in load_catalog().values():
        if args.model is None or args.model == spec.id:
            print(f'Preparing {spec.id} ({spec.revision})', flush=True)
            print(prepare_model(spec, settings.models_dir / spec.directory), flush=True)

if __name__ == '__main__':
    main()
