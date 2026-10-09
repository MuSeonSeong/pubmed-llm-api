"""Prepare the pinned weights of the models this server answers with; never depend on external cache symlinks.

By default these are the enabled models (MODELS, or the MLX models on Apple Silicon and the Transformers
models elsewhere); --model picks others, --all takes every model in models.json.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors import safe_open
from app.catalog import ModelSpec, load_catalog, model_weight_files
from app.settings import Settings, enabled_models

PATTERNS = ['*.json', '*.safetensors', '*.jinja', 'merges.txt', 'vocab.txt', 'tokenizer.model', 'README.md', 'LICENSE*']


def validate_files(path: Path) -> None:
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json']:
        if not (path / name).is_file():
            raise ValueError(f'Missing {name} in {path}')
    weights = model_weight_files(path)
    for file in weights:
        with safe_open(file, framework='numpy') as data:
            if not list(data.keys()):
                raise ValueError(f'Empty weights: {file}')


def prepare_model(spec: ModelSpec, destination: Path) -> Path:
    marker = destination / '.model-ready.json'
    if marker.exists():
        try:
            info = json.loads(marker.read_text())
            if info.get('revision') == spec.revision and info.get('repo_id') == spec.repo_id:
                if info.get('files') and all((destination / name).is_file() and (destination / name).stat().st_size == size
                       for name, size in info['files'].items()):
                    validate_files(destination)
                    return destination
        except (ValueError, OSError, TypeError, AttributeError):
            pass
        marker.unlink(missing_ok=True)
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
    pending = marker.with_name(marker.name + '.partial')
    pending.write_text(json.dumps({'repo_id': spec.repo_id, 'revision': spec.revision, 'files': files}, indent=2))
    pending.replace(marker)
    return destination


def chosen_models(settings: Settings, catalog: dict[str, ModelSpec], models: list[str] | None,
                  everything: bool) -> list[str]:
    if everything:
        return list(catalog)
    return list(dict.fromkeys(models)) if models else enabled_models(settings, catalog)


def main(argv: list[str] | None = None) -> None:
    catalog = load_catalog()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', action='append', choices=list(catalog), help='this model only (repeatable)')
    parser.add_argument('--all', action='store_true', help='every model in models.json')
    args = parser.parse_args(argv)
    settings = Settings()
    for model_id in chosen_models(settings, catalog, args.model, args.all):
        spec = catalog[model_id]
        print(f'Preparing {spec.id} ({spec.revision})', flush=True)
        print(prepare_model(spec, settings.models_dir / spec.directory), flush=True)

if __name__ == '__main__':
    main()
