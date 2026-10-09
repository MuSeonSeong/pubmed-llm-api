import json
from pathlib import Path
import pytest


def snapshot(tmp_path):
    from safetensors.numpy import save_file
    import numpy as np
    src = tmp_path / "cache source"
    src.mkdir()
    (src / "config.json").write_text('{}')
    (src / "tokenizer_config.json").write_text('{}')
    (src / "tokenizer.json").write_text('{}')
    save_file({"weight": np.zeros((2, 2), dtype=np.float32)}, src / "model.safetensors")
    return src


def test_cache_symlinks_become_independent_files(tmp_path, monkeypatch):
    from app.catalog import load_catalog
    from scripts import prepare_models as mod
    src = snapshot(tmp_path)
    blob = tmp_path / "blob"
    (src / "model.safetensors").rename(blob)
    (src / "model.safetensors").symlink_to(blob)
    monkeypatch.setattr(mod, "snapshot_download", lambda **kw: str(src))
    dst = tmp_path / "new project" / "model"
    mod.prepare_model(load_catalog()["qwen3-0.6b"], dst)
    blob.unlink()
    assert (dst / "model.safetensors").stat().st_size > 0
    assert not any(p.is_symlink() for p in dst.rglob('*'))
    assert json.loads((dst / '.model-ready.json').read_text())['revision'] == 'c1899de289a04d12100db370d81485cdf75e47ca'


def test_partial_download_is_not_complete(tmp_path, monkeypatch):
    from app.catalog import load_catalog
    from scripts import prepare_models as mod
    src = snapshot(tmp_path)
    (src / "model.safetensors").unlink()
    monkeypatch.setattr(mod, "snapshot_download", lambda **kw: str(src))
    dst = tmp_path / "destination"
    with pytest.raises(ValueError):
        mod.prepare_model(load_catalog()["qwen3-0.6b"], dst)
    assert not (dst / '.model-ready.json').exists()


def test_existing_complete_model_skips_download(tmp_path, monkeypatch):
    from app.catalog import load_catalog
    from scripts import prepare_models as mod
    src = snapshot(tmp_path)
    spec = load_catalog()["qwen3-0.6b"]
    monkeypatch.setattr(mod, "snapshot_download", lambda **kw: str(src))
    dst = tmp_path / "destination"
    mod.prepare_model(spec, dst)
    def unexpected(**kw):
        raise AssertionError("complete model must stay offline")
    monkeypatch.setattr(mod, "snapshot_download", unexpected)
    assert mod.prepare_model(spec, dst) == dst


def test_missing_shard_index_triggers_online_completion(tmp_path,monkeypatch):
    from app.catalog import load_catalog
    from scripts import prepare_models as mod
    import shutil
    partial=snapshot(tmp_path)
    shard='model.safetensors-00001-of-00001.safetensors'
    (partial/'model.safetensors').rename(partial/shard)
    full=tmp_path/'complete'
    shutil.copytree(partial,full)
    (full/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'weight':shard}}))
    calls=[]
    def download(**kw):
        calls.append(kw.get('local_files_only',False))
        return str(partial if kw.get('local_files_only') else full)
    monkeypatch.setattr(mod,'snapshot_download',download)
    dst=tmp_path/'destination'
    mod.prepare_model(load_catalog()['qwen3.5-2b'],dst)
    assert calls == [True,False]
    assert (dst/'model.safetensors.index.json').is_file()
    assert (dst/'.model-ready.json').is_file()


def test_stale_ready_marker_missing_index_is_repaired(tmp_path,monkeypatch):
    from app.catalog import load_catalog
    from scripts import prepare_models as mod
    import shutil
    src=snapshot(tmp_path)
    shard='model.safetensors-00001-of-00001.safetensors'
    (src/'model.safetensors').rename(src/shard)
    dst=tmp_path/'destination'
    shutil.copytree(src,dst)
    spec=load_catalog()['qwen3.5-2b']
    (dst/'.model-ready.json').write_text(json.dumps({'repo_id':spec.repo_id,'revision':spec.revision,'files':{p.name:p.stat().st_size for p in dst.iterdir()}}))
    (src/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'weight':shard}}))
    monkeypatch.setattr(mod,'snapshot_download',lambda **kw:str(src))
    mod.prepare_model(spec,dst)
    assert (dst/'model.safetensors.index.json').exists()


def test_the_enabled_models_are_prepared_unless_others_are_named(monkeypatch):
    from app.catalog import load_catalog
    from app.settings import Settings
    from scripts import prepare_models as mod
    catalog = load_catalog()
    monkeypatch.setattr('app.settings.mlx_available', lambda: True)
    assert mod.chosen_models(Settings(models=''), catalog, None, False) == ['qwen3.5-9b', 'qwen3.5-4b']
    assert mod.chosen_models(Settings(models='qwen3.5-2b'), catalog, None, False) == ['qwen3.5-2b']
    assert mod.chosen_models(Settings(), catalog, ['qwen3-0.6b', 'qwen3-0.6b', 'qwen3.5-2b'], False) == \
        ['qwen3-0.6b', 'qwen3.5-2b']  # e.g. for Docker on a Mac
    assert mod.chosen_models(Settings(), catalog, None, True) == list(catalog)


def test_main_prepares_each_chosen_model_in_its_directory(tmp_path, monkeypatch, capsys):
    from scripts import prepare_models as mod
    prepared = []
    monkeypatch.setenv('MODELS_DIR', str(tmp_path))
    monkeypatch.setattr(mod, 'prepare_model', lambda spec, destination: prepared.append((spec.id, destination)))
    mod.main(['--model', 'qwen3.5-4b', '--model', 'qwen3-0.6b'])
    assert prepared == [('qwen3.5-4b', tmp_path / 'qwen3.5-4b'), ('qwen3-0.6b', tmp_path / 'qwen3-0.6b')]
    assert 'Preparing qwen3.5-4b (32f3e8ecf65426fc3306969496342d504bfa13f3)' in capsys.readouterr().out


def test_mlx_shards_with_an_index_are_validated_like_any_checkpoint(tmp_path, monkeypatch):
    """MLX repos ship config.json, tokenizer files and safetensors shards named by an index, as Transformers ones."""
    from safetensors.numpy import save_file
    import numpy as np
    from app.catalog import load_catalog
    from scripts import prepare_models as mod
    src = tmp_path / 'snapshot'
    src.mkdir()
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'processor_config.json']:
        (src / name).write_text('{}')
    (src / 'chat_template.jinja').write_text('{{ messages }}')
    for n in (1, 2):
        save_file({f'w{n}': np.zeros((2, 2), dtype=np.uint32)}, src / f'model-0000{n}-of-00002.safetensors')
    (src / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
        'w1': 'model-00001-of-00002.safetensors', 'w2': 'model-00002-of-00002.safetensors'}}))
    monkeypatch.setattr(mod, 'snapshot_download', lambda **kw: str(src))
    destination = mod.prepare_model(load_catalog()['qwen3.5-9b'], tmp_path / 'qwen3.5-9b')
    assert (destination / 'chat_template.jinja').is_file() and (destination / 'model-00002-of-00002.safetensors').is_file()
    assert json.loads((destination / '.model-ready.json').read_text())['repo_id'] == 'mlx-community/Qwen3.5-9B-MLX-4bit'
