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
