import pytest


def test_catalog_and_settings_supply_portable_defaults(tmp_path, monkeypatch):
    from app.catalog import load_catalog
    from app.settings import Settings, ROOT
    monkeypatch.chdir(tmp_path)
    catalog = load_catalog()
    assert set(catalog) == {"qwen3-0.6b", "qwen3.5-2b"}
    assert catalog["qwen3.5-2b"].revision == "15852e8c16360a2fea060d615a32b45270f8a8fc"
    assert Settings().models_dir == ROOT / "models"
    assert Settings().max_context_tokens == 4096


def test_invalid_context_or_device_is_rejected():
    from app.settings import Settings
    with pytest.raises(ValueError):
        Settings(max_context_tokens=8193)
    with pytest.raises(ValueError):
        Settings(device="unknown")
