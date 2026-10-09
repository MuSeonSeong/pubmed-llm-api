import pytest


def test_catalog_and_settings_supply_portable_defaults(tmp_path, monkeypatch):
    from app.catalog import load_catalog
    from app.settings import Settings, ROOT
    monkeypatch.chdir(tmp_path)
    catalog = load_catalog()
    assert set(catalog) == {"qwen3-0.6b", "qwen3.5-2b"}
    assert catalog["qwen3.5-2b"].revision == "15852e8c16360a2fea060d615a32b45270f8a8fc"
    assert Settings().models_dir == ROOT / "models"
    assert Settings().max_context_tokens == 8192


def test_invalid_context_or_device_is_rejected():
    from app.settings import Settings
    with pytest.raises(ValueError):
        Settings(max_context_tokens=8193)
    with pytest.raises(ValueError):
        Settings(device="unknown")


def test_data_settings_default_to_local_mode_in_the_project(monkeypatch):
    from app.settings import Settings, ROOT
    monkeypatch.delenv('DATA_DIR')
    monkeypatch.delenv('AUTH_ENABLED')
    settings = Settings(_env_file=None)
    assert settings.data_dir == ROOT / 'data' and settings.auth_enabled is False
    assert (settings.session_days, settings.user_storage_mb) == (30, 500)


def test_relative_data_dir_resolves_against_the_project(monkeypatch):
    from app.settings import Settings, ROOT
    monkeypatch.setenv('DATA_DIR', 'var/chats')
    monkeypatch.setenv('AUTH_ENABLED', 'true')
    settings = Settings()
    assert settings.data_dir == ROOT / 'var' / 'chats' and settings.auth_enabled is True


@pytest.mark.parametrize('field,value', [('session_days', 0), ('session_days', 366), ('user_storage_mb', 0),
                                         ('user_storage_mb', 100001)])
def test_session_and_storage_limits_are_bounded(field, value):
    from app.settings import Settings
    with pytest.raises(ValueError):
        Settings(**{field: value})
