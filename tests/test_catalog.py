import pytest


def test_catalog_and_settings_supply_portable_defaults(tmp_path, monkeypatch):
    from app.catalog import load_catalog
    from app.settings import Settings, ROOT
    monkeypatch.chdir(tmp_path)
    catalog = load_catalog()
    assert list(catalog) == ["qwen3.5-9b", "qwen3.5-4b", "qwen3.5-2b", "qwen3-0.6b"]  # first of a runtime: its default
    assert catalog["qwen3.5-2b"].revision == "15852e8c16360a2fea060d615a32b45270f8a8fc"
    assert catalog["qwen3.5-9b"].revision == "938d8919941c6e7efd3c7150eff7fe9d12afa631"
    assert catalog["qwen3.5-4b"].revision == "32f3e8ecf65426fc3306969496342d504bfa13f3"
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


def test_mlx_entries_carry_their_sampling_context_and_tool_parsing():
    from app.catalog import load_catalog
    nine = load_catalog()['qwen3.5-9b']
    assert (nine.loader, nine.repo_id, nine.directory) == ('mlx', 'mlx-community/Qwen3.5-9B-MLX-4bit', 'qwen3.5-9b')
    assert (nine.label, nine.summary, nine.vision, nine.reasoning, nine.context_tokens) == \
        ('Qwen3.5 9B', '더 정확함', True, True, 32768)
    assert nine.sampling == {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 1.5}
    assert nine.thinking_sampling == {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'presence_penalty': 1.5}
    assert nine.tool_parsing == 'app'
    assert load_catalog()['qwen3.5-4b'].summary == '더 빠름'


def test_optional_fields_default_to_the_old_rules():
    from app.catalog import ModelSpec
    spec = ModelSpec(id='m', repo_id='r', revision='v', directory='m', loader='qwen3_5')
    assert (spec.label, spec.summary, spec.vision, spec.reasoning, spec.context_tokens) == ('m', '', True, True, None)
    assert (spec.sampling, spec.thinking_sampling, spec.tool_parsing) == ({}, {}, 'app')
    assert ModelSpec(id='m', repo_id='r', revision='v', directory='m', loader='causal_lm').vision is False
    assert ModelSpec(id='m', repo_id='r', revision='v', directory='m', loader='mlx', vision=True).vision is True


@pytest.mark.parametrize('entry', [{'loader': 'gguf'}, {'tool_parsing': 'both'}, {'sampling': {'seed': 1}},
                                   {'context_tokens': 0}, {'directory': '../elsewhere'}])
def test_invalid_manifest_entries_are_refused(entry, monkeypatch, tmp_path):
    import json
    import app.catalog
    (tmp_path / 'models.json').write_text(json.dumps([{'id': 'm', 'repo_id': 'r', 'revision': 'v', 'directory': 'm',
                                                       'loader': 'mlx', **entry}]))
    monkeypatch.setattr(app.catalog, 'ROOT', tmp_path)
    with pytest.raises(ValueError):
        app.catalog.load_catalog()


def test_enabled_models_follow_the_platform_unless_models_is_set():
    from app.catalog import load_catalog
    from app.settings import Settings, enabled_models
    catalog = load_catalog()
    assert enabled_models(Settings(models=''), catalog, mlx=True) == ['qwen3.5-9b', 'qwen3.5-4b']
    assert enabled_models(Settings(models=''), catalog, mlx=False) == ['qwen3.5-2b', 'qwen3-0.6b']
    assert enabled_models(Settings(models=' qwen3-0.6b, qwen3.5-2b,qwen3-0.6b'), catalog, mlx=False) == \
        ['qwen3-0.6b', 'qwen3.5-2b']  # the order sets the default; repeats are dropped
    assert enabled_models(Settings(models='qwen3.5-4b,qwen3.5-2b'), catalog, mlx=True) == ['qwen3.5-4b', 'qwen3.5-2b']


def test_models_from_the_environment_and_wrong_ids_fail_at_startup(monkeypatch):
    from app.catalog import load_catalog
    from app.settings import Settings, enabled_models
    catalog = load_catalog()
    monkeypatch.setenv('MODELS', 'qwen3.5-4b')
    assert enabled_models(Settings(), catalog, mlx=True) == ['qwen3.5-4b']
    with pytest.raises(ValueError, match='unknown'):
        enabled_models(Settings(models='qwen3.5-9b,gpt-5'), catalog, mlx=True)
    with pytest.raises(ValueError, match='Apple Silicon'):  # no sidecar could ever start
        enabled_models(Settings(models='qwen3.5-9b'), catalog, mlx=False)


def test_mlx_needs_apple_silicon_and_the_package(monkeypatch):
    import importlib.util
    import app.settings as settings
    monkeypatch.setattr(settings.sys, 'platform', 'linux')
    assert settings.mlx_available() is False
    monkeypatch.setattr(settings.sys, 'platform', 'darwin')
    monkeypatch.setattr(settings.platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)
    assert settings.mlx_available() is False
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: object() if name == 'mlx_vlm' else None)
    assert settings.mlx_available() is True


def test_mlx_settings_defaults_and_bounds(monkeypatch):
    from app.settings import Settings
    for name in ['PRELOAD_MODELS', 'MLX_IDLE_MINUTES', 'KV_CACHE_GB']:
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    # Only the default model stays loaded; another one is stopped after 10 minutes without a question.
    assert (settings.preload_models, settings.mlx_idle_minutes) == ('default', 10)
    # The prefix cache stays in memory unless KV_CACHE_GB asks for the disk tier, whose files outlive chats.
    assert settings.kv_cache_gb == 0
    assert (settings.mlx_checkpoint_interval, settings.mlx_checkpoint_entries) == (512, 8)
    assert not hasattr(settings, 'mlx_port')  # sidecars listen on private Unix sockets, not ports
    for field, value in [('kv_cache_gb', -1), ('mlx_checkpoint_entries', 0), ('mlx_idle_minutes', -1),
                         ('preload_models', 'some'), ('preload_models', 2)]:
        with pytest.raises(ValueError):
            Settings(**{field: value})
    monkeypatch.setenv('KV_CACHE_GB', '4')
    monkeypatch.setenv('MLX_IDLE_MINUTES', '0')  # never stop a model
    assert (Settings(_env_file=None).kv_cache_gb, Settings(_env_file=None).mlx_idle_minutes) == (4, 0)


@pytest.mark.parametrize('value,mode', [('default', 'default'), ('all', 'all'), ('none', 'none'), (' All ', 'all'),
                                        # PRELOAD_MODELS was a boolean; an old .env keeps working
                                        ('true', 'all'), ('false', 'none'), ('True', 'all'), ('FALSE', 'none'),
                                        ('1', 'all'), ('0', 'none'), ('yes', 'all'), ('off', 'none')])
def test_preload_models_takes_a_mode_or_the_old_boolean(monkeypatch, value, mode):
    from app.settings import Settings
    monkeypatch.setenv('PRELOAD_MODELS', value)
    assert Settings(_env_file=None).preload_models == mode
    assert (Settings(preload_models=True).preload_models, Settings(preload_models=False).preload_models) == \
        ('all', 'none')
