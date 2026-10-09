import pytest


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Each test keeps accounts, chats and uploads in its own directory, never in the repository's data/."""
    path = tmp_path / 'data'
    monkeypatch.setenv('DATA_DIR', str(path))
    monkeypatch.setenv('AUTH_ENABLED', 'false')  # a developer's .env may turn accounts on
    return path
