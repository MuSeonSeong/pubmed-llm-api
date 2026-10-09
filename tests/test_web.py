from tests.test_api import client


def test_chat_page_and_assets_are_served_without_model_loading():
    c, generator, _ = client()
    with c:
        page = c.get('/')
        assert page.status_code == 200
        assert 'text/html' in page.headers['content-type']
        assert 'lang="ko"' in page.text
        assert '/static/app.mjs' in page.text
        for name, mime in [('app.mjs', 'javascript'), ('chat.mjs', 'javascript'),
                           ('style.css', 'text/css')]:
            asset = c.get('/static/' + name)
            assert asset.status_code == 200
            assert mime in asset.headers['content-type']
        assert generator.messages == []


def test_static_files_do_not_expose_project_sources():
    c, _, _ = client()
    with c:
        assert c.get('/static/../settings.py').status_code == 404
        assert c.get('/static/%2e%2e/settings.py').status_code == 404
