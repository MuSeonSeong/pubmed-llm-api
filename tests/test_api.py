import json
import os
import pytest
from fastapi.testclient import TestClient
from tests.test_chat_service import Generator, PubMed, call


def client(outputs=('hello',), *, offline=False, generator=None, base_url='http://testserver', address='testclient',
           **settings):
    """An app on fakes; its data lives in the directory tests/conftest.py gives each test."""
    from app.main import create_app
    from app.settings import Settings
    g,p=generator or Generator(outputs),PubMed(fail=offline)
    options={'device':'cpu','data_dir':os.environ['DATA_DIR'],'auth_enabled':False,**settings}
    return TestClient(create_app(Settings(**options),g,p),base_url=base_url,client=(address,50000)),g,p


def test_model_list_is_exact_and_does_not_load():
    c,g,p=client()
    with c:
        r=c.get('/v1/models')
        assert r.status_code == 200
        assert {m['id'] for m in r.json()['data']} == {'qwen3-0.6b','qwen3.5-2b'}
        assert r.json()['default'] == 'qwen3.5-2b'
        assert g.messages == []


def test_defaults_and_numeric_bounds():
    c,g,p=client()
    with c:
        r=c.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'hi'}],'use_pubmed':False})
        assert r.status_code == 200
        assert r.json()['model'] == 'qwen3.5-2b'
        assert r.json()['choices'][0]['message']['content'] == 'hello'

@pytest.mark.parametrize('patch',[{'model':'missing'},{'max_tokens':0},{'max_tokens':2049},{'max_tokens':'512'},{'temperature':-0.1},{'temperature':2.1},{'temperature':'0.2'},{'messages':[]},{'messages':[{'role':'tool','content':'bad'}]}])
def test_invalid_model_role_or_numeric_bounds_are_422(patch):
    c,g,p=client()
    with c:
        payload={'messages':[{'role':'user','content':'hi'}],'use_pubmed':False,**patch}
        r=c.post('/v1/chat/completions',json=payload)
        assert r.status_code == 422
        assert g.messages == []

@pytest.mark.parametrize('temperature',[float('nan'),float('inf')])
def test_non_finite_temperature_returns_422(temperature):
    c,g,p=client()
    with c:
        r=c.post('/v1/chat/completions',content=json.dumps({'messages':[{'role':'user','content':'hi'}],'temperature':temperature}),headers={'Content-Type':'application/json'})
        assert r.status_code == 422


def test_health_reports_unloaded_and_pubmed_unavailable():
    c,g,p=client(offline=True)
    with c:
        r=c.get('/health')
        assert r.status_code == 200
        assert r.json()['inference']['loaded_model'] is None
        assert r.json()['pubmed']['status'] == 'unavailable'


def test_pubmed_outage_does_not_break_plain_chat():
    c,g,p=client(offline=True)
    with c:
        r=c.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'hi'}],'use_pubmed':False})
        assert r.status_code == 200
        assert p.discoveries == 0


def test_tool_failure_returns_error_and_trace():
    c,g,p=client([call()*3,call()*2])
    with c:
        r=c.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'find'}]})
        assert r.status_code == 502
        assert r.json()['error']['code'] == 'tool_limit'
        assert len(r.json()['error']['details']['tool_calls']) == 3


def test_messages_without_user_query_are_422():
    c,g,p=client()
    with c:
        r=c.post('/v1/chat/completions',json={'messages':[{'role':'system','content':'hello'}],'use_pubmed':False})
        assert r.status_code == 422
        assert g.messages == []


def events_from(response):
    return [json.loads(line[6:]) for line in response.text.split('\n') if line.startswith('data: ')]


def test_streaming_chat_returns_server_sent_events():
    c,g,p=client(['hello there'])
    with c:
        r=c.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'hi'}],'use_pubmed':False,'stream':True})
        assert r.status_code == 200
        assert r.headers['content-type'].startswith('text/event-stream')
        assert 'content-length' not in r.headers
        events=events_from(r)
        assert ''.join(e['content'] for e in events if e['type']=='delta') == 'hello there'
        assert events[-1]['type'] == 'done'
        assert events[-1]['choices'][0]['message']['content'] == 'hello there'


def test_streaming_service_error_is_reported_in_stream():
    c,g,p=client(offline=True)
    with c:
        r=c.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'find'}],'stream':True})
        assert r.status_code == 200
        events=events_from(r)
        assert events[-1]['type'] == 'error'
        assert events[-1]['error']['code'] == 'pubmed_unavailable'


def test_streaming_request_is_still_validated_before_streaming():
    c,g,p=client()
    with c:
        r=c.post('/v1/chat/completions',json={'messages':[],'stream':True})
        assert r.status_code == 422


def test_importing_the_app_creates_no_data_directory(tmp_path):
    import subprocess
    import sys
    from app.catalog import ROOT
    target=tmp_path/'never-created'
    done=subprocess.run([sys.executable,'-c','import app.main'],cwd=ROOT,capture_output=True,text=True,
                        env={**os.environ,'DATA_DIR':str(target)},timeout=300)
    assert done.returncode == 0, done.stderr
    assert not target.exists()


def test_the_database_appears_only_when_data_is_first_used(data_dir):
    c,g,p=client()
    with c:
        assert c.get('/health').status_code == 200 and c.get('/').status_code == 200
        assert c.get('/api/session').status_code == 200
        assert not data_dir.exists()
        assert c.get('/api/chats').status_code == 200
        assert (data_dir/'chat.db').is_file()
