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
    # The Transformers models, whatever this machine would enable by default.
    options={'device':'cpu','data_dir':os.environ['DATA_DIR'],'auth_enabled':False,'models':'qwen3.5-2b,qwen3-0.6b',**settings}
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

@pytest.mark.parametrize('patch',[{'model':'missing'},{'model':'qwen3.5-9b'},{'max_tokens':0},{'max_tokens':4097},{'max_tokens':'512'},{'temperature':-0.1},{'temperature':2.1},{'temperature':'0.2'},{'reasoning_effort':'extreme'},{'reasoning_effort':True},{'messages':[]},{'messages':[{'role':'tool','content':'bad'}]}])
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


class Mlx(Generator):
    """A manager whose MLX models report their own states, as ModelManager.status() does."""
    default_model='qwen3.5-9b'
    states={'qwen3.5-9b':'ready','qwen3.5-4b':'starting'}
    def status(self):
        return {**super().status(),'models':{model_id:{'state':state,'runtime':'mlx','label':model_id,'error':None}
                                             for model_id,state in self.states.items()}}


def mlx_client(monkeypatch,outputs=('hello',),**settings):
    monkeypatch.setattr('app.settings.mlx_available',lambda:True)
    return client(generator=Mlx(outputs),models='',**settings)  # the platform default: every MLX model


def test_model_list_describes_each_enabled_model(monkeypatch):
    c,g,p=mlx_client(monkeypatch)
    with c:
        r=c.get('/v1/models').json()
        assert r['object'] == 'list' and r['default'] == 'qwen3.5-9b'
        assert [m['id'] for m in r['data']] == ['qwen3.5-9b','qwen3.5-4b']
        assert r['data'][0] == {'id':'qwen3.5-9b','object':'model','repo_id':'mlx-community/Qwen3.5-9B-MLX-4bit',
                                'revision':'938d8919941c6e7efd3c7150eff7fe9d12afa631','label':'Qwen3.5 9B',
                                'summary':'더 정확함','vision':True,'reasoning':True,
                                'efforts':['none','light','medium','high','max'],'context_tokens':32768,'state':'ready'}
        assert r['data'][1]['state'] == 'starting'
        health=c.get('/health').json()
        assert health['ready_for_chat'] is True
        assert health['inference']['models']['qwen3.5-4b'] == {'state':'starting','runtime':'mlx','label':'qwen3.5-4b','error':None}


def test_transformers_models_report_files_and_the_shared_context(monkeypatch):
    c,g,p=client()
    with c:
        two,small=c.get('/v1/models').json()['data']
        assert (two['id'],two['state'],two['context_tokens'],two['vision']) == ('qwen3.5-2b','ready',8192,True)
        assert (small['id'],small['label'],small['vision']) == ('qwen3-0.6b','Qwen3 0.6B',False)


def test_a_model_without_reasoning_offers_no_efforts():
    from app.catalog import ModelSpec
    from app.main import model_card
    from app.settings import Settings
    spec=ModelSpec(id='m',repo_id='r',revision='v',directory='m',loader='mlx',reasoning=False,context_tokens=4096)
    card=model_card(spec,'stopped',Settings())
    assert (card['efforts'],card['reasoning'],card['context_tokens'],card['state']) == ([],False,4096,'stopped')


def test_health_is_not_ready_while_the_default_model_starts(monkeypatch):
    monkeypatch.setattr(Mlx,'states',{'qwen3.5-9b':'starting','qwen3.5-4b':'ready'})
    c,g,p=mlx_client(monkeypatch)
    with c:
        assert c.get('/health').json()['ready_for_chat'] is False


def test_requests_without_a_model_use_the_default_and_pass_effort_and_temperature(monkeypatch):
    c,g,p=mlx_client(monkeypatch,['a','b','c'])
    with c:
        r=c.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'hi'}],'use_pubmed':False,
                                              'reasoning_effort':'high','max_tokens':4096})
        assert r.status_code == 200 and r.json()['model'] == 'qwen3.5-9b'
        assert (g.model_id,g.effort,g.temperature,g.max_tokens,g.tenant) == ('qwen3.5-9b','high',None,4096,None)
        c.post('/v1/chat/completions',json={'model':'qwen3.5-4b','messages':[{'role':'user','content':'hi'}],
                                            'use_pubmed':False,'thinking':True,'temperature':0.4})
        assert (g.model_id,g.effort,g.temperature) == ('qwen3.5-4b','medium',0.4)  # the old switch means medium
        c.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'hi'}],'use_pubmed':False,
                                            'thinking':True,'reasoning_effort':'none'})
        assert g.effort == 'none'
        r=c.post('/v1/chat/completions',json={'model':'qwen3.5-2b','messages':[{'role':'user','content':'hi'}]})
        assert r.status_code == 422 and r.json()['error']['code'] == 'unknown_model'  # in models.json, not enabled


def test_reasoning_is_dropped_for_models_without_it(monkeypatch):
    import dataclasses
    import app.main
    original=app.main.load_catalog
    def catalog():
        models=original()
        return {**models,'qwen3-0.6b':dataclasses.replace(models['qwen3-0.6b'],reasoning=False)}
    monkeypatch.setattr(app.main,'load_catalog',catalog)
    c,g,p=client()
    with c:
        c.post('/v1/chat/completions',json={'model':'qwen3-0.6b','messages':[{'role':'user','content':'hi'}],
                                            'use_pubmed':False,'reasoning_effort':'max'})
        assert g.effort == 'none'


def test_the_vision_error_names_no_other_model():
    from tests.test_images import image_message
    c,g,p=client()
    with c:
        r=c.post('/v1/chat/completions',json={'model':'qwen3-0.6b','use_pubmed':False,'messages':[image_message()]})
        assert r.json()['error']['code'] == 'vision_unsupported'
        assert 'qwen3.5' not in r.json()['error']['message']


def test_accounts_salt_the_prefix_cache_per_user():
    from tests.test_auth import accounts, sign_in
    c,g,p=accounts(outputs=['one','two'])
    with c:
        sign_in(c)
        storage=c.app.state.storage
        user_id=storage.find_user('alice')[0]
        tenant=f'user-{user_id}-{storage.user_created(user_id)}'
        c.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'hi'}],'use_pubmed':False})
        assert g.tenant == tenant
        events=events_from(c.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'hi'}],
                                                               'use_pubmed':False,'stream':True}))
        assert events[-1]['type'] == 'done' and g.tenant == tenant
        assert g.prefix is None  # each account has its own cache, so nothing is warmed for everyone


def test_an_account_that_reuses_a_removed_accounts_id_gets_its_own_cache():
    from tests.test_auth import password_hash, sign_in
    c,g,p=client(['one','two'],auth_enabled=True)
    ask={'messages':[{'role':'user','content':'hi'}],'use_pubmed':False}
    with c:
        storage=c.app.state.storage
        storage.clock=lambda:1_760_000_000
        first=storage.add_user('alice',password_hash())
        sign_in(c)
        c.post('/v1/chat/completions',json=ask)
        assert g.tenant == f'user-{first}-1760000000'
        storage.remove_user('alice')
        assert c.post('/v1/chat/completions',json=ask).status_code == 401  # the session went with the account
        storage.clock=lambda:1_760_000_100
        assert storage.add_user('bob',password_hash()) == first  # SQLite hands out the newest id again
        sign_in(c,'bob')
        c.post('/v1/chat/completions',json=ask)
        assert g.tenant == f'user-{first}-1760000100'
        storage.user_created=lambda user_id:None  # removed after its session was checked
        r=c.post('/v1/chat/completions',json=ask)
        assert r.status_code == 401 and r.json()['error']['code'] == 'unauthorized'


def test_the_lifespan_starts_the_models_with_the_pubmed_prompt_to_cache():
    import asyncio
    from app.chat_service import GUIDANCE
    c,g,p=client()
    with c:
        assert c.get('/health').status_code == 200
    system,tools=asyncio.run(g.prefix())
    assert system == GUIDANCE and tools[0]['function']['name'] == 'pubmed_search_articles'


def test_a_wrong_models_setting_stops_the_app():
    with pytest.raises(ValueError,match='unknown'):
        client(models='qwen3.5-2b,gpt-5')
