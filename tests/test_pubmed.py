import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import pytest
from mcp.types import Tool, ListToolsResult, CallToolResult, TextContent


def client(monkeypatch, *, result=None, error=None):
    from app.pubmed import PubMedClient
    obj=PubMedClient('http://example.invalid/mcp')
    tools=[Tool.model_validate(t) for t in json.loads((Path(__file__).parent/'fixtures/pubmed-tools.json').read_text())]
    calls=[]
    class Session:
        async def list_tools(self):
            return ListToolsResult(tools=tools+[Tool(name='unrelated',inputSchema={'type':'object'})])
        async def call_tool(self,name,arguments):
            calls.append((name,arguments))
            if error:
                raise error
            return result or CallToolResult(content=[],structuredContent={'pmids':['123'],'totalCount':1})
    @asynccontextmanager
    async def session():
        yield Session()
    monkeypatch.setattr(obj,'_session',session)
    return obj,calls


def test_only_two_tools_exposed(monkeypatch):
    obj,_=client(monkeypatch)
    schemas=asyncio.run(obj.tool_schemas())
    assert {s['function']['name'] for s in schemas} == {'pubmed_search_articles','pubmed_fetch_articles'}
    assert schemas[0]['function']['parameters']['properties']['maxResults']['maximum'] == 5
    assert schemas[1]['function']['parameters']['properties']['pmids']['maxItems'] == 3


def test_unknown_or_invalid_arguments_never_execute(monkeypatch):
    from app.settings import ServiceError
    obj,calls=client(monkeypatch)
    for name,args in [('shell',{}),('pubmed_search_articles',{'query':''}),('pubmed_fetch_articles',{'pmids':['bad']})]:
        with pytest.raises(ServiceError):
            asyncio.run(obj.call(name,args))
    assert calls == []


def test_search_and_fetch_limits(monkeypatch):
    obj,calls=client(monkeypatch)
    asyncio.run(obj.call('pubmed_search_articles',{'query':'diabetes','maxResults':30,'summaryCount':30}))
    asyncio.run(obj.call('pubmed_fetch_articles',{'pmids':['1','2','3','4']}))
    assert calls[0][1]['maxResults'] == 5
    assert calls[0][1]['summaryCount'] == 5
    assert calls[1][1]['pmids'] == ['1','2','3']


def test_large_result_is_bounded_and_marked():
    from app.pubmed import format_tool_result
    text=format_tool_result({'abstract':'한글"\\'*20000})
    assert len(text) <= 12000
    assert json.loads(text)['truncated'] is True

@pytest.mark.parametrize('result',[CallToolResult(content=[TextContent(type='text',text='failed')],isError=True),CallToolResult(content=[],structuredContent={'error':{'message':'NCBI unavailable'}})])
def test_mcp_tool_error_is_not_success(monkeypatch,result):
    from app.settings import ServiceError
    obj,_=client(monkeypatch,result=result)
    with pytest.raises(ServiceError) as e:
        asyncio.run(obj.call('pubmed_search_articles',{'query':'diabetes'}))
    assert e.value.status_code == 502
    assert obj.status()['status'] == 'unavailable'


def test_mcp_timeout_is_not_success(monkeypatch):
    from app.settings import ServiceError
    obj,_=client(monkeypatch,error=TimeoutError('deadline'))
    with pytest.raises(ServiceError) as e:
        asyncio.run(obj.call('pubmed_search_articles',{'query':'diabetes'}))
    assert e.value.status_code == 504


def test_client_construction_does_not_connect():
    from app.pubmed import PubMedClient
    assert PubMedClient('http://example.invalid/mcp').status()['status'] == 'unknown'
