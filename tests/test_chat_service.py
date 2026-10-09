import asyncio
import json
import pytest
from app.settings import ServiceError
from app.model_manager import GenerationResult

SEARCH='pubmed_search_articles'

def call(name=SEARCH,arguments=None):
    return '<tool_call>'+json.dumps({'name':name,'arguments':arguments or {'query':'diabetes'}})+'</tool_call>'

class Generator:
    def __init__(self,outputs):
        self.outputs=iter(outputs)
        self.messages=[]
    async def generate(self,model_id,messages,tools,max_tokens,temperature):
        self.messages.append(list(messages))
        value=next(self.outputs)
        if isinstance(value,Exception):
            raise value
        return GenerationResult(value,'stop',10,5)
    def status(self):
        return {'device':'cpu','loaded_model':None,'model_files':{'qwen3-0.6b':True,'qwen3.5-2b':True}}
    def close(self):
        pass

class PubMed:
    def __init__(self,fail=False):
        self.executed=[]
        self.discoveries=0
        self.fail=fail
    async def tool_schemas(self):
        self.discoveries+=1
        if self.fail:
            raise ServiceError('pubmed_unavailable','offline',502)
        return [{'type':'function','function':{'name':SEARCH,'parameters':{'type':'object'}}}]
    async def validate_call(self,name,arguments):
        if name != SEARCH:
            raise ServiceError('unknown_tool','unknown',502)
        return arguments
    async def call(self,name,arguments):
        self.executed.append((name,arguments))
        return {'pmids':['31542292'],'totalCount':1}
    def status(self):
        return {'status':'unavailable' if self.fail else 'unknown','error':None}
    async def close(self):
        pass


def complete(outputs,pubmed=None,**kw):
    from app.chat_service import ChatService
    from app.schemas import ChatRequest
    generator=Generator(outputs)
    pubmed=pubmed or PubMed()
    request=ChatRequest(messages=[{'role':'user','content':'question'}],**kw)
    return asyncio.run(ChatService(generator,pubmed).complete(request)),generator,pubmed


def test_tool_round_trip_adds_results_and_cleans_control_tokens():
    r,g,p=complete([call(),'Answer<|im_end|>\n<|endoftext|>'])
    assert r.choices[0].message.content.startswith('Answer\n')
    assert '<|im_end|>' not in r.choices[0].message.content
    assert '<|endoftext|>' not in r.choices[0].message.content
    assert len(r.tool_calls) == 1
    assert g.messages[1][-1]['role'] == 'tool'
    assert '31542292' in g.messages[1][-1]['content']

@pytest.mark.parametrize('text',['<tool_call>{bad}</tool_call>','<tool_call>{}', '<tool_call>{"name":"x","arguments":[]}</tool_call>'])
def test_malformed_tool_calls_are_errors(text):
    from app.chat_service import parse_tool_calls
    with pytest.raises(ServiceError):
        parse_tool_calls(text)


def test_all_calls_validated_before_execution():
    pub=PubMed()
    with pytest.raises(ServiceError):
        complete([call()+call('unknown')],pub)
    assert pub.executed == []


def test_multiple_calls_share_four_call_limit():
    pub=PubMed()
    with pytest.raises(ServiceError) as e:
        complete([call()*3,call()*2],pub)
    assert len(pub.executed) == 3
    assert e.value.code == 'tool_limit'
    assert len(e.value.details['tool_calls']) == 3


def test_context_overflow_after_tool_results():
    pub=PubMed()
    with pytest.raises(ServiceError) as e:
        complete([call(),ServiceError('context_overflow','large',413)],pub)
    assert e.value.status_code == 413
    assert len(e.value.details['tool_calls']) == 1


def test_disabled_pubmed_never_connects():
    r,g,p=complete(['hello'],PubMed(fail=True),use_pubmed=False)
    assert r.choices[0].message.content == 'hello'
    assert p.discoveries == 0
    assert p.executed == []


def test_pubmed_outage_is_not_a_successful_answer():
    with pytest.raises(ServiceError) as e:
        complete(['hello'],PubMed(fail=True))
    assert e.value.status_code == 502


def test_qwen35_native_xml_arguments_use_declared_schema_types():
    from app.chat_service import parse_tool_calls
    tools=[{'type':'function','function':{'name':SEARCH,'parameters':{'properties':{'query':{'type':'string'},'maxResults':{'type':'integer'}}}}}]
    raw='<tool_call>\n<function=pubmed_search_articles>\n<parameter=query>2024</parameter>\n<parameter=maxResults>1</parameter>\n</function>\n</tool_call>'
    calls=parse_tool_calls(raw,tools)
    assert calls[0].name == SEARCH
    assert calls[0].arguments == {'query':'2024','maxResults':1}


def test_qwen35_xml_array_and_boolean_parameters():
    from app.chat_service import parse_tool_calls
    tools=[{'function':{'name':'pubmed_fetch_articles','parameters':{'properties':{'pmids':{'type':'array'},'includeMesh':{'type':'boolean'}}}}}]
    raw='<tool_call><function=pubmed_fetch_articles><parameter=pmids>["22745249"]</parameter><parameter=includeMesh>false</parameter></function></tool_call>'
    assert parse_tool_calls(raw,tools)[0].arguments == {'pmids':['22745249'],'includeMesh':False}

@pytest.mark.parametrize('raw',['<tool_call><function=x>junk</function></tool_call>','<tool_call><function=x><parameter=a>1</parameter><parameter=a>2</parameter></function></tool_call>'])
def test_qwen35_malformed_xml_cannot_execute(raw):
    from app.chat_service import parse_tool_calls
    with pytest.raises(ServiceError):
        parse_tool_calls(raw,[])


def test_user_system_message_is_merged_with_tool_guidance():
    from app.chat_service import ChatService
    from app.schemas import ChatRequest
    g,p=Generator(['hello']),PubMed()
    request=ChatRequest(messages=[{'role':'system','content':'Reply in Korean.'},{'role':'user','content':'hi'}])
    asyncio.run(ChatService(g,p).complete(request))
    assert sum(m['role']=='system' for m in g.messages[0]) == 1
    assert 'Reply in Korean.' in g.messages[0][0]['content']


def test_tool_answer_always_contains_grounded_pubmed_link():
    r,_,_=complete([call(),'Summary without citations'])
    assert '31542292' in r.choices[0].message.content
    assert 'https://pubmed.ncbi.nlm.nih.gov/31542292/' in r.choices[0].message.content
