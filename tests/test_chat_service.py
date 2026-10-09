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
    async def generate(self,model_id,messages,tools,max_tokens,temperature,on_text=None,cancel=None,thinking=False,continue_final=False):
        self.messages.append(list(messages))
        self.thinking=thinking
        self.continue_final=continue_final
        value=next(self.outputs)
        if isinstance(value,Exception):
            raise value
        if isinstance(value,GenerationResult):
            return value
        if on_text:
            import anyio
            # Like the real manager, report cumulative text from a worker thread.
            await anyio.to_thread.run_sync(lambda:[on_text(value[:i]) for i in range(0,len(value)+1)])
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


def stream(outputs,pubmed=None,cancel=None,**kw):
    import threading
    from app.chat_service import ChatService
    from app.schemas import ChatRequest
    events=[]
    request=ChatRequest(messages=[{'role':'user','content':'question'}],stream=True,**kw)
    asyncio.run(ChatService(Generator(outputs),pubmed or PubMed()).stream(request,events.append,cancel or threading.Event()))
    return events


def test_stream_emits_text_deltas_tool_progress_and_final_answer():
    events=stream([call(),'Answer text<|im_end|>'])
    kinds=[e['type'] for e in events]
    assert kinds[-1] == 'done' and kinds.count('done') == 1
    tools=[(e['status'],e['name']) for e in events if e['type']=='tool']
    assert tools == [('running',SEARCH),('done',SEARCH)]
    deltas=''.join(e['content'] for e in events if e['type']=='delta')
    assert deltas == 'Answer text'
    assert '<tool_call>' not in deltas
    final=events[-1]
    assert final['choices'][0]['message']['content'].startswith('Answer text')
    assert '31542292' in final['choices'][0]['message']['content']
    assert len(final['tool_calls']) == 1
    assert {'type':'status','stage':'loading_model','model':'qwen3.5-2b'} in events
    stages=[e['stage'] for e in events if e['type']=='status']
    assert stages[0] == 'loading_model' and 'generating' in stages


def test_stream_failure_ends_with_error_event_and_traces():
    events=stream([call()*3,call()*2])
    assert events[-1]['type'] == 'error'
    assert events[-1]['status'] == 502
    assert events[-1]['error']['code'] == 'tool_limit'
    assert len(events[-1]['error']['details']['tool_calls']) == 3


def test_cancelled_generation_finishes_with_partial_text():
    events=stream([GenerationResult('Partial answer <tool_','cancelled',3,2)],use_pubmed=False)
    assert events[-1]['type'] == 'done'
    assert events[-1]['choices'][0]['finish_reason'] == 'cancelled'
    assert events[-1]['choices'][0]['message']['content'] == 'Partial answer'


@pytest.mark.parametrize('raw,shown',[
    ('Hello <tool_call>{"name"', 'Hello '),
    ('Hello <tool', 'Hello '),
    ('Hello <|im_e', 'Hello '),
    ('<think>plan</think>\n\nHi', 'Hi'),
    ('<think>still thinking', ''),
    ('a < b and c', 'a < b and c'),
    ('Done<|im_end|>', 'Done'),
])
def test_visible_text_hides_markup_until_it_is_unambiguous(raw,shown):
    from app.chat_service import visible_text
    assert visible_text(raw) == shown


class Budget(Generator):
    """Overflows while the prompt holds more than `limit` messages."""
    def __init__(self,outputs,limit):
        super().__init__(outputs)
        self.limit=limit
    async def generate(self,model_id,messages,tools,max_tokens,temperature,**kw):
        if len(messages) > self.limit:
            self.messages.append(list(messages))
            raise ServiceError('context_overflow','large',413)
        return await super().generate(model_id,messages,tools,max_tokens,temperature,**kw)


def history(*contents):
    roles=['user','assistant']
    return [{'role':roles[i%2],'content':c} for i,c in enumerate(contents)]


def test_truncation_drops_oldest_turns_but_keeps_system_and_latest_question():
    from app.chat_service import ChatService
    from app.schemas import ChatRequest
    g=Budget(['ok'],limit=3)
    request=ChatRequest(messages=[{'role':'system','content':'Be brief.'},*history('q1','a1','q2','a2','q3')],
                        use_pubmed=False,truncate_history=True)
    r=asyncio.run(ChatService(g,PubMed()).complete(request))
    assert r.dropped_messages == 4
    assert [m['content'] for m in g.messages[-1]] == ['Be brief.','q3']
    assert r.choices[0].message.content == 'ok'


def test_overflow_without_truncation_or_with_only_the_question_is_an_error():
    from app.chat_service import ChatService
    from app.schemas import ChatRequest
    for truncate,messages in [(False,history('q1','a1','q2')),(True,history('only question'))]:
        g=Budget(['ok'],limit=0)
        request=ChatRequest(messages=messages,use_pubmed=False,truncate_history=truncate)
        with pytest.raises(ServiceError) as e:
            asyncio.run(ChatService(g,PubMed()).complete(request))
        assert e.value.code == 'context_overflow'


def test_drop_oldest_turn_removes_a_whole_turn():
    from app.chat_service import drop_oldest_turn
    messages=history('q1','a1','q2')
    assert drop_oldest_turn(messages) == 2
    assert [m['content'] for m in messages] == ['q2']
    assert drop_oldest_turn(messages) == 0


@pytest.mark.parametrize('raw,expected',[
    ('<think>\nplan\n</think>\n\nAnswer', ('plan','\n\nAnswer')),   # Qwen3 writes the tag
    ('plan\n</think>\n\nAnswer', ('plan','\n\nAnswer')),             # Qwen3.5 pre-fills it
    ('<thi', ('',None)),
    ('<think>still going', ('still going',None)),
    ('still going</thi', ('still going',None)),
])
def test_split_reasoning(raw,expected):
    from app.chat_service import split_reasoning
    assert split_reasoning(raw) == expected


def test_thinking_streams_reasoning_separately_and_returns_it():
    events=stream(['<think>\nLet me add.\n</think>\n\n4'],use_pubmed=False,thinking=True)
    reasoning=''.join(e['content'] for e in events if e['type']=='reasoning')
    answer=''.join(e['content'] for e in events if e['type']=='delta')
    assert reasoning == 'Let me add.'
    assert answer == '4'
    message=events[-1]['choices'][0]['message']
    assert message['content'] == '4'
    assert message['reasoning_content'] == 'Let me add.'


def test_thinking_ignores_tool_markup_inside_reasoning_and_runs_real_calls():
    raw='<think>I could write <tool_call> here</think>'+call()
    r,g,p=complete([raw,'<think>ok</think>Final'],thinking=True)
    assert len(p.executed) == 1
    assert r.choices[0].message.content.startswith('Final')
    assert r.choices[0].message.reasoning_content == 'I could write <tool_call> here\n\nok'
    assert g.thinking is True


def test_thinking_cut_off_by_length_returns_reasoning_without_answer():
    r,_,_=complete([GenerationResult('<think>long reasoning','length',5,5)],use_pubmed=False,thinking=True)
    assert r.choices[0].message.content == ''
    assert r.choices[0].message.reasoning_content == 'long reasoning'
    assert r.choices[0].finish_reason == 'length'


def test_without_thinking_reasoning_field_is_absent():
    r,g,_=complete(['hello'],use_pubmed=False)
    assert r.choices[0].message.reasoning_content is None
    assert g.thinking is False


def test_stream_reports_queueing_behind_another_request():
    import threading
    from app.chat_service import ChatService
    from app.schemas import ChatRequest
    class Busy(Generator):
        def status(self):
            return {**super().status(),'busy':True,'loaded_model':'qwen3.5-2b'}
    events=[]
    request=ChatRequest(messages=[{'role':'user','content':'q'}],use_pubmed=False,stream=True)
    asyncio.run(ChatService(Busy(['hi']),PubMed()).stream(request,events.append,threading.Event()))
    stages=[e['stage'] for e in events if e['type']=='status']
    assert stages[0] == 'queued'
    assert 'loading_model' not in stages


def test_sources_merge_search_summaries_and_fetched_articles():
    from app.chat_service import collect_sources
    from app.schemas import ToolTrace
    search=ToolTrace(name=SEARCH,arguments={},result={'pmids':['24393785','37185680','bad id'],'summaries':[
        {'pmid':'24393785','title':'Metformin mode of action.','authors':'Pernicova I, Korbonits M',
         'source':'Nat Rev Endocrinol','pubDate':'2014-03-01','doi':'10.1038/x','pubmedUrl':'javascript:alert(1)'}]})
    fetch=ToolTrace(name='pubmed_fetch_articles',arguments={},result={'articles':[
        {'pmid':'24393785','abstractText':'Metformin has been...'},
        {'pmid':'37185680','title':'Metformin and Cancer.','authors':[{'lastName':'Yu','initials':'OHY'},
         {'lastName':'Suissa','initials':'S'},{'lastName':'A','initials':'B'},{'lastName':'C','initials':'D'}],
         'journalInfo':{'isoAbbreviation':'Diabetes Care','publicationDate':{'year':'2023'}}}]})
    first,second=collect_sources([search,fetch])
    assert first.pmid == '24393785' and first.abstract_fetched
    assert first.url == 'https://pubmed.ncbi.nlm.nih.gov/24393785/'
    assert (first.title,first.journal,first.year) == ('Metformin mode of action.','Nat Rev Endocrinol','2014')
    assert second.authors == 'Yu OHY, Suissa S, A B et al.'
    assert (second.journal,second.year,second.abstract_fetched) == ('Diabetes Care','2023',False)


def test_answers_carry_sources_only_for_tool_results():
    r,_,_=complete([call(),'Answer'])
    assert [s.pmid for s in r.sources] == ['31542292']
    r,_,_=complete(['hello'],use_pubmed=False)
    assert r.sources == []


def test_response_has_openai_style_metadata_and_usage_over_all_rounds():
    r,_,_=complete([call(),'Answer'])
    assert r.id.startswith('chatcmpl-') and r.object == 'chat.completion' and r.created > 0
    assert (r.usage.prompt_tokens,r.usage.completion_tokens,r.usage.total_tokens) == (20,10,30)


def test_null_lists_in_tool_results_do_not_break_the_answer():
    from app.chat_service import append_citations, collect_sources
    from app.schemas import ToolTrace
    traces=[ToolTrace(name=SEARCH,arguments={},result={'pmids':None,'summaries':None,
            'articles':[{'pmid':'1','authors':None,'journalInfo':None}]})]
    assert [s.pmid for s in collect_sources(traces)] == ['1']
    assert 'PMID [1]' in append_citations('answer',traces)


def test_continue_final_extends_the_answer_without_tools_and_keeps_leading_space():
    import threading
    from app.chat_service import ChatService
    from app.schemas import ChatRequest
    g,p=Generator([' 두 번째 문단입니다.<|im_end|>']),PubMed()
    request=ChatRequest(messages=[{'role':'user','content':'긴 글'},{'role':'assistant','content':'첫 문단.'}],
                        continue_final=True,thinking=True,stream=True)
    events=[]
    asyncio.run(ChatService(g,p).stream(request,events.append,threading.Event()))
    assert ''.join(e['content'] for e in events if e['type']=='delta') == ' 두 번째 문단입니다.'
    assert events[-1]['choices'][0]['message']['content'] == ' 두 번째 문단입니다.'
    assert g.continue_final is True and g.thinking is False
    assert p.discoveries == 0
    assert g.messages[0][-1] == {'role':'assistant','content':'첫 문단.'}


def test_continue_final_requires_a_trailing_assistant_message():
    from pydantic import ValidationError
    from app.schemas import ChatRequest
    with pytest.raises(ValidationError):
        ChatRequest(messages=[{'role':'user','content':'q'}],continue_final=True)
