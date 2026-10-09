import asyncio
import json
import logging
from pathlib import Path
import pytest
from app.settings import ServiceError
from tests.test_chat_service import Generator, PubMed, SEARCH, call, complete

SCHEMAS = json.loads((Path(__file__).parent / 'fixtures' / 'pubmed-tools.json').read_text())


def schema(name):
    return next(tool['inputSchema'] for tool in SCHEMAS if tool['name'] == name)


SEARCH_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['query'],
    'properties': {
        'query': {'type': 'string', 'minLength': 1},
        'maxResults': {'type': 'integer', 'minimum': 0, 'maximum': 1000},
        'sort': {'type': 'string', 'enum': ['relevance', 'pub_date', 'author', 'journal']},
        'species': {'type': 'string', 'enum': ['humans', 'animals']},
        'author': {'type': 'string'},
        'hasAbstract': {'type': 'boolean'},
        'meshTerms': {'type': 'array', 'items': {'type': 'string'}},
        'dateRange': {'type': 'object', 'required': ['minDate', 'maxDate'], 'properties': {
            'minDate': {'type': 'string', 'pattern': r'^$|^\d{4}([/\-.]\d{1,2}([/\-.]\d{1,2})?)?$'},
            'maxDate': {'type': 'string', 'pattern': r'^$|^\d{4}([/\-.]\d{1,2}([/\-.]\d{1,2})?)?$'},
            'dateType': {'type': 'string', 'enum': ['pdat', 'mdat', 'edat']}}},
    },
}
FETCH_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['pmids'],
    'properties': {'pmids': {'type': 'array', 'minItems': 1, 'items': {'type': 'string', 'pattern': r'^\d+$'}},
                   'includeMesh': {'type': 'boolean'}},
}


def repaired(arguments, schema):
    from jsonschema import validate
    from app.pubmed import repair_arguments
    notes = []
    fixed = repair_arguments(arguments, schema, notes)
    validate(fixed, schema)
    return fixed, notes


def test_flattened_date_bounds_move_into_date_range():
    fixed, notes = repaired({'query': 'metformin', 'minDate': '2020', 'maxDate': '2025'}, SEARCH_SCHEMA)
    assert fixed == {'query': 'metformin', 'dateRange': {'minDate': '2020', 'maxDate': '2025'}}
    assert any('moved into dateRange' in note for note in notes)


def test_one_sided_and_wordy_dates_become_an_open_range():
    fixed, _ = repaired({'query': 'x', 'minDate': 2020}, SEARCH_SCHEMA)
    assert fixed['dateRange'] == {'minDate': '2020', 'maxDate': '3000'}
    fixed, _ = repaired({'query': 'x', 'dateRange': {'maxDate': '2015년까지'}}, SEARCH_SCHEMA)
    assert fixed['dateRange'] == {'minDate': '1000', 'maxDate': '2015'}


def test_unknown_and_empty_fields_are_dropped_and_types_coerced():
    fixed, _ = repaired({'query': 'x', 'colour': 'red', 'author': '', 'maxResults': '3', 'hasAbstract': 'true',
                         'sort': 'Pub Date', 'meshTerms': 'Diabetes Mellitus, Type 2'}, SEARCH_SCHEMA)
    assert fixed == {'query': 'x', 'maxResults': 3, 'hasAbstract': True, 'sort': 'pub_date',
                     'meshTerms': ['Diabetes Mellitus, Type 2']}  # one heading, never split on spaces or commas


def test_misnamed_filters_are_mapped_instead_of_silently_lost():
    fixed, _ = repaired({'query': 'x', 'start_date': '2020', 'end_date': '2022', 'mesh_term': ['Heart Failure'],
                         'max_results': '2'}, SEARCH_SCHEMA)
    assert fixed == {'query': 'x', 'dateRange': {'minDate': '2020', 'maxDate': '2022'},
                     'meshTerms': ['Heart Failure'], 'maxResults': 2}
    assert repaired({'query': 'x', 'year': 2020}, SEARCH_SCHEMA)[0]['dateRange'] == {'minDate': '2020', 'maxDate': '2020'}


@pytest.mark.parametrize('bounds, expected', [
    ({'minDate': '2020~2022'}, {'minDate': '2020', 'maxDate': '2022'}),
    ({'maxDate': '2020–2022'}, {'minDate': '2020', 'maxDate': '2022'}),
    ({'minDate': '2020-2022'}, {'minDate': '2020', 'maxDate': '2022'}),  # never the impossible month "2020-20"
    ({'minDate': '2020년부터 2022년까지'}, {'minDate': '2020', 'maxDate': '2022'}),
    ({'minDate': '2020-12'}, {'minDate': '2020-12', 'maxDate': '3000'}),
    ({'minDate': '2020', 'maxDate': ''}, {'minDate': '2020', 'maxDate': '3000'}),  # an empty bound is an open one
])
def test_written_ranges_and_empty_bounds(bounds, expected):
    assert repaired({'query': 'x', 'dateRange': bounds}, SEARCH_SCHEMA)[0]['dateRange'] == expected


def test_a_date_range_without_values_means_no_date_filter():
    fixed, _ = repaired({'query': 'x', 'dateRange': {'minDate': None, 'maxDate': None, 'dateType': 'pdat'}}, SEARCH_SCHEMA)
    assert fixed == {'query': 'x'}
    fixed, _ = repaired({'query': 'x', 'dateRange': '{"minDate": "2020"}', 'maxDate': '2022'}, SEARCH_SCHEMA)
    assert fixed['dateRange'] == {'minDate': '2020', 'maxDate': '2022'}


def test_huge_numbers_are_left_for_validation_instead_of_crashing():
    from jsonschema import ValidationError, validate
    from app.pubmed import repair_arguments
    fixed = repair_arguments({'query': 'x', 'maxResults': '1' + '0' * 400}, SEARCH_SCHEMA)
    with pytest.raises(ValidationError):
        validate(fixed, SEARCH_SCHEMA)


def test_pmids_lose_labels_and_separators_only():
    assert repaired({'pmids': 'PMID: 24393785, 37185680'}, FETCH_SCHEMA)[0] == {'pmids': ['24393785', '37185680']}
    assert repaired({'pmids': [24393785, 'PMID 37185680']}, FETCH_SCHEMA)[0] == {'pmids': ['24393785', '37185680']}
    assert repaired({'pmids': ['24393785, 37185680']}, FETCH_SCHEMA)[0] == {'pmids': ['24393785', '37185680']}
    assert repaired({'pmid': '24393785'}, FETCH_SCHEMA)[0] == {'pmids': ['24393785']}
    assert repaired({'pmids': '["24393785"]', 'includeMesh': 'false'}, FETCH_SCHEMA)[0] == {'pmids': ['24393785'], 'includeMesh': False}


@pytest.mark.parametrize('value', [['PMC7095418'], 'PMC7095418', '10.1056/NEJMoa2001017', ['NCT04368728'],
                                   ['24393785 (2014)'], ['n/a']])
def test_other_identifiers_are_never_turned_into_pmids(value):
    from jsonschema import ValidationError, validate
    from app.pubmed import repair_arguments
    fixed = repair_arguments({'pmids': value}, FETCH_SCHEMA)
    with pytest.raises(ValidationError):  # the model is asked to correct it instead of fetching unrelated papers
        validate(fixed, FETCH_SCHEMA)


def test_repair_notes_never_contain_argument_values():
    from app.pubmed import repair_arguments
    notes = []
    repair_arguments({'query': '제 어머니 tamoxifen', 'minDate': '제 어머니(62세)가 2023년', 'sort': 'Pub Date',
                      'meshTerms': 'Breast Neoplasms'}, SEARCH_SCHEMA, notes)
    assert notes and not any('어머니' in note or '62' in note or 'Breast' in note for note in notes)


def test_valid_arguments_are_left_untouched():
    arguments = {'query': 'x', 'maxResults': 2, 'dateRange': {'minDate': '2020', 'maxDate': '2021'}}
    fixed, notes = repaired(arguments, SEARCH_SCHEMA)
    assert fixed == arguments and notes == []


def test_repair_runs_inside_validate_call_against_the_real_fixture_schema():
    from app.pubmed import PubMedClient
    client = PubMedClient('http://unused')
    client._tools = {tool['name']: type('Tool', (), {'inputSchema': tool['inputSchema'], 'description': '',
                     'name': tool['name']})() for tool in SCHEMAS}

    async def schemas():
        return []
    client.tool_schemas = schemas
    args = asyncio.run(client.validate_call(SEARCH, {'query': 'metformin', 'maxResults': '2', 'unknownFlag': True}))
    assert args['maxResults'] == 2 and 'unknownFlag' not in args


class Picky(PubMed):
    """Rejects arguments the way the real server would until the call is correct."""
    async def validate_call(self, name, arguments):
        if name != SEARCH:
            raise ServiceError('unknown_tool', 'unknown', 502)
        if 'minDate' in arguments:
            raise ServiceError('invalid_tool_arguments', "'minDate' was unexpected", 502, {'path': '(arguments)'})
        return arguments


def test_a_rejected_tool_call_is_explained_to_the_model_which_then_succeeds(caplog):
    caplog.set_level(logging.INFO, logger='app')
    bad = call(arguments={'query': 'x', 'minDate': '2020'})
    r, g, p = complete([bad, call(), 'Answer'], Picky())
    assert r.choices[0].message.content.startswith('Answer')
    assert len(p.executed) == 1 and [t.name for t in r.tool_calls] == [SEARCH]
    note = json.loads(g.messages[1][-1]['content'])
    assert g.messages[1][-1]['role'] == 'tool' and note['error'] == 'invalid_tool_arguments'
    assert g.messages[1][-2] == {'role': 'assistant', 'content': bad}
    assert 'asking it to correct (1/2)' in caplog.text


def test_malformed_calls_and_unknown_tools_get_corrections_then_fail_with_the_last_code(caplog):
    caplog.set_level(logging.INFO, logger='app')
    with pytest.raises(ServiceError) as e:
        complete(['<tool_call>{broken', call('unknown'), call('unknown')], Picky())
    assert e.value.code == 'unknown_tool'
    assert 'chat failed: unknown_tool' in caplog.text


def test_a_tool_call_with_pubmed_off_is_redirected_to_a_direct_answer():
    r, g, p = complete([call(), 'Direct answer'], PubMed(fail=True), use_pubmed=False)
    assert r.choices[0].message.content == 'Direct answer'
    assert 'No tools are available' in json.loads(g.messages[1][-1]['content'])['hint']
    assert p.discoveries == 0


def tool_failure(reason, retryable=False, message='PMID 9 not found', recovery='Check the PMID'):
    return ServiceError('pubmed_tool_failed', 'PubMed returned a tool error', 502, {'result': {'error': {
        'code': -32603, 'message': message, 'data': {'reason': reason, 'retryable': retryable, 'recovery': recovery}}}})


class Rejecting(PubMed):
    def __init__(self, failure):
        super().__init__()
        self.failure = failure

    async def call(self, name, arguments):
        self.executed.append((name, arguments))
        raise self.failure


def test_a_request_pubmed_rejects_is_passed_to_the_model_instead_of_failing():
    r, g, p = complete([call(), 'That article does not exist'], Rejecting(tool_failure('ncbi_resource_not_found')))
    assert r.choices[0].message.content.startswith('That article does not exist')
    assert r.tool_calls[0].error == 'PubMed returned a tool error'
    assert 'PMID 9 not found' in g.messages[1][-1]['content'] and 'Check the PMID' in g.messages[1][-1]['content']


@pytest.mark.parametrize('failure', [tool_failure('ncbi_unreachable', True), tool_failure('ncbi_rate_limited'),
                                     tool_failure('queue_full'), tool_failure(None),
                                     ServiceError('pubmed_tool_failed', 'x', 502, {'result': {'error': 'NCBI down'}})])
def test_pubmed_outages_end_the_answer_instead_of_reaching_the_model(failure):
    with pytest.raises(ServiceError) as e:
        complete([call(), 'an answer from memory'], Rejecting(failure))
    assert e.value.code == 'pubmed_unavailable'


def test_a_request_pubmed_keeps_rejecting_stops_after_two_attempts():
    with pytest.raises(ServiceError) as e:
        complete([call(), call(), call(), 'never'], Rejecting(tool_failure('invalid_date_range')))
    assert e.value.code == 'pubmed_tool_failed'
    assert len(e.value.details['tool_calls']) == 3


def test_a_tool_call_cut_off_by_the_length_limit_is_not_retried():
    from app.model_manager import GenerationResult
    with pytest.raises(ServiceError) as e:
        complete([GenerationResult('<tool_call>{"name": "pubmed_search_articles", "argu', 'length', 10, 5), call(), 'x'], Picky())
    assert e.value.code == 'tool_call_truncated'


def test_the_replayed_turn_has_no_end_of_turn_tokens():
    bad = call(arguments={'query': 'x', 'minDate': '2020'})
    r, g, p = complete([bad + '<|im_end|>\n<|endoftext|>', call(), 'Answer'], Picky())
    assert g.messages[1][-2] == {'role': 'assistant', 'content': bad}


def test_with_pubmed_off_a_malformed_call_gets_the_answer_directly_hint():
    r, g, p = complete(['<tool_call>{broken', 'Direct answer'], PubMed(fail=True), use_pubmed=False)
    assert r.choices[0].message.content == 'Direct answer'
    assert 'No tools are available' in json.loads(g.messages[1][-1]['content'])['hint']


def test_qwen35_text_values_for_typed_parameters_reach_the_repair():
    from app.chat_service import parse_tool_calls
    tools = [{'type': 'function', 'function': {'name': 'pubmed_fetch_articles', 'parameters': FETCH_SCHEMA}}]
    raw = '<tool_call><function=pubmed_fetch_articles><parameter=pmids>24393785, 37185680</parameter></function></tool_call>'
    call_, = parse_tool_calls(raw, tools)
    assert repaired(call_.arguments, FETCH_SCHEMA)[0] == {'pmids': ['24393785', '37185680']}


class Offline(PubMed):
    async def call(self, name, arguments):
        raise ServiceError('pubmed_unavailable', 'offline', 502)


def test_outages_still_end_the_answer():
    with pytest.raises(ServiceError) as e:
        complete([call(), 'never used'], Offline())
    assert e.value.code == 'pubmed_unavailable'


def test_streams_tell_the_client_to_discard_text_from_a_rejected_round():
    import threading
    from app.chat_service import ChatService
    from app.schemas import ChatRequest
    events = []
    request = ChatRequest(messages=[{'role': 'user', 'content': 'q'}], stream=True)
    asyncio.run(ChatService(Generator(['Let me search <tool_call>{broken', call(), 'Answer']), Picky())
                .stream(request, events.append, threading.Event()))
    kinds = [e['type'] for e in events]
    assert kinds.index('retry') < kinds.index('tool')
    assert events[-1]['type'] == 'done'
