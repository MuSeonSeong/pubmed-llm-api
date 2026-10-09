import asyncio
import threading
import weakref
import pytest
from app.model_manager import THINKING_BUDGETS

class Tokenizer:
    eos_token_id = 99
    def apply_chat_template(self, messages, **kw):
        import torch
        n = 20 if kw.get('tools') else 3
        return {'input_ids': torch.ones((1, n), dtype=torch.long), 'attention_mask': torch.ones((1, n), dtype=torch.long)}
    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return '<tool_call>{"name":"search","arguments":{}}</tool_call><|im_end|>'

class Model:
    def generate(self, **kw):
        import torch
        assert kw['do_sample'] is False
        return torch.cat((kw['input_ids'], torch.tensor([[42, 99]])), dim=1)


def manager(monkeypatch, *, budget=4096, loader=None):
    from app.model_manager import ModelManager
    from app.catalog import load_catalog
    from app.settings import Settings
    obj = ModelManager(Settings(device='cpu', max_context_tokens=budget), load_catalog())
    monkeypatch.setattr(obj, '_load', loader or (lambda spec: (Model(), Tokenizer())))
    return obj

@pytest.mark.parametrize('cuda,mps,expected', [(True,True,'cuda'),(False,True,'mps'),(False,False,'cpu')])
def test_auto_device_selection(cuda,mps,expected):
    from app.model_manager import select_device
    assert select_device('auto',cuda_available=cuda,mps_available=mps) == expected

@pytest.mark.parametrize('device',['cuda','mps'])
def test_explicit_unavailable_device_fails(device):
    from app.model_manager import select_device
    from app.settings import ServiceError
    with pytest.raises(ServiceError) as e:
        select_device(device,cuda_available=False,mps_available=False)
    assert e.value.status_code == 503


def test_context_counts_template_and_tools(monkeypatch):
    from app.settings import ServiceError
    obj=manager(monkeypatch,budget=22)
    args=('qwen3-0.6b',[{'role':'user','content':'hello'}],[{'type':'function'}])
    result=asyncio.run(obj.generate(*args,2,0))
    assert result.prompt_tokens == 20
    with pytest.raises(ServiceError) as e:
        asyncio.run(obj.generate(*args,3,0))
    assert e.value.status_code == 413


def test_raw_generation_preserves_tool_markers(monkeypatch):
    obj=manager(monkeypatch)
    result=asyncio.run(obj.generate('qwen3-0.6b',[{'role':'user','content':'hi'}],[],10,0))
    assert result.raw_text.startswith('<tool_call>')
    assert result.finish_reason == 'stop'
    assert result.completion_tokens == 2


def test_model_switch_releases_previous_model(monkeypatch):
    refs=[]
    def load(spec):
        if refs:
            assert refs[-1]() is None
        model=Model()
        refs.append(weakref.ref(model))
        return model,Tokenizer()
    obj=manager(monkeypatch,loader=load)
    async def run():
        for name in ['qwen3-0.6b','qwen3.5-2b']:
            await obj.generate(name,[{'role':'user','content':'hi'}],[],10,0)
    asyncio.run(run())
    assert obj.status()['loaded_model'] == 'qwen3.5-2b'
    obj.close()
    assert refs[-1]() is None


def test_cancelled_request_keeps_generation_serialized(monkeypatch):
    entered,release=threading.Event(),threading.Event()
    order=[]
    class Blocking(Model):
        def generate(self,**kw):
            order.append('first-start')
            entered.set()
            assert release.wait(3)
            order.append('first-end')
            return super().generate(**kw)
    def load(spec):
        order.append(spec.id)
        return (Blocking() if spec.id == 'qwen3-0.6b' else Model()),Tokenizer()
    obj=manager(monkeypatch,loader=load)
    async def run():
        a=asyncio.create_task(obj.generate('qwen3-0.6b',[],[],10,0))
        assert await asyncio.to_thread(entered.wait,2)
        a.cancel()
        b=asyncio.create_task(obj.generate('qwen3.5-2b',[],[],10,0))
        try:
            await asyncio.sleep(.05)
            assert 'qwen3.5-2b' not in order
        finally:
            release.set()
        await asyncio.gather(a,b,return_exceptions=True)
    asyncio.run(run())
    assert order.index('first-end') < order.index('qwen3.5-2b')


def test_mps_load_stages_weights_on_cpu_before_move(monkeypatch, tmp_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    obj=manager(monkeypatch)
    obj.device='mps'
    obj.settings.models_dir = tmp_path
    folder = tmp_path / 'qwen3-0.6b'
    folder.mkdir()
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json']:
        (folder / name).write_text('{}')
    (folder / 'model.safetensors').write_bytes(b'test fixture')
    calls=[]
    class Loaded:
        def to(self,device):
            calls.append(('move',device))
            return self
        def eval(self):
            return self
    def from_pretrained(path,**kw):
        calls.append(('load',kw['device_map']))
        return Loaded()
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**kw:Tokenizer())
    monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',from_pretrained)
    # Exercise the production loader, not the slow-load double used in other tests.
    from app.model_manager import ModelManager
    ModelManager._load(obj,obj.catalog['qwen3-0.6b'])
    assert calls == [('load','cpu'),('move','mps')]


def test_config_without_weights_is_not_ready(tmp_path):
    from app.model_manager import ModelManager
    from app.catalog import load_catalog
    from app.settings import Settings
    path=tmp_path/'qwen3-0.6b'
    path.mkdir()
    (path/'config.json').write_text('{}')
    m=ModelManager(Settings(device='cpu',models_dir=tmp_path),load_catalog())
    assert m.status()['model_files']['qwen3-0.6b'] is False


def test_sharded_weights_without_index_are_not_ready(tmp_path):
    from app.model_manager import ModelManager
    from app.catalog import load_catalog
    from app.settings import Settings
    p=tmp_path/'qwen3.5-2b'
    p.mkdir()
    for name in ['config.json','tokenizer_config.json','tokenizer.json']:
        (p/name).write_text('{}')
    (p/'model.safetensors-00001-of-00001.safetensors').write_bytes(b'weight')
    manager=ModelManager(Settings(device='cpu',models_dir=tmp_path),load_catalog())
    assert manager.status()['model_files']['qwen3.5-2b'] is False

@pytest.mark.parametrize('index',[[],{'weight_map':{}},{'weight_map':{'w':{'invalid':1}}}])
def test_invalid_shard_index_is_not_ready(tmp_path,index):
    import json
    from app.model_manager import ModelManager
    from app.catalog import load_catalog
    from app.settings import Settings
    p=tmp_path/'qwen3.5-2b'
    p.mkdir()
    for name in ['config.json','tokenizer_config.json','tokenizer.json']:
        (p/name).write_text('{}')
    (p/'model.safetensors.index.json').write_text(json.dumps(index))
    manager=ModelManager(Settings(device='cpu',models_dir=tmp_path),load_catalog())
    assert manager.status()['model_files']['qwen3.5-2b'] is False


class StreamingModel(Model):
    def generate(self, **kw):
        import torch
        kw['streamer'].put(kw['input_ids'])
        for token in (42, 99):
            kw['streamer'].put(torch.tensor([token]))
        kw['streamer'].end()
        return super().generate(**kw)


def test_streamer_reports_cumulative_completion_text(monkeypatch):
    obj=manager(monkeypatch,loader=lambda spec:(StreamingModel(),Tokenizer()))
    seen=[]
    asyncio.run(obj.generate('qwen3-0.6b',[{'role':'user','content':'hi'}],[],10,0,on_text=seen.append))
    assert seen[0] == ''  # generation started
    assert len(seen) == 3
    assert seen[-1].startswith('<tool_call>')


def test_cancel_before_start_skips_generation(monkeypatch):
    class Never(Model):
        def generate(self, **kw):
            raise AssertionError('generation should not start')
    obj=manager(monkeypatch,loader=lambda spec:(Never(),Tokenizer()))
    cancel=threading.Event()
    cancel.set()
    result=asyncio.run(obj.generate('qwen3-0.6b',[],[],10,0,cancel=cancel))
    assert result.finish_reason == 'cancelled'


def test_cancel_criteria_stops_running_generation():
    import torch
    from app.model_manager import CancelCriteria
    cancel=threading.Event()
    criteria=CancelCriteria(cancel)
    ids=torch.ones((1,4),dtype=torch.long)
    assert not criteria(ids,None).any()
    cancel.set()
    assert criteria(ids,None).all()


def test_streamer_waits_for_complete_multibyte_characters(monkeypatch):
    import torch
    from app.model_manager import TextCallbackStreamer
    class Partial:
        def decode(self, ids, skip_special_tokens=False):
            return '안\ufffd' if len(ids) == 1 else '안녕'
    seen=[]
    streamer=TextCallbackStreamer(Partial(),seen.append)
    streamer.put(torch.ones((1,3)))
    streamer.put(torch.tensor([1]))
    streamer.put(torch.tensor([2]))
    assert seen == ['', '안녕']


def test_thinking_flag_reaches_the_chat_template(monkeypatch):
    seen=[]
    class Recording(Tokenizer):
        def apply_chat_template(self, messages, **kw):
            seen.append(kw['enable_thinking'])
            return super().apply_chat_template(messages, **kw)
    obj=manager(monkeypatch,loader=lambda spec:(Model(),Recording()))
    asyncio.run(obj.generate('qwen3-0.6b',[],[],10,0))
    asyncio.run(obj.generate('qwen3-0.6b',[],[],10,0,effort='medium'))
    assert seen == [False, True]


def test_status_reports_whether_generation_is_running(monkeypatch):
    obj=manager(monkeypatch)
    assert obj.status()['busy'] is False
    with obj._lock:
        assert obj.status()['busy'] is True


def test_continue_final_uses_the_open_assistant_template(monkeypatch):
    seen=[]
    class Recording(Tokenizer):
        def apply_chat_template(self, messages, **kw):
            seen.append((kw['add_generation_prompt'], kw.get('continue_final_message', False)))
            return super().apply_chat_template(messages, **kw)
    obj=manager(monkeypatch,loader=lambda spec:(Model(),Recording()))
    asyncio.run(obj.generate('qwen3-0.6b',[],[],10,0))
    asyncio.run(obj.generate('qwen3-0.6b',[],[],10,0,continue_final=True))
    assert seen == [(True, False), (False, True)]


def test_answer_length_shrinks_to_the_room_the_prompt_leaves(monkeypatch):
    seen=[]
    class Recording(Model):
        def generate(self, **kw):
            seen.append(kw['max_new_tokens'])
            return super().generate(**kw)
    obj=manager(monkeypatch,budget=120,loader=lambda spec:(Recording(),Tokenizer()))
    asyncio.run(obj.generate('qwen3-0.6b',[{'role':'user','content':'hi'}],[],500,0))
    assert seen == [117]  # 120 budget - 3 prompt tokens
    assert obj.status()['max_context_tokens'] == 120


def test_thinking_budget_closes_runaway_reasoning_and_lets_the_model_answer(monkeypatch):
    import torch
    calls=[]
    class Thinker(Model):
        def generate(self, **kw):
            calls.append((kw['input_ids'].shape[-1], kw['max_new_tokens'], kw['attention_mask'].shape[-1]))
            new = [7]*kw['max_new_tokens'] if len(calls) == 1 else [42, 99]  # think forever, then answer
            return torch.cat((kw['input_ids'], torch.tensor([new])), dim=1)
    class Words(Tokenizer):
        def __call__(self, text, add_special_tokens=False):
            assert '</think>' in text
            return {'input_ids': [5, 6]}
        def decode(self, ids, skip_special_tokens=False):
            return 'reasoning' if 5 not in ids else 'reasoning</think>answer'
    seen=[]
    monkeypatch.setitem(THINKING_BUDGETS,'medium',75)
    obj=manager(monkeypatch,loader=lambda spec:(Thinker(),Words()))
    result=asyncio.run(obj.generate('qwen3-0.6b',[{'role':'user','content':'hi'}],[],100,0,effort='medium',on_text=seen.append))
    assert calls[0] == (3, 75, 3)            # the effort's reasoning budget
    assert calls[1] == (3 + 75 + 2, 98, 80)  # prompt + reasoning + forced close; the close counts toward 75 + 100
    assert result.finish_reason == 'stop' and result.raw_text.endswith('answer')
    assert any('</think>' in text for text in seen)


def test_reasoning_that_finishes_in_budget_is_not_forced(monkeypatch):
    calls=[]
    class Quick(Model):
        def generate(self, **kw):
            calls.append(kw['max_new_tokens'])
            return super().generate(**kw)
    obj=manager(monkeypatch,loader=lambda spec:(Quick(),Tokenizer()))
    asyncio.run(obj.generate('qwen3-0.6b',[],[],100,0,effort='light'))
    assert calls == [512]


def test_an_answer_in_progress_at_the_budget_continues_without_forcing(monkeypatch):
    import torch
    calls=[]
    class Answering(Model):
        def generate(self, **kw):
            calls.append((kw['input_ids'].shape[-1], kw['max_new_tokens']))
            new = [8]*kw['max_new_tokens'] if len(calls) == 1 else [42, 99]
            return torch.cat((kw['input_ids'], torch.tensor([new])), dim=1)
    class Closed(Tokenizer):
        def __call__(self, text, add_special_tokens=False):
            raise AssertionError('no closing words once reasoning has ended')
        def decode(self, ids, skip_special_tokens=False):
            return 'r</think>answer so far'
    monkeypatch.setitem(THINKING_BUDGETS,'high',75)
    obj=manager(monkeypatch,loader=lambda spec:(Answering(),Closed()))
    result=asyncio.run(obj.generate('qwen3-0.6b',[],[],100,0,effort='high'))
    assert calls == [(3, 75), (3 + 75, 100)]
    assert result.finish_reason == 'stop'


def test_strict_budget_refuses_instead_of_shrinking(monkeypatch):
    from app.settings import ServiceError
    obj=manager(monkeypatch,budget=120)
    with pytest.raises(ServiceError) as e:
        asyncio.run(obj.generate('qwen3-0.6b',[{'role':'user','content':'hi'}],[],500,0,strict_budget=True))
    assert e.value.status_code == 413


def test_reasoning_budget_is_capped_by_the_room_the_answer_leaves(monkeypatch):
    calls=[]
    class Quick(Model):
        def generate(self, **kw):
            calls.append(kw['max_new_tokens'])
            return super().generate(**kw)
    obj=manager(monkeypatch,budget=300,loader=lambda spec:(Quick(),Tokenizer()))
    asyncio.run(obj.generate('qwen3-0.6b',[],[],100,0,effort='max'))
    assert calls == [300 - 3 - 100]  # the answer keeps its 100 tokens; reasoning gets the rest, not 8192


def test_without_room_to_reason_the_model_answers_directly(monkeypatch):
    seen,calls=[],[]
    class Recording(Tokenizer):
        def apply_chat_template(self, messages, **kw):
            seen.append(kw['enable_thinking'])
            return super().apply_chat_template(messages, **kw)
    class Quick(Model):
        def generate(self, **kw):
            calls.append(kw['max_new_tokens'])
            return super().generate(**kw)
    obj=manager(monkeypatch,budget=150,loader=lambda spec:(Quick(),Recording()))
    result=asyncio.run(obj.generate('qwen3-0.6b',[],[],100,0,effort='medium'))
    assert seen == [True, False] and calls == [100]  # 47 tokens would be too few to reason with
    assert result.raw_text.startswith('<think></think><tool_call>')  # an empty reasoning keeps the reply's shape


def test_effort_is_ignored_for_models_without_reasoning(monkeypatch):
    import dataclasses
    seen=[]
    class Recording(Tokenizer):
        def apply_chat_template(self, messages, **kw):
            seen.append(kw['enable_thinking'])
            return super().apply_chat_template(messages, **kw)
    obj=manager(monkeypatch,loader=lambda spec:(Model(),Recording()))
    obj.catalog={**obj.catalog,'qwen3-0.6b':dataclasses.replace(obj.catalog['qwen3-0.6b'],reasoning=False)}
    asyncio.run(obj.generate('qwen3-0.6b',[],[],10,0,effort='max'))
    assert seen == [False]


def test_temperature_left_to_the_model_is_the_old_default(monkeypatch):
    import torch
    seen=[]
    class Sampling(Model):
        def generate(self, **kw):
            seen.append((kw['do_sample'], kw.get('temperature')))
            return torch.cat((kw['input_ids'], torch.tensor([[42, 99]])), dim=1)
    obj=manager(monkeypatch,loader=lambda spec:(Sampling(),Tokenizer()))
    asyncio.run(obj.generate('qwen3-0.6b',[],[],10,None))
    assert seen == [(True, 0.2)]


def test_plan_output_keeps_the_answer_and_gives_reasoning_the_rest():
    from app.model_manager import plan_output
    from app.settings import ServiceError
    assert plan_output(32768, 2000, 1024, 2048, strict=True) == (1024, 2048)
    assert plan_output(4000, 2000, 1024, 8192, strict=True) == (1024, 976)
    assert plan_output(3060, 2000, 1024, 8192, strict=False) == (1024, 0)   # 36 tokens left: no reasoning
    assert plan_output(2500, 2000, 1024, 2048, strict=False) == (500, 0)    # the answer shrinks to the room left
    for limit, strict in [(2900, True), (2050, False)]:
        with pytest.raises(ServiceError) as e:
            plan_output(limit, 2000, 1024, 0, strict=strict)
        assert e.value.code == 'context_overflow' and e.value.status_code == 413


def test_status_lists_every_enabled_model_with_its_state(tmp_path):
    from app.model_manager import ModelManager
    from app.catalog import load_catalog
    from app.settings import Settings
    folder=tmp_path/'qwen3-0.6b'
    folder.mkdir()
    for name in ['config.json','tokenizer_config.json','tokenizer.json']:
        (folder/name).write_text('{}')
    (folder/'model.safetensors').write_bytes(b'weights')
    m=ModelManager(Settings(device='cpu',models_dir=tmp_path,models='qwen3-0.6b,qwen3.5-2b'),load_catalog())
    assert m.default_model == 'qwen3-0.6b' and m.pool is None
    status=m.status()
    assert {'device','loaded_model','busy','max_context_tokens','model_files'} <= set(status)
    assert status['models'] == {
        'qwen3-0.6b':{'state':'ready','runtime':'transformers','label':'Qwen3 0.6B','error':None},
        'qwen3.5-2b':{'state':'missing','runtime':'transformers','label':'Qwen3.5 2B','error':None}}


def test_mlx_models_are_served_by_the_pool(monkeypatch):
    from app.model_manager import ModelManager, GenerationResult
    from app.catalog import load_catalog
    from app.settings import Settings
    monkeypatch.setattr('app.settings.mlx_available', lambda: True)  # MODELS may name MLX models on any machine here
    class Pool:
        def __init__(self): self.calls=[]
        def __contains__(self, model_id): return model_id == 'qwen3.5-9b'
        def status(self): return {'qwen3.5-9b':{'state':'starting','runtime':'mlx','label':'Qwen3.5 9B','error':None}}
        async def generate(self, spec, *args, **kw):
            self.calls.append((spec.id, args, kw))
            return GenerationResult('hi','stop',1,1)
        def close(self): self.closed=True
    pool=Pool()
    m=ModelManager(Settings(device='cpu',models='qwen3.5-9b,qwen3-0.6b'),load_catalog(),pool=pool)
    result=asyncio.run(m.generate('qwen3.5-9b',[{'role':'user','content':'q'}],[],100,None,effort='high',tenant='user-1'))
    assert result.raw_text == 'hi'
    spec_id,args,kw=pool.calls[0]
    assert spec_id == 'qwen3.5-9b' and args == ([{'role':'user','content':'q'}],[],100,None)
    assert kw['effort'] == 'high' and kw['tenant'] == 'user-1'
    assert m.status()['models']['qwen3.5-9b']['state'] == 'starting'
    m.close()
    assert pool.closed


def test_the_manager_hands_mlx_starts_to_its_pool_for_the_apps_lifetime(tmp_path, monkeypatch):
    from app.model_manager import ModelManager
    from app.catalog import load_catalog
    from app.settings import Settings
    monkeypatch.setattr('app.settings.mlx_available', lambda: True)
    settings=Settings(device='cpu',models_dir=tmp_path,models='qwen3.5-4b,qwen3-0.6b,qwen3.5-9b')
    m=ModelManager(settings,load_catalog())
    assert list(m.pool.sidecars) == ['qwen3.5-4b','qwen3.5-9b']
    calls=[]
    async def run(prefix, preload_ids):
        calls.append((prefix, preload_ids))
    m.pool.run=run
    asyncio.run(m.start('prefix'))
    assert calls == [('prefix', ['qwen3.5-4b'])]  # PRELOAD_MODELS=default: the 9B starts on its first question


@pytest.mark.parametrize('models,mode,preloaded', [
    ('qwen3.5-9b,qwen3.5-4b', 'default', ['qwen3.5-9b']),
    ('qwen3.5-9b,qwen3.5-4b', 'all', ['qwen3.5-9b','qwen3.5-4b']),
    ('qwen3.5-9b,qwen3.5-4b', 'none', []),
    ('qwen3.5-4b,qwen3.5-9b', 'default', ['qwen3.5-4b']),            # the first in MODELS is the default
    ('qwen3-0.6b,qwen3.5-4b,qwen3.5-9b', 'default', []),             # a Transformers default loads on first use
    ('qwen3-0.6b,qwen3.5-4b,qwen3.5-9b', 'all', ['qwen3.5-4b','qwen3.5-9b'])])
def test_preload_models_names_the_mlx_models_started_with_the_app(tmp_path, monkeypatch, models, mode, preloaded):
    from app.model_manager import ModelManager
    from app.catalog import load_catalog
    from app.settings import Settings
    monkeypatch.setattr('app.settings.mlx_available', lambda: True)
    m=ModelManager(Settings(device='cpu',models_dir=tmp_path,models=models,preload_models=mode),load_catalog())
    assert m.preload_ids() == preloaded
