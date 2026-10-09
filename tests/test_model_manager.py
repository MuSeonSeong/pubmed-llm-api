import asyncio
import threading
import weakref
import pytest

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
    asyncio.run(obj.generate('qwen3-0.6b',[],[],10,0,thinking=True))
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
    obj=manager(monkeypatch,loader=lambda spec:(Thinker(),Words()))
    result=asyncio.run(obj.generate('qwen3-0.6b',[{'role':'user','content':'hi'}],[],100,0,thinking=True,on_text=seen.append))
    assert calls[0] == (3, 75, 3)            # 75% of the answer budget for reasoning
    assert calls[1] == (3 + 75 + 2, 25, 80)  # prompt + reasoning + forced close, rest for the answer
    assert result.finish_reason == 'stop' and result.raw_text.endswith('answer')
    assert any('</think>' in text for text in seen)


def test_reasoning_that_finishes_in_budget_is_not_forced(monkeypatch):
    calls=[]
    class Quick(Model):
        def generate(self, **kw):
            calls.append(kw['max_new_tokens'])
            return super().generate(**kw)
    obj=manager(monkeypatch,loader=lambda spec:(Quick(),Tokenizer()))
    asyncio.run(obj.generate('qwen3-0.6b',[],[],100,0,thinking=True))
    assert calls == [75]


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
    obj=manager(monkeypatch,loader=lambda spec:(Answering(),Closed()))
    result=asyncio.run(obj.generate('qwen3-0.6b',[],[],100,0,thinking=True))
    assert calls == [(3, 75), (3 + 75, 25)]
    assert result.finish_reason == 'stop'
