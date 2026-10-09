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


def test_mps_load_stages_weights_on_cpu_before_move(monkeypatch):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    obj=manager(monkeypatch)
    obj.device='mps'
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
