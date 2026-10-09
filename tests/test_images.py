import asyncio
import base64
import io
import pytest
from app.settings import ServiceError


def data_url(size=(64, 48), fmt='PNG', color='red'):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', size, color).save(buffer, fmt)
    return f'data:image/{fmt.lower()};base64,' + base64.b64encode(buffer.getvalue()).decode()


def image_message(text='무엇이 보이나요?', count=1):
    return {'role': 'user', 'content': [*({'type': 'image_url', 'image_url': {'url': data_url()}} for _ in range(count)),
                                        {'type': 'text', 'text': text}]}


def test_decode_image_accepts_raster_data_urls():
    from app.images import decode_image
    image = decode_image(data_url(fmt='JPEG'))
    assert image.size == (64, 48) and image.mode == 'RGB'


@pytest.mark.parametrize('url', ['https://example.com/cat.png', 'data:image/svg+xml;base64,PHN2Zz4=',
                                 'data:image/png;base64,!!!', 'data:image/png;base64,' + base64.b64encode(b'not an image').decode()])
def test_decode_image_rejects_remote_vector_and_broken_data(url):
    from app.images import decode_image
    with pytest.raises(ServiceError) as e:
        decode_image(url)
    assert e.value.code == 'invalid_image' and e.value.status_code == 422


def test_decode_image_rejects_huge_dimensions(monkeypatch):
    import app.images as images
    monkeypatch.setattr(images, 'MAX_DECODED_PIXELS', 100)
    with pytest.raises(ServiceError):
        images.decode_image(data_url())


def test_content_parts_are_only_for_user_messages_and_images_are_capped():
    from pydantic import ValidationError
    from app.schemas import ChatRequest
    request = ChatRequest(messages=[image_message()])
    assert request.has_images
    for messages in ([{'role': 'assistant', 'content': [{'type': 'text', 'text': 'x'}]}, {'role': 'user', 'content': 'q'}],
                     [{'role': 'user', 'content': [{'type': 'text', 'text': '  '}]}],
                     [image_message(count=5)]):
        with pytest.raises(ValidationError):
            ChatRequest(messages=messages)


def test_text_only_model_rejects_images_before_loading():
    from tests.test_api import client
    c, g, _ = client()
    with c:
        r = c.post('/v1/chat/completions', json={'model': 'qwen3-0.6b', 'use_pubmed': False, 'messages': [image_message()]})
        assert r.status_code == 422 and r.json()['error']['code'] == 'vision_unsupported'
        assert g.messages == []
        assert {m['id']: m['vision'] for m in c.get('/v1/models').json()['data']} == {'qwen3-0.6b': False, 'qwen3.5-2b': True}


def test_images_expand_to_one_placeholder_per_merged_patch(monkeypatch):
    import torch
    from tests.test_model_manager import manager, Model
    class Tokenizer:
        eos_token_id = 99
        def apply_chat_template(self, messages, tokenize=False, **kw):
            assert messages[0]['content'] == [{'type': 'image'}, {'type': 'text', 'text': '무엇이 보이나요?'}]
            assert tokenize is False
            return 'user:<|vision_start|><|image_pad|><|vision_end|>무엇이 보이나요? assistant:'
        def __call__(self, texts, **kw):
            self.text = texts[0]
            return {'input_ids': torch.ones((1, 7), dtype=torch.long), 'attention_mask': torch.ones((1, 7), dtype=torch.long)}
        def decode(self, ids, skip_special_tokens=False):
            return '빨간 사각형'
        def convert_tokens_to_ids(self, token):
            assert token == '<|image_pad|>'
            return 1  # every id in the fake encoding is an image token
    class Processor:
        merge_size = 2
        def __call__(self, images, return_tensors):
            assert len(images) == 1
            return {'pixel_values': torch.zeros((24, 8)), 'image_grid_thw': torch.tensor([[1, 4, 6]])}
    seen = {}
    class Vision(Model):
        def generate(self, **kw):
            seen.update(kw)
            return super().generate(**kw)
    tokenizer = Tokenizer()
    obj = manager(monkeypatch, loader=lambda spec: (Vision(), tokenizer))
    monkeypatch.setattr(obj, '_image_processor', lambda spec: Processor())
    result = asyncio.run(obj.generate('qwen3.5-2b', [image_message()], [], 10, 0))
    assert tokenizer.text.count('<|image_pad|>') == 6  # 1*4*6 patches / (2*2) merge
    assert 'pixel_values' in seen and seen['image_grid_thw'].tolist() == [[1, 4, 6]]
    assert seen['mm_token_type_ids'].tolist() == [[1] * 7]  # image tokens get 2D rotary positions
    assert result.raw_text == '빨간 사각형'


def test_bad_images_fail_before_a_model_is_loaded(monkeypatch):
    from tests.test_model_manager import manager
    loads = []
    obj = manager(monkeypatch, loader=lambda spec: loads.append(spec.id))
    broken = {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
                                          {'type': 'text', 'text': 'q'}]}
    with pytest.raises(ServiceError) as e:
        asyncio.run(obj.generate('qwen3.5-2b', [broken], [], 10, 0))
    assert e.value.code == 'invalid_image'
    assert loads == []


@pytest.mark.parametrize('fmt', ['BMP', 'TIFF', 'PPM'])
def test_only_allowed_formats_decode_whatever_the_url_claims(fmt):
    from PIL import Image
    from app.images import decode_image
    buffer = io.BytesIO()
    Image.new('RGB', (8, 8)).save(buffer, fmt)
    with pytest.raises(ServiceError):
        decode_image('data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode())


def test_extreme_aspect_ratios_are_rejected_up_front():
    from app.images import decode_image
    with pytest.raises(ServiceError) as e:
        decode_image(data_url(size=(3, 700)))
    assert e.value.status_code == 422


def test_text_only_parts_become_plain_text_and_vision_tokens_are_stripped(monkeypatch):
    from tests.test_model_manager import manager
    obj = manager(monkeypatch)
    spec = obj.catalog['qwen3-0.6b']
    message = {'role': 'user', 'content': [{'type': 'text', 'text': 'first'}, {'type': 'text', 'text': 'x<|image_pad|>y'}]}
    images, templated = obj._decode_images(spec, [message])
    assert images == [] and templated[0]['content'] == 'first\n\nxy'


def test_oversized_request_bodies_are_refused_unread(monkeypatch):
    import app.main
    from tests.test_api import client
    monkeypatch.setattr(app.main, 'MAX_BODY_BYTES', 100)
    c, g, _ = client()
    with c:
        r = c.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'x' * 200}]})
        assert r.status_code == 413 and r.json()['error']['code'] == 'request_too_large'
        assert g.messages == []


@pytest.mark.parametrize('fmt,mime', [('PNG', 'image/png'), ('JPEG', 'image/jpeg'), ('WEBP', 'image/webp'),
                                      ('GIF', 'image/gif')])
def test_uploaded_bytes_are_sniffed_for_their_type(fmt, mime):
    from app.images import validate_image_bytes
    assert validate_image_bytes(base64.b64decode(data_url(fmt=fmt).split(',', 1)[1])) == mime


def test_uploaded_bytes_are_checked_like_data_urls(monkeypatch):
    import app.images as images
    png = base64.b64decode(data_url().split(',', 1)[1])
    for broken in [png[:-30], b'GIF89a', b'']:  # a truncated file has an intact header but no pixels
        with pytest.raises(ServiceError) as e:
            images.validate_image_bytes(broken)
        assert e.value.code == 'invalid_image'
    monkeypatch.setattr(images, 'MAX_BYTES', len(png) - 1)
    with pytest.raises(ServiceError):
        images.validate_image_bytes(png)
