"""Decode user images for vision models, refusing anything but a modest raster image in a data URL."""
import base64
import binascii
import io
import re
from app.settings import ServiceError

MAX_BYTES = 8 * 1024 * 1024
MAX_DECODED_PIXELS = 40_000_000
MAX_ASPECT_RATIO = 150  # the Qwen2-VL processor refuses ratios of 200 or more
FORMATS = ['PNG', 'JPEG', 'WEBP', 'GIF']  # never let Pillow pick other decoders (EPS would run Ghostscript)
# The JPEG decoder reports multi-picture camera files as MPO; their first picture is a plain JPEG.
MIME_TYPES = {'PNG': 'image/png', 'JPEG': 'image/jpeg', 'MPO': 'image/jpeg', 'WEBP': 'image/webp', 'GIF': 'image/gif'}
# Resize target for the model: about 256 image tokens on Qwen3.5 (16 px patches merged 2x2),
# small enough to leave room for the conversation in a 4k context.
MODEL_MAX_PIXELS = 512 * 512
IMAGE_TOKEN = '<|image_pad|>'
# Vision control tokens typed by a user would be parsed as real placeholders, so text never carries them.
VISION_TOKENS = ('<|image_pad|>', '<|video_pad|>', '<|vision_start|>', '<|vision_end|>')
DATA_URL = re.compile(r'data:image/(?:png|jpeg|jpg|webp|gif);base64,([A-Za-z0-9+/=\s]+)', re.S)


def invalid(message: str) -> ServiceError:
    return ServiceError('invalid_image', message, 422)


def _decode(data: bytes, convert: bool):
    """Fully decode an allowed image; returns (RGB image or None, MIME type sniffed from the bytes)."""
    from PIL import Image, UnidentifiedImageError
    if len(data) > MAX_BYTES:
        raise invalid('Images must be 8 MB or smaller')
    try:
        with Image.open(io.BytesIO(data), formats=FORMATS) as image:
            mime = MIME_TYPES.get(image.format)
            if mime is None:
                raise invalid('Images must be PNG, JPEG, WebP or GIF')
            if image.width * image.height > MAX_DECODED_PIXELS:
                raise invalid('Image dimensions are too large')
            if max(image.size) > MAX_ASPECT_RATIO * min(image.size):
                raise invalid('Image is too narrow or too wide')
            if convert:
                return image.convert('RGB'), mime  # GIFs use their first frame
            image.load()  # a header can be intact while the pixel data is truncated
            return None, mime
    except ServiceError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise invalid('Image could not be decoded') from exc


def validate_image_bytes(data: bytes) -> str:
    """The MIME type of an uploaded image, sniffed from its bytes; whatever the client claimed is ignored."""
    return _decode(data, convert=False)[1]


def decode_image(url: str):
    match = DATA_URL.fullmatch(url.strip())
    if not match:
        raise invalid('Images must be base64 data URLs of PNG, JPEG, WebP or GIF')
    try:
        data = base64.b64decode(re.sub(r'\s+', '', match.group(1)), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise invalid('Image data is not valid base64') from exc
    return _decode(data, convert=True)[0]
