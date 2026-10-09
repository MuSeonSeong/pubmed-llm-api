"""Plain-text extraction from uploaded PDFs, bounded so a large or hostile file cannot stall the server."""
import io
import re
from app.settings import ServiceError

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PAGES = 60
MAX_CHARS = 200_000


def extract_pdf_text(data: bytes) -> dict:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
    if len(data) > MAX_PDF_BYTES:
        raise ServiceError('document_too_large', 'PDF files must be 20 MB or smaller', 413)
    if not data.startswith(b'%PDF-'):
        raise ServiceError('invalid_document', 'Only PDF files can be extracted', 422)
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ServiceError('invalid_document', 'Password-protected PDFs are not supported', 422)
        pages, texts, total = len(reader.pages), [], 0
        for page in reader.pages[:MAX_PAGES]:
            text = page.extract_text() or ''
            texts.append(text)
            total += len(text)
            if total >= MAX_CHARS:
                break
    except ServiceError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise ServiceError('invalid_document', 'The PDF could not be read', 422) from exc
    # Join hyphenated line breaks and collapse the whitespace PDFs scatter between words.
    text = '\n\n'.join(texts).replace('\x00', '')
    text = re.sub(r'-[ \t]*\n(?=[a-z])', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    title = (reader.metadata.title if reader.metadata else None) or None
    return {'text': text[:MAX_CHARS], 'pages': pages, 'pages_read': len(texts),
            'truncated': len(text) > MAX_CHARS or len(texts) < pages, 'title': title}
