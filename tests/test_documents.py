import pytest
from app.settings import ServiceError


def pdf_bytes(pages=('Hello PDF world',)):
    """A minimal valid PDF with one text line per page, built by hand to avoid a writer dependency."""
    objects = ['<< /Type /Catalog /Pages 2 0 R >>', None, '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>']
    kids = []
    for text in pages:
        stream = f'BT /F1 12 Tf 72 720 Td ({text}) Tj ET'
        objects.append(f'<< /Length {len(stream)} >>\nstream\n{stream}\nendstream')
        objects.append(f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> '
                       f'/Contents {len(objects)} 0 R >>')
        kids.append(f'{len(objects)} 0 R')
    objects[1] = f'<< /Type /Pages /Kids [{" ".join(kids)}] /Count {len(kids)} >>'
    out, offsets = b'%PDF-1.4\n', []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f'{number} 0 obj\n{body}\nendobj\n'.encode()
    xref = len(out)
    out += f'xref\n0 {len(objects) + 1}\n0000000000 65535 f \n'.encode()
    out += ''.join(f'{offset:010d} 00000 n \n' for offset in offsets).encode()
    out += f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()
    return out


def test_extracts_text_from_every_page():
    from app.documents import extract_pdf_text
    result = extract_pdf_text(pdf_bytes(('First page', 'Second page')))
    assert 'First page' in result['text'] and 'Second page' in result['text']
    assert (result['pages'], result['pages_read'], result['truncated']) == (2, 2, False)


def test_page_cap_marks_the_result_truncated(monkeypatch):
    import app.documents as documents
    monkeypatch.setattr(documents, 'MAX_PAGES', 1)
    result = documents.extract_pdf_text(pdf_bytes(('One', 'Two')))
    assert result['pages_read'] == 1 and result['truncated'] and 'Two' not in result['text']


@pytest.mark.parametrize('data', [b'not a pdf', b'%PDF-1.4\ngarbage'])
def test_rejects_non_pdf_and_broken_files(data):
    from app.documents import extract_pdf_text
    with pytest.raises(ServiceError) as e:
        extract_pdf_text(data)
    assert e.value.status_code == 422


def test_extract_endpoint_streams_the_body_and_limits_size(monkeypatch):
    import app.main
    from tests.test_api import client
    c, g, _ = client()
    with c:
        r = c.post('/v1/documents/extract', content=pdf_bytes(), headers={'Content-Type': 'application/pdf'})
        assert r.status_code == 200 and 'Hello PDF world' in r.json()['text']
        monkeypatch.setattr(app.main, 'MAX_PDF_BYTES', 10)
        r = c.post('/v1/documents/extract', content=pdf_bytes(), headers={'Content-Type': 'application/pdf'})
        assert r.status_code == 413 and r.json()['error']['code'] == 'document_too_large'
        assert g.messages == []


def test_hyphenated_line_breaks_are_joined(monkeypatch):
    import app.documents as documents
    class Page:
        def extract_text(self):
            return 'insulin sensitiv- \nity and HbA1c'
    class Reader:
        is_encrypted, pages, metadata = False, [Page()], None
        def __init__(self, stream):
            pass
    monkeypatch.setattr('pypdf.PdfReader', Reader)
    assert documents.extract_pdf_text(b'%PDF-1.4')['text'] == 'insulin sensitivity and HbA1c'
