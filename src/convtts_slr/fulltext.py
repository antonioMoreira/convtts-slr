"""Full-text acquisition and parsing.

Order of attempts: a PDF you placed manually in <workdir>/pdfs/<id>.pdf, then arXiv,
then the open-access URL from the source. Papers that fail are logged as
'unavailable' (never silently dropped) and show up in the human queue.

GROBID gives better structure; PyMuPDF is used here because it has no service to run.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx

from .models import Paper
from .store import safe_name

_REFS = re.compile(r"^\s*(references|bibliography|acknowledg(e)?ments?)\s*$", re.I | re.M)
_HEADING = re.compile(
    r"^\s*((\d+(\.\d+)*\.?)|([IVX]+\.))\s+[A-Z][A-Za-z0-9 ,:&/()-]{2,80}\s*$"
    r"|^\s*[A-Z][A-Z0-9 ,:&/-]{3,60}\s*$",
    re.M,
)
_PRIORITY = re.compile(
    r"data|corpus|corpora|collect|record|annotat|construct|pipeline|process|statistic|"
    r"baseline|model|architect|method|experiment|evaluat|release|licen|prosod|pitch|energy|duration",
    re.I,
)


def download_pdf(p: Paper, workdir: Path, client: httpx.Client | None = None) -> Path | None:
    manual = workdir / "pdfs" / f"{safe_name(p.id)}.pdf"
    if manual.exists():
        return manual
    client = client or httpx.Client(
        timeout=90,
        follow_redirects=True,
        headers={"User-Agent": "convtts-slr/0.1 (systematic review)"},
    )
    urls = ([f"https://arxiv.org/pdf/{p.arxiv_id}"] if p.arxiv_id else []) + (
        [p.pdf_url] if p.pdf_url else []
    )
    for url in urls:
        try:
            r = client.get(url)
        except httpx.HTTPError:
            continue
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            manual.parent.mkdir(parents=True, exist_ok=True)
            manual.write_bytes(r.content)
            return manual
    return None


def pdf_to_text(path: Path) -> str:
    import pymupdf

    with pymupdf.open(path) as doc:
        text = "\n".join(page.get_text("text") for page in doc)
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line breaks
    return re.sub(r"[ \t]+", " ", text)


def select_sections(text: str, budget: int) -> str:
    """Keep the opening (abstract + intro head) and the sections most likely to
    describe the data and the baseline model, in document order, within `budget` chars."""
    if len(text) <= budget:
        return text
    if (m := _REFS.search(text)) and m.start() > len(text) * 0.4:
        text = text[: m.start()]
        if len(text) <= budget:
            return text
    starts = [0] + [m.start() for m in _HEADING.finditer(text)] + [len(text)]
    sections = [text[a:b] for a, b in zip(starts, starts[1:], strict=False) if b > a]
    head, rest = sections[0][:3000], sections[1:]
    used = len(head)
    priority = [i for i, s in enumerate(rest) if _PRIORITY.search(s.split("\n", 1)[0])]
    others = [i for i in range(len(rest)) if i not in set(priority)]
    pieces: dict[int, str] = {}
    for i in priority + others:  # priority sections first, then fill remaining room
        room = budget - used
        if len(rest[i]) <= room:
            pieces[i] = rest[i]
        elif i in priority and room > 1500:  # a long key section: keep its beginning
            pieces[i] = rest[i][:room]
        else:
            continue
        used += len(pieces[i])
    return "\n".join([head] + [pieces[i] for i in sorted(pieces)])[:budget]
