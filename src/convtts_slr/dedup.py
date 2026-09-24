"""EC4: deterministic deduplication. arXiv preprints and their venue versions are
merged into one record (matching on DOI, arXiv id, or near-identical title)."""

import hashlib
import re
import unicodedata

from rapidfuzz import fuzz, process

from .models import Paper

_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.I)
_ARXIV_VERSION = re.compile(r"v\d+$")


def norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    d = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", d)
    return d or None


def norm_arxiv(aid: str | None) -> str | None:
    if not aid:
        return None
    a = aid.strip().lower()
    a = re.sub(r"^(arxiv:|https?://arxiv\.org/(abs|pdf)/)", "", a).removesuffix(".pdf")
    return _ARXIV_VERSION.sub("", a) or None


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"\s+", " ", t)).strip()


def normalize(p: Paper) -> Paper:
    doi = norm_doi(p.doi)
    arxiv = norm_arxiv(p.arxiv_id)
    if doi and (m := _ARXIV_DOI.match(doi)):
        arxiv = arxiv or norm_arxiv(m.group(1))
        doi = None  # an arXiv DOI is not a venue DOI
    return p.model_copy(update={"doi": doi, "arxiv_id": arxiv})


def canonical_id(p: Paper) -> str:
    if p.arxiv_id:
        return f"arxiv:{p.arxiv_id}"
    if p.doi:
        return f"doi:{p.doi}"
    return "title:" + hashlib.sha1(norm_title(p.title).encode()).hexdigest()[:12]


def same_work(a: Paper, b: Paper, title_threshold: float = 95.0) -> bool:
    if a.doi and a.doi == b.doi:
        return True
    if a.arxiv_id and a.arxiv_id == b.arxiv_id:
        return True
    if a.year and b.year and abs(a.year - b.year) > 1:
        return False
    ta, tb = norm_title(a.title), norm_title(b.title)
    return len(ta) > 15 and fuzz.ratio(ta, tb) >= title_threshold


def merge(keep: Paper, other: Paper) -> Paper:
    """Fill gaps in `keep` from `other`; ids, role and seed flag are never downgraded."""
    upd = {}
    for f in (
        "abstract",
        "year",
        "publication_date",
        "venue",
        "doi",
        "arxiv_id",
        "openalex_id",
        "s2_id",
        "pdf_url",
    ):
        if not getattr(keep, f) and getattr(other, f):
            upd[f] = getattr(other, f)
    if len(other.abstract or "") > len(keep.abstract or "") and "abstract" not in upd:
        upd["abstract"] = other.abstract
    upd["sources"] = sorted(set(keep.sources) | set(other.sources))
    upd["is_seed"] = keep.is_seed or other.is_seed
    upd["found_in_round"] = min(keep.found_in_round, other.found_in_round)
    if other.role != keep.role and other.is_seed:
        upd["role"] = other.role
    return keep.model_copy(update=upd)


class _DuplicateIndex:
    """Incremental exact-id + fuzzy-title lookup over a growing pool of papers."""

    def __init__(self, pool: dict[str, Paper]):
        self.pool = pool
        self.by_doi = {p.doi: p.id for p in pool.values() if p.doi}
        self.by_arxiv = {p.arxiv_id: p.id for p in pool.values() if p.arxiv_id}
        self.titles = {p.id: norm_title(p.title) for p in pool.values()}

    def find(self, p: Paper) -> Paper | None:
        pid = (p.doi and self.by_doi.get(p.doi)) or (p.arxiv_id and self.by_arxiv.get(p.arxiv_id))
        if pid:
            return self.pool[pid]
        t = norm_title(p.title)
        if len(t) <= 15 or not self.titles:
            return None
        hit = process.extractOne(t, self.titles, scorer=fuzz.ratio, score_cutoff=95.0)
        if hit and same_work(self.pool[hit[2]], p):
            return self.pool[hit[2]]
        return None

    def add(self, p: Paper) -> None:
        if p.doi:
            self.by_doi[p.doi] = p.id
        if p.arxiv_id:
            self.by_arxiv[p.arxiv_id] = p.id
        self.titles[p.id] = norm_title(p.title)


def integrate(existing: dict[str, Paper], incoming: list[Paper]) -> tuple[list[Paper], list[Paper]]:
    """Returns (new papers, updated existing papers). Incoming duplicates of each
    other are collapsed too. Exact id lookups first, then a fast fuzzy title match."""
    idx = _DuplicateIndex(dict(existing))
    new: dict[str, Paper] = {}
    updated: dict[str, Paper] = {}

    for raw in incoming:
        p = normalize(raw)
        match = idx.find(p)
        if match is None:
            p = p.model_copy(update={"id": canonical_id(p)})
            idx.pool[p.id] = new[p.id] = p
            idx.add(p)
            continue
        merged = merge(match, p)
        if merged != match:
            idx.pool[match.id] = merged
            idx.add(merged)
            (new if match.id in new else updated)[match.id] = merged
    return list(new.values()), list(updated.values())
