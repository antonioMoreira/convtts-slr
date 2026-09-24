from convtts_slr.dedup import canonical_id, integrate, merge, norm_arxiv, norm_doi, same_work
from convtts_slr.models import Paper, Role


def test_norm_doi_variants():
    assert norm_doi(None) is None
    assert norm_doi("  HTTPS://DOI.ORG/10.1/ABC  ") == "10.1/abc"
    assert norm_doi("10.1/xyz") == "10.1/xyz"
    assert norm_doi("") is None


def test_norm_arxiv_variants():
    assert norm_arxiv(None) is None
    assert norm_arxiv("arXiv:2207.01063v3") == "2207.01063"
    assert norm_arxiv("https://arxiv.org/abs/2207.01063") == "2207.01063"
    assert norm_arxiv("https://arxiv.org/pdf/2207.01063.pdf") == "2207.01063"


def test_canonical_id_falls_back_to_a_stable_title_hash():
    p = Paper(id="x", title="Some Untitled Corpus With No Identifiers")
    cid = canonical_id(p)
    assert cid.startswith("title:")
    assert cid == canonical_id(p)  # deterministic
    assert len(cid) == len("title:") + 12


def test_same_work_rejects_matching_titles_more_than_a_year_apart():
    a = Paper(id="a", title="DailyTalk: Spoken Dialogue Dataset for Conversational TTS", year=2020)
    b = Paper(id="b", title="DailyTalk: Spoken Dialogue Dataset for Conversational TTS", year=2024)
    assert not same_work(a, b)


def test_merge_prefers_longer_abstract_and_upgrades_from_seed():
    keep = Paper(
        id="a",
        title="t",
        abstract="short",
        role=Role.CANDIDATE,
        is_seed=False,
        found_in_round=2,
    )
    other = Paper(
        id="b",
        title="t",
        abstract="a much longer and more informative abstract",
        role=Role.REFERENCE_BASELINE,
        is_seed=True,
        found_in_round=0,
    )
    merged = merge(keep, other)
    assert merged.abstract == other.abstract
    assert merged.is_seed is True
    assert merged.role == Role.REFERENCE_BASELINE  # only upgraded because other.is_seed
    assert merged.found_in_round == 0


def test_merge_never_downgrades_role_from_a_non_seed_other():
    keep = Paper(id="a", title="t", role=Role.REFERENCE_BASELINE)
    other = Paper(id="b", title="t", role=Role.CANDIDATE, is_seed=False)
    assert merge(keep, other).role == Role.REFERENCE_BASELINE


def test_integrate_collapses_duplicates_within_the_incoming_batch():
    p1 = Paper(id="a", title="Same Paper Title Repeated Twice Here", doi="10.1/dup")
    p2 = Paper(id="b", title="Same Paper Title Repeated Twice Here", doi="10.1/dup", venue="X")
    new, updated = integrate({}, [p1, p2])
    assert len(new) == 1
    assert not updated
    assert new[0].venue == "X"
