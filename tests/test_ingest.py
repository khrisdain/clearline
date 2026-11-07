from pathlib import Path

from clearline.ingest import chunk_document, load_file, pack, parse_front_matter, split_sections
from clearline.models import Classification, Document

SAMPLE = Path(__file__).parent.parent / "sample_data"


def test_front_matter_sets_title_classification_and_groups():
    document = load_file(SAMPLE / "budget-forecast.md")
    assert document.title == "2027 Operating Budget Forecast"
    assert document.classification is Classification.PROTECTED_B
    assert document.groups == frozenset({"finance"})
    assert not document.text.startswith("---")


def test_missing_front_matter_means_unclassified_and_open():
    meta, body = parse_front_matter("# Hello\n\nworld")
    assert meta == {} and body.startswith("# Hello")


def test_sections_follow_headings_and_keep_the_lead_text():
    sections = split_sections("Intro line.\n\n# A\nalpha\n\n## B\nbeta")
    assert sections == [("Overview", "Intro line."), ("A", "alpha"), ("B", "beta")]


def test_packing_respects_the_window_and_overlaps():
    body = "\n\n".join(f"Sentence number {i} is here and says something useful." for i in range(40))
    windows = pack(body, max_chars=300, overlap=60)
    assert len(windows) > 1
    assert all(len(w) <= 300 for w in windows)
    assert windows[1].split()[0] in windows[0]  # the next window starts inside the previous one's tail


def test_chunks_inherit_the_documents_access_labels():
    doc = Document(id="d1", title="T", text="# S\n" + "word " * 600, classification=Classification.PROTECTED_A,
                   groups=frozenset({"it"}))
    chunks = chunk_document(doc, max_chars=500)
    assert len(chunks) > 1
    assert {(c.classification, c.groups, c.section) for c in chunks} == {(Classification.PROTECTED_A, frozenset({"it"}), "S")}
    assert [c.position for c in chunks] == list(range(len(chunks)))
