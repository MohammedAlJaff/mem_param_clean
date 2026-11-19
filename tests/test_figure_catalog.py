"""Check figure identity, archived builds, and exact-build approval semantics."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
import figures


def recorded_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "OUTPUT_ROOT", tmp_path)
    spec = copy.deepcopy(figures.FIGURES["M3"])
    spec["stem"] = tmp_path / "matrix"
    spec["aliases"] = ["F99"]
    spec["stem"].with_suffix(".pdf").write_bytes(b"first PDF")
    source = tmp_path / "renderer.py"
    source.write_text("# exact source\n")
    identity = {"code": {str(source): figures.sha256(source)}, "inputs": {"data.csv": "first"}}
    monkeypatch.setattr(figures, "build_identity", lambda _: identity)
    figures.record_build("M3", spec, identity)
    return spec, identity


def test_legacy_aliases_and_scientific_contracts():
    assert figures.resolve_id("F07") == "A4_DIAGONAL"
    assert figures.resolve_id("f12") == "M5"
    assert figures.resolve_id("DS01") == "M2"
    assert figures.resolve_id("BT03") == "M3"
    assert figures.resolve_id("m2") == "M2"
    for spec in figures.FIGURES.values():
        figures.validate_spec(spec)
    with pytest.raises(ValueError, match="Duplicate"):
        figures.unique_keys([("BT03", {}), ("BT03", {})])
    spec = copy.deepcopy(figures.FIGURES["M3"])
    spec["metrics"] = ["inception_fid", "f_mem_sscd"]
    with pytest.raises(ValueError, match="new metric"):
        figures.validate_spec(spec)


def test_rebuild_preserves_first_archived_copy(tmp_path, monkeypatch):
    spec, identity = recorded_figure(tmp_path, monkeypatch)
    spec["stem"].with_suffix(".pdf").write_bytes(b"new PDF timestamp")
    figures.record_build("M3", spec, identity)
    assert (tmp_path / "M3.pdf").read_bytes() == b"first PDF"
    assert not (tmp_path / "F99.pdf").exists()


def test_approval_becomes_stale_after_data_or_pdf_change(tmp_path, monkeypatch):
    spec, identity = recorded_figure(tmp_path, monkeypatch)
    spec.update(status="approved", approved_build=figures.identity_id(identity))
    assert figures.figure_state("M3", spec)[:2] == ("approved", "current")
    identity["inputs"]["data.csv"] = "changed"
    assert figures.figure_state("M3", spec)[:2] == ("needs-review", "stale")
    identity["inputs"]["data.csv"] = "first"
    (tmp_path / "M3.pdf").write_bytes(b"edited outside the catalog")
    assert figures.figure_state("M3", spec)[:2] == ("needs-review", "stale")


def test_modified_archive_is_not_overwritten(tmp_path, monkeypatch):
    spec, identity = recorded_figure(tmp_path, monkeypatch)
    record = json.loads((tmp_path / "M3.build.json").read_text())
    archive = tmp_path / "versions" / "M3" / record["build_id"] / "M3.pdf"
    archive.write_bytes(b"external change")
    with pytest.raises(ValueError, match="Archived artifact changed"):
        figures.record_build("M3", spec, identity)
    assert archive.read_bytes() == b"external change"


@pytest.mark.parametrize("code,changes", [
    ("M3-SSCD", {"rho": 0.3}),
    ("M3-SSCD", {"alpha": 0.02}),
    ("M3-DIT", {"uncertainty": "sd"}),
    ("M3-DIT", {"seeds": [0, 1, 2]}),
    ("A1_DIT", {"uncertainty": "sd"}),
    ("A1_DIT", {"seeds": [0, 1, 2]}),
])
def test_matrix_variants_reject_misleading_thresholds_and_uncertainty(code, changes):
    spec = copy.deepcopy(figures.FIGURES[code])
    spec.update(changes)
    with pytest.raises(ValueError):
        figures.validate_spec(spec)
