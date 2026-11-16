"""Tests for the figure exporter's manifest and prune logic."""

import sys
from pathlib import Path

import figure_export as fe
import pytest


class _FakeChart:
    """Stand in for an Altair chart: writes a file instead of calling vl-convert."""

    def __init__(self) -> None:
        self.engine: str | None = None
        self.props: dict[str, int] = {}

    def properties(self, **kwargs: int) -> "_FakeChart":
        self.props = kwargs
        return self

    def save(self, path: Path, engine: str) -> None:
        self.engine = engine
        Path(path).write_bytes(b"x")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch):
    fe._written.clear()
    for key in ("QUARTO_DOCUMENT_FILE", "QUARTO_DOCUMENT_PATH", "QUARTO_PROJECT_ROOT"):
        monkeypatch.delenv(key, raising=False)
    yield
    fe._written.clear()


def _manifest(out: Path, slug: str) -> list[str]:
    return (out / ".figure_manifest" / f"{slug}.txt").read_text().splitlines()


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _write_manifest(directory: Path, source: str, names: list[str]) -> Path:
    path = directory / ".figure_manifest" / f"{fe.source_slug(source)}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# source: {source}", *names]) + "\n")
    return path


class TestSource:
    def test_quarto_document_is_project_relative(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "a.qmd")
        monkeypatch.setenv("QUARTO_DOCUMENT_PATH", str(tmp_path / "benchmarks"))
        monkeypatch.setenv("QUARTO_PROJECT_ROOT", str(tmp_path))
        assert fe.current_source() == "benchmarks/a.qmd"

    def test_interactive_without_main_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys.modules["__main__"], "__file__", None, raising=False)
        assert fe.current_source() == "interactive"

    def test_slug_is_a_safe_file_name(self) -> None:
        assert fe.source_slug("benchmarks/a b.qmd") == "benchmarks__a_b.qmd"


class TestSaveChart:
    def test_exports_with_vl_convert_and_records(self, tmp_path: Path) -> None:
        chart = _FakeChart()
        path = fe.save_chart(
            chart, "mc_x", output_dir=tmp_path / "new", width=100, height=50
        )
        assert path == tmp_path / "new" / "mc_x.pdf"
        assert path.exists()
        assert chart.engine == "vl-convert"
        assert chart.props == {"width": 100, "height": 50}
        slug = fe.source_slug(fe.current_source())
        assert _manifest(tmp_path / "new", slug)[1:] == ["mc_x.pdf"]

    def test_real_altair_chart_exports_a_pdf(self, tmp_path: Path) -> None:
        alt = pytest.importorskip("altair")
        pytest.importorskip("vl_convert")
        chart = (
            alt.Chart(alt.Data(values=[{"x": 1, "y": 2}]))
            .mark_point()
            .encode(x="x:Q", y="y:Q")
        )
        path = fe.save_chart(chart, "real", output_dir=tmp_path)
        assert path.read_bytes().startswith(b"%PDF")

    def test_truncate_then_append_without_duplicates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "n.qmd")
        monkeypatch.setenv("QUARTO_DOCUMENT_PATH", str(tmp_path))
        monkeypatch.setenv("QUARTO_PROJECT_ROOT", str(tmp_path))
        out = tmp_path / "figs"
        _write_manifest(out, "n.qmd", ["old.pdf"])
        fe.save_chart(_FakeChart(), "a", output_dir=out)
        fe.save_chart(_FakeChart(), "b", output_dir=out)
        fe.save_chart(_FakeChart(), "a", output_dir=out)
        assert _manifest(out, "n.qmd") == ["# source: n.qmd", "a.pdf", "b.pdf"]
        # A new process (cleared state) truncates again.
        fe._written.clear()
        fe.save_chart(_FakeChart(), "c", output_dir=out)
        assert _manifest(out, "n.qmd") == ["# source: n.qmd", "c.pdf"]

    def test_sources_get_separate_manifests(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("QUARTO_DOCUMENT_PATH", str(tmp_path))
        monkeypatch.setenv("QUARTO_PROJECT_ROOT", str(tmp_path))
        for doc in ("one.qmd", "two.qmd"):
            monkeypatch.setenv("QUARTO_DOCUMENT_FILE", doc)
            fe.save_chart(_FakeChart(), doc[:3], output_dir=tmp_path)
        assert _manifest(tmp_path, "one.qmd")[1:] == ["one.pdf"]
        assert _manifest(tmp_path, "two.qmd")[1:] == ["two.pdf"]


class TestPrune:
    def _tree(self, tmp_path: Path) -> tuple[Path, Path]:
        project = tmp_path / "proj"
        (project / "nb").mkdir(parents=True)
        (project / "nb" / "live.qmd").write_text("")
        root = project / "figures"
        managed = root / "managed"
        _touch(managed / "keep.pdf")
        _touch(managed / "stale.pdf")
        _touch(managed / "gone.png")
        _touch(managed / "note.txt")
        _write_manifest(managed, "nb/live.qmd", ["keep.pdf"])
        _write_manifest(managed, "nb/deleted.qmd", ["gone.png"])
        _touch(root / "loose" / "hand_made.pdf")
        return project, root

    def test_dry_run_changes_nothing(self, tmp_path: Path) -> None:
        project, root = self._tree(tmp_path)
        report = fe.prune(root, project)
        assert sorted(p.name for p in report.stale) == ["gone.png", "stale.pdf"]
        assert [p.name for p in report.orphan_manifests] == ["nb__deleted.qmd.txt"]
        assert list(report.unmanaged_dirs) == [root / "loose"]
        assert report.kept == 1
        assert (root / "managed" / "stale.pdf").exists()
        assert (root / "managed" / ".figure_manifest" / "nb__deleted.qmd.txt").exists()

    def test_apply_removes_stale_and_orphans_but_not_unmanaged(
        self, tmp_path: Path
    ) -> None:
        project, root = self._tree(tmp_path)
        fe.prune(root, project, apply=True)
        managed = root / "managed"
        assert (managed / "keep.pdf").exists()
        assert not (managed / "stale.pdf").exists()
        assert not (managed / "gone.png").exists()
        assert (managed / "note.txt").exists()
        assert not (managed / ".figure_manifest" / "nb__deleted.qmd.txt").exists()
        assert (managed / ".figure_manifest" / "nb__live.qmd.txt").exists()
        assert (root / "loose" / "hand_made.pdf").exists()

    def test_apply_with_include_unmanaged(self, tmp_path: Path) -> None:
        project, root = self._tree(tmp_path)
        fe.prune(root, project, apply=True, include_unmanaged=True)
        assert not (root / "loose" / "hand_made.pdf").exists()

    def test_interactive_manifest_is_never_orphaned(self, tmp_path: Path) -> None:
        root = tmp_path / "figs"
        _touch(root / "a.pdf")
        _write_manifest(root, "interactive", ["a.pdf"])
        report = fe.prune(root, tmp_path)
        assert report.stale == []
        assert report.orphan_manifests == []

    def test_cli_reports_and_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project, root = self._tree(tmp_path)
        assert fe.main(["prune", str(root), "--project-root", str(project)]) == 0
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "stale.pdf" in out
        assert "loose/" in out
        assert (root / "managed" / "stale.pdf").exists()
        assert (
            fe.main(["prune", str(root), "--project-root", str(project), "--apply"])
            == 0
        )
        assert not (root / "managed" / "stale.pdf").exists()

    def test_missing_root_is_not_an_error(self, tmp_path: Path) -> None:
        assert fe.main(["prune", str(tmp_path / "nope")]) == 0
