"""Step 8 remains strictly descriptive Social Network Analysis (#380)."""
from pathlib import Path
import ast
from ep24_sna import render_markdown_report, basic_metrics
import step_8_roihu_social_network_analysis as sna


def test_only_one_model_call_site_in_step_eight():
    tree = ast.parse(Path(sna.__file__).read_text())
    sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "chat"]
    assert len(sites) == 1
    assert not hasattr(sna, "CASTELLS_SYSTEM")
    assert not hasattr(sna, "_castells_interpretation")


def test_textbook_style_report_has_no_theoretical_interpretation():
    report = render_markdown_report(
        {"_storage_id": "one", "country": "finland"},
        [], [], basic_metrics([], []), "An empirical observation.")
    assert "Basic network statistics" in report
    assert "Castells" not in report
    assert "Luhmann" not in report


def test_step_eight_prompt_separates_ties_from_discursive_stances():
    assert "Disagreement" in sna.EXTRACTION_SYSTEM
    assert "NOT social opposition" in sna.EXTRACTION_SYSTEM
    assert "No additional political" in sna.EXTRACTION_SYSTEM
