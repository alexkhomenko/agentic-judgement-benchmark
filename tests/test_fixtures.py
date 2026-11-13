"""The tests exercise the harness against their own fixtures, never the project's experiment data."""

from __future__ import annotations

import pytest

from .conftest import PROJECT, ROOT


@pytest.mark.parametrize("folder", ["configs", "rubrics", "scenarios"])
def test_experiment_data_cannot_be_read_from_tests(folder: str) -> None:
    with pytest.raises(PermissionError, match="must not read experiment data"):
        (PROJECT / folder / "anything.yaml").read_text()


def test_fixture_tree_is_self_contained() -> None:
    assert ROOT.is_relative_to(PROJECT / "tests")
    assert (ROOT / "configs/models/fake/fake-alpha.yaml").read_text().startswith("# TEST FIXTURE")
