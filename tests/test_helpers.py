from bybit_app.utils.helpers import ensure_link_id


def test_ensure_link_id_keeps_suffix_for_long_ids() -> None:
    original = "OCO-THIS-IS-A-SUPER-LONG-GROUP-NAME-PRIMARY"

    result = ensure_link_id(original)

    assert result.endswith("-PRIMARY")
    assert len(result) <= 36


def test_ensure_link_id_distinguishes_similar_suffixes() -> None:
    base = "OCO-EXTREMELY-LONG-GROUP-NAME"
    primary = ensure_link_id(f"{base}-PRIMARY")
    take_profit = ensure_link_id(f"{base}-TP")

    assert primary != take_profit
    assert primary.endswith("-PRIMARY")
    assert take_profit.endswith("-TP")


def test_ensure_link_id_handles_none() -> None:
    assert ensure_link_id(None) is None


def test_ensure_link_id_preserves_prefix_context() -> None:
    base = "OCO-THIS-PREFIX-IS-IMPORTANT-AND-VERY-LONG"
    result = ensure_link_id(f"{base}-PRIMARY")

    assert result.startswith("OCO-THIS-PRE")


def test_ensure_link_id_differs_for_middle_changes() -> None:
    template = "OCO-AAA-{}-ZZZ-PRIMARY"
    first = ensure_link_id(template.format("MIDDLE-ONE"))
    second = ensure_link_id(template.format("MIDDLE-TWO"))

    assert first != second
