from ragcompactor.citations import format_citations


def test_format_citations_expands_line_ranges():
    text = "A claim\u30102\u2020L1-L4\u3011\u30101\u2020L4-L7\u3011."

    assert format_citations(text) == (
        "A claim (Source 2, lines 1\u20134; Source 1, lines 4\u20137)."
    )


def test_format_citations_handles_single_line_references():
    assert format_citations("Claim\u30103\u2020L8\u3011.") == "Claim (Source 3, line 8)."


def test_format_citations_leaves_unmatched_text_unchanged():
    text = "Claim\u30102\u2020L1-L4\u3011 and malformed \u3010ref\u3011."

    assert format_citations(text) == "Claim (Source 2, lines 1\u20134) and malformed \u3010ref\u3011."
