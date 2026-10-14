from pathlib import Path

from src.format.formatter import to_csv, to_linearized_deplot, to_markdown
from src.format.parser import parse_csv_table, parse_deplot_linearized
from src.format.schema import TableSchema

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "deplot_output.txt"
RAW_DEPLOT = FIXTURE_PATH.read_text(encoding="utf-8")


def test_parse_fixed_deplot_output():
    table = parse_deplot_linearized(RAW_DEPLOT)

    assert table.title == "Revenue vs Expenses"
    assert table.columns == ["product", "revenue", "expenses"]
    assert table.rows == [
        ["Speakers", 112472.5, 89181.4],
        ["Blu-Ray Players", 64945.5, 60411.8],
    ]


def test_linearized_output_contract_and_round_trip():
    table = parse_deplot_linearized(RAW_DEPLOT)

    serialized = to_linearized_deplot(table)
    restored = parse_deplot_linearized(serialized)

    assert serialized.startswith("<START> TITLE | Revenue vs Expenses\n")
    assert "<NEW_ROW> product | revenue | expenses" in serialized
    assert "112472.5 | 89181.4" in serialized
    assert serialized.endswith("<END>")
    assert restored.title == table.title
    assert restored.columns == table.columns
    assert restored.rows == table.rows


def test_linearized_output_always_includes_title():
    table = TableSchema(columns=["Item", "Value"], rows=[["A", 2.5]])

    serialized = to_linearized_deplot(table)
    restored = parse_deplot_linearized(serialized)

    assert serialized.startswith("<START> TITLE |")
    assert restored.title == ""
    assert restored.columns == ["item", "value"]
    assert restored.rows == [["A", 2.5]]


def test_markdown_and_csv_serializers_use_canonical_table():
    table = parse_deplot_linearized(RAW_DEPLOT)
    markdown = to_markdown(table)
    csv_text = to_csv(table)
    restored_csv = parse_csv_table(csv_text)

    assert "### Revenue vs Expenses" in markdown
    assert "Speakers" in markdown
    assert "112472.5" in markdown
    assert "89181.4" in markdown
    assert restored_csv.columns == ["product", "revenue", "expenses"]
    assert restored_csv.rows == table.rows

def main():
    print("Test parse fixed deplot output !")
    test_parse_fixed_deplot_output()
    print("Passed")
    print("Test linearized output contract and round trip!")
    test_linearized_output_contract_and_round_trip()
    print("Passed")
    print("Test linearized output always include title!")
    test_linearized_output_always_includes_title()
    print("Passed")
    print("Test mardown and csv serializers use canonical table!")
    test_markdown_and_csv_serializers_use_canonical_table()
    print("Passed")


if __name__ == "__main__":
    main()