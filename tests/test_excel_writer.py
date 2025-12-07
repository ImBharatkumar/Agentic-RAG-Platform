import os
import csv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.excel_writer import save_query_to_csv


def test_save_query_to_csv():
    """Tests the save_query_to_csv function."""
    file_path = "test_queries.csv"
    query = "What is the capital of France?"
    answer = "Paris"
    doc_name = "france.txt"

    # Clean up before the test
    if os.path.exists(file_path):
        os.remove(file_path)

    # First call to create the file and write the header
    save_query_to_csv(query, answer, doc_name, file_path=file_path)

    assert os.path.exists(file_path)

    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        assert header == ["Timestamp", "Query", "Answer", "Source Document"]

        row1 = next(reader)
        assert row1[1] == query
        assert row1[2] == answer
        assert row1[3] == doc_name

    # Second call to append a new row
    query2 = "What is the capital of Germany?"
    answer2 = "Berlin"
    doc_name2 = "germany.txt"
    save_query_to_csv(query2, answer2, doc_name2, file_path=file_path)

    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        next(reader)  # Skip first row
        row2 = next(reader)
        assert row2[1] == query2
        assert row2[2] == answer2
        assert row2[3] == doc_name2

    # # Clean up after the test
    # os.remove(file_path)


if __name__ == "__main__":
    test_save_query_to_csv()
    print("Test passed!")
