import csv
import os
from datetime import datetime


def save_query_to_csv(query, answer, doc_name, file_path="output_files/queries.csv"):
    """
    Saves the user query, retrieved answer, and document name to a CSV file.

    Args:
        query (str): The user's query.
        answer (str): The retrieved answer.
        doc_name (str): The name of the document from which the answer was generated.
        file_path (str): The path to the output CSV file.
    """
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(["Timestamp", "Query", "Answer", "Source Document"])

        writer.writerow(
            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), query, answer, doc_name]
        )
