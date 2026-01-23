import re
import csv
from pathlib import Path


def result_script(path):
    text = Path(path).read_text(encoding="utf-8")

    # Regex patterns to match different formats
    pattern = re.compile(r"\d+\.\s+(?P<metric>[A-Za-z ]+)\s+[-–—]\s+Score:\s+(?P<score>\d+)\s+\((?P<label>[^)]+)\)", re.I)

    results = []

    for match in pattern.finditer(text):
        results.append({
                "metric": match.group("metric").strip(),
                "score": int(match.group("score")),
                "label": match.group("label").strip(),
            })
        
    return results


def write_csv(task,dataset,llm_summarizer,summarization_prompt,judging_llm, judging_prompt, results, csv_path,):
    """
    Write the results to a csv file
    """

    fieldnames = [
        "task",
        "dataset",
        "llm_summarizer",
        "summarization_prompt",
        "judging_llm",
        "judging_prompt",
        "metric",
        "score",
        "label",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            
            row["task"] = task
            row["dataset"] = dataset
            row["llm_summarizer"] = llm_summarizer
            row["summarization_prompt"] = summarization_prompt
            row["judging_llm"] = judging_llm
            row["judging_prompt"] = judging_prompt
       
            writer.writerow(row)
