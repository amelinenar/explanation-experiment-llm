# import re
# import csv
# from pathlib import Path


# def result_script(path):
#     text = Path(path).read_text(encoding="utf-8")

#     # Regex patterns to match different formats
#     pattern = re.compile(r"\d+\.\s+(?P<metric>[A-Za-z ]+)\s+[-–—]\s+Score:\s+(?P<score>\d+)\s+\((?P<label>[^)]+)\)", re.I)

#     results = []

#     for match in pattern.finditer(text):
#         results.append({
#                 "metric": match.group("metric").strip(),
#                 "score": int(match.group("score")),
#                 "label": match.group("label").strip(),
#             })
        
#     return results


# def write_csv(task,dataset,llm_summarizer,summarization_prompt,judging_llm, judging_prompt, results, csv_path,):
#     """
#     Write the results to a csv file
#     """

#     fieldnames = [
#         "task",
#         "dataset",
#         "llm_summarizer",
#         "summarization_prompt",
#         "judging_llm",
#         "judging_prompt",
#         "metric",
#         "score",
#         "label",
#     ]

#     with open(csv_path, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()

#         for row in results:
            
#             row["task"] = task
#             row["dataset"] = dataset
#             row["llm_summarizer"] = llm_summarizer
#             row["summarization_prompt"] = summarization_prompt
#             row["judging_llm"] = judging_llm
#             row["judging_prompt"] = judging_prompt
       
#             writer.writerow(row)



import re
import csv
import os
from pathlib import Path


import re
from pathlib import Path

def result_script(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8")

    # Only allow the four metrics
    allowed_metrics = "Completeness|Accuracy|Relevance|Clarity"

    pattern = re.compile(
        rf"""
        ^\s*                                      # start of line, optional spaces
        (?:\#+\s*)?                               # optional markdown headers (#, ##, ###)
        (?:\-\s*)?
        (?:\d+\.\s*)?                             # optional numbering (1., 2., ...)
        (?:\*\*)?
        (?P<metric>{allowed_metrics})             # ONLY allowed metrics
        (?:\:)?
        (?:\*\*)?
        \s*                                       # optional spaces
        (?:                                       # value patterns
            [-–—:]?\s*Score:\s*(?P<score>\d+)\s*\((?P<label>[^)]+)\)[*+]?       # single-line
            |
            \n\s*Score:\s*(?P<score2>\d+)\s*\((?P<label2>[^)]+)\)         # two-line
            |
            :\s*(?P<score3>\d+)\s*\((?P<label3>[^)]+)\)                   # colon-only
            |
            :\s*Score\s*(?P<score4>\d+)\s*\((?P<label4>[^)]+)\)           # colon + Score
        )
        (?:\*\*)? 
        """,
        re.IGNORECASE | re.VERBOSE | re.MULTILINE,
    )

    results = []
    seen = set()

    for match in pattern.finditer(text):
        metric = match.group("metric")
        score = match.group("score") or match.group("score2") or match.group("score3") or match.group("score4")
        label = match.group("label") or match.group("label2") or match.group("label3") or match.group("label4")

        if not metric or not score or not label:
            continue

        score = int(score)

        key = (metric.strip(), score, label.strip())
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "metric": metric.strip(),
            "score": score,
            "label": label.strip(),
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

    file_exists = os.path.exists(csv_path)
    print(file_exists)
    write_header = not file_exists or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()


        for row in results:
            row.update(
                {
                "task": task,
                "dataset": dataset,
                "llm_summarizer": llm_summarizer,
                "summarization_prompt": summarization_prompt,
                "judging_llm": judging_llm,
                "judging_prompt": judging_prompt,
                }
            )
            writer.writerow(row)
