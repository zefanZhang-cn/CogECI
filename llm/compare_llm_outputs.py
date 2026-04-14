import argparse
import ast
import csv
import os
from typing import Any, List, Optional, Tuple


def _safe_literal_eval(value: str, default: Any) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return default


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _read_rows(path: str) -> List[List[str]]:
    with open(path, mode="r", newline="", errors="ignore") as f:
        return list(csv.reader(f))


def _maybe_inherit_field(prev: Optional[Any], raw: str) -> Tuple[Any, Any]:
    """
    Some legacy CSVs store text/sentences only on the first row of a block.
    If parsing fails, fall back to previous parsed value.
    """
    parsed = _safe_literal_eval(raw, default=None)
    if parsed is None:
        return prev, prev
    return parsed, parsed


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two LLM outputs CSVs and write diffs.")
    p.add_argument("--llm_file1", required=True)
    p.add_argument("--llm_file2", required=True)
    p.add_argument("--out_diff_csv", required=True)
    args = p.parse_args()

    rows1 = _read_rows(args.llm_file1)
    rows2 = _read_rows(args.llm_file2)
    if len(rows1) != len(rows2):
        raise ValueError(f"Row count mismatch: {len(rows1)} vs {len(rows2)}")

    _ensure_parent_dir(args.out_diff_csv)
    with open(args.out_diff_csv, mode="w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["id", "doc_id", "sentences", "event_pair", "answer_diff", "sentence_text"])
        writer.writeheader()

    prev_sentences_1 = None
    prev_sentences_2 = None

    same = 0
    diff = 0
    for i in range(1, len(rows1)):
        r1 = rows1[i]
        r2 = rows2[i]

        # Expected legacy schema: [0]=id, [1]=doc_id, [3]=sentences, [4]=event_pair, [6]=answer(id)
        rid = r1[0]
        doc_id = r1[1] if len(r1) > 1 else ""

        if len(r1) > 3:
            prev_sentences_1, sentences_1 = _maybe_inherit_field(prev_sentences_1, r1[3])
        else:
            sentences_1 = prev_sentences_1

        if len(r2) > 3:
            prev_sentences_2, sentences_2 = _maybe_inherit_field(prev_sentences_2, r2[3])
        else:
            sentences_2 = prev_sentences_2

        event_pairs_1 = _safe_literal_eval(r1[4], default=[])
        event_pairs_2 = _safe_literal_eval(r2[4], default=[])

        answers_1 = _safe_literal_eval(r1[6], default=[])
        answers_2 = _safe_literal_eval(r2[6], default=[])

        # Compare per pair
        n = min(len(answers_1), len(answers_2), len(event_pairs_1), len(event_pairs_2))
        for j in range(n):
            a1 = answers_1[j]
            a2 = answers_2[j]
            if a1 == "None" and a2 == "None":
                continue
            if a1 != a2:
                diff += 1
                sent_text = ""
                # If answers are single indices, try to show sentence text.
                try:
                    if a1 != "None" and sentences_1 is not None:
                        sent_text += f"file1: {sentences_1[int(a1)]}\n"
                    else:
                        sent_text += "file1: None\n"
                except Exception:
                    sent_text += "file1: (unparsed)\n"
                try:
                    if a2 != "None" and sentences_2 is not None:
                        sent_text += f"file2: {sentences_2[int(a2)]}\n"
                    else:
                        sent_text += "file2: None\n"
                except Exception:
                    sent_text += "file2: (unparsed)\n"

                with open(args.out_diff_csv, mode="a", newline="", encoding="utf-8") as out:
                    w = csv.writer(out)
                    w.writerow([rid, doc_id, sentences_1, event_pairs_1[j], f"{a1}:{a2}", sent_text.strip()])
            else:
                same += 1

    total = same + diff
    if total:
        print(f"Done. same={same} diff={diff} acc={same/total:.4f}")
    else:
        print("Done. No comparable labels found.")


if __name__ == "__main__":
    main()

