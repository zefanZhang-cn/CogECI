import argparse
import ast
import csv
import os
import pickle
from typing import Any, Dict, List, Sequence


def _safe_literal_eval(value: str, default: Any) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return default


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _numbered_sentences(sentences: Sequence[str]) -> List[str]:
    return [f"{i}. {s}" for i, s in enumerate(sentences)]


def _openai_client(api_key_env: str, base_url: str):
    from openai import OpenAI  # lazy import

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing env var {api_key_env} for OpenAI-compatible provider")
    return OpenAI(api_key=api_key, base_url=base_url)


def _gpt_event_svo(client, model: str, text: Sequence[str], event: str, sen_num: int) -> str:
    messages = [
        {"role": "system", "content": "You are an expert with few words and will only answer questions accurately"},
        {
            "role": "user",
            "content": (
                f"I have a document and an event word who is in sentence ({sen_num}) in the document, "
                f"please fully understand this event and generat a subject-verb-object short sentence. "
                f"It is possible to describe this event. Output only this sentence. "
                f"document:{list(text)} event:{event}"
            ),
        },
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Event extension: generate SVO-like attributes for event mentions (CTB).")
    p.add_argument("--features_pkl", required=True, help="Path to features.pkl (used to locate event sentence index).")
    p.add_argument("--data_csv", required=True, help="Input CTB csv (same order as features).")
    p.add_argument("--out_csv", required=True, help="Output csv path.")
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI-compatible model name.")
    p.add_argument("--base_url", default="https://api.chatanywhere.tech/v1", help="OpenAI-compatible base_url.")
    p.add_argument("--api_key_env", default="OPENAI_API_KEY", help="Env var name for the API key.")
    args = p.parse_args()

    features = pickle.load(open(args.features_pkl, "rb"))
    node_sens: List[List[int]] = []
    for f in features:
        node_event = getattr(f, "node_event")
        node_sens.append([int(x[0]) for x in node_event])

    rows = []
    with open(args.data_csv, mode="r", newline="", errors="ignore") as datafile:
        rows = list(csv.reader(datafile))

    client = _openai_client(api_key_env=args.api_key_env, base_url=args.base_url)

    _ensure_parent_dir(args.out_csv)
    with open(args.out_csv, mode="w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=["id", "doc_id", "text", "sentences", "node_event", "attributes", "event_sen"],
        )
        writer.writeheader()

    for i in range(1, len(rows)):
        row = rows[i]
        rid = row[0]
        doc_id = row[1]
        content = row[2]
        sentences = _safe_literal_eval(row[3], default=[])
        events_list = _safe_literal_eval(row[4], default=[])
        numbered = _numbered_sentences(sentences)

        attrs: List[str] = []
        for j, ev in enumerate(events_list):
            sen_num = node_sens[i - 1][j] if j < len(node_sens[i - 1]) else 0
            attrs.append(_gpt_event_svo(client, args.model, numbered, ev, sen_num))

        with open(args.out_csv, mode="a", newline="", encoding="utf-8") as out:
            w = csv.writer(out)
            w.writerow([rid, doc_id, content, numbered, row[4], row[5] if len(row) > 5 else "", attrs])


if __name__ == "__main__":
    main()

