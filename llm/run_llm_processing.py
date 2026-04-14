import argparse
import ast
import csv
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _read_csv_rows(path: str) -> List[List[str]]:
    with open(path, mode="r", newline="", errors="ignore") as f:
        return list(csv.reader(f))


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _parse_last_bracket_list(text: str) -> Optional[str]:
    """
    Extract the last [...] substring. Returns a string like "[1,2]" or None.
    This matches the original scripts' behavior that tries to find a list in model output.
    """
    if not text:
        return None
    start = text.rfind("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def _parse_single_label_in_brackets(text: str, default: str = "0") -> str:
    """
    Extract a single digit label from output like "[0]" "[1]" "[2]".
    """
    if not text:
        return default
    m = re.findall(r"\[(\d+)\]", text)
    return m[-1] if m else default


def _safe_literal_eval(value: str, default: Any) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return default


def _load_features(pkl_path: str) -> Sequence[Any]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _get_feature_attr(feature: Any, name: str) -> Any:
    if hasattr(feature, name):
        return getattr(feature, name)
    raise AttributeError(f"Feature object has no attribute {name!r}.")


def _build_numbered_sentences(sentences: Sequence[str]) -> List[str]:
    return [f"{i}. {s}" for i, s in enumerate(sentences)]


def _normalize_event_sentence_field(raw: str) -> Any:
    # Matches original scripts that clean up nested brackets before literal_eval.
    doc_cleaned = raw.replace("[[", "[").replace("]]", "]").replace("[", "").replace("]", "")
    raw2 = "[" + doc_cleaned + "]"
    return _safe_literal_eval(raw2, default=[])


@dataclass
class ProviderConfig:
    provider: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    qianfan_access_key_env: str = "QIANFAN_ACCESS_KEY"
    qianfan_secret_key_env: str = "QIANFAN_SECRET_KEY"
    device_map: str = "auto"


class LLMProvider:
    def generate(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class OpenAICompatProvider(LLMProvider):
    def __init__(self, model: str, base_url: Optional[str], api_key: str):
        from openai import OpenAI  # lazy import

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model

    def generate(self, messages: List[Dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(model=self._model, messages=messages)
        return resp.choices[0].message.content or ""


class QianfanErnieProvider(LLMProvider):
    def __init__(self, model: str):
        import qianfan  # lazy import

        self._chat = qianfan.ChatCompletion()
        self._model = model

    def generate(self, messages: List[Dict[str, str]]) -> str:
        # qianfan wants [{role, content}] like OpenAI.
        resp = self._chat.do(model=self._model, messages=messages)
        return resp.get("body", {}).get("result", "") or ""


class HFPipelineProvider(LLMProvider):
    def __init__(self, model_id: str):
        import torch  # lazy import
        import transformers  # lazy import

        self._pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self._terminators = [
            self._pipeline.tokenizer.eos_token_id,
            self._pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def generate(self, messages: List[Dict[str, str]]) -> str:
        out = self._pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=self._terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        return out[0]["generated_text"][-1]["content"]


class HFCausalLMChatTemplateProvider(LLMProvider):
    def __init__(self, model_id: str, device_map: str = "auto"):
        import torch  # lazy import
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._model.device)
        outputs = self._model.generate(**inputs, do_sample=True, max_new_tokens=256)
        decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return "".join(decoded)


def _make_provider(cfg: ProviderConfig) -> LLMProvider:
    if cfg.provider == "openai":
        api_key_env = cfg.api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing env var {api_key_env} for provider=openai")
        if not cfg.model:
            raise RuntimeError("--model is required for provider=openai")
        return OpenAICompatProvider(model=cfg.model, base_url=cfg.base_url, api_key=api_key)

    if cfg.provider == "qianfan":
        if not cfg.model:
            cfg.model = "ERNIE-4.0-8K"
        access = os.environ.get(cfg.qianfan_access_key_env, "")
        secret = os.environ.get(cfg.qianfan_secret_key_env, "")
        if not access or not secret:
            raise RuntimeError(
                f"Missing env vars {cfg.qianfan_access_key_env}/{cfg.qianfan_secret_key_env} for provider=qianfan"
            )
        # qianfan SDK reads env vars, so just ensure they are present.
        return QianfanErnieProvider(model=cfg.model)

    if cfg.provider == "hf_pipeline":
        if not cfg.model:
            raise RuntimeError("--model is required for provider=hf_pipeline")
        return HFPipelineProvider(model_id=cfg.model)

    if cfg.provider == "hf_causallm":
        if not cfg.model:
            raise RuntimeError("--model is required for provider=hf_causallm")
        return HFCausalLMChatTemplateProvider(model_id=cfg.model, device_map=cfg.device_map)

    raise ValueError(f"Unknown provider: {cfg.provider}")


def _prompt_select_sentences(
    numbered_sentences: Sequence[str],
    e1: str,
    e2: str,
    target: int,
    rel_type: int,
    num_sen1: int,
    num_sen2: int,
    event_sen1: str,
    event_sen2: str,
) -> List[Dict[str, str]]:
    # Keep the prompt structure close to original scripts.
    if target == 1:
        if rel_type == 0:
            user = (
                f"The input document is labeled with an ordinal number before each sentence, e.g. (0,1...) , "
                f"knowing that event: {e1} and event: {e2} in the {num_sen1} sentence are causally related, "
                f"{e1} means [{event_sen1}].{e2} means[{event_sen2}]"
                f"identify which sentences from the document articulate the causal relationship between events {e1} and {e2}. "
                f"( they may be clustered in the {num_sen1} sentence, but this does not preclude the need for multiple sentences.) "
                f"Answer Present the sentence numbers as a list, e.g., “[1,2,...]”.don't output other content. "
                f"Document :{list(numbered_sentences)}"
            )
        else:
            user = (
                f"The input document is labeled with an ordinal number before each sentence, e.g. (0,1...) , "
                f"knowing that event: {e1} and event: {e2} in the {num_sen1} and {num_sen2} sentences are causally related, "
                f"{e1} means [{event_sen1}].{e2} means[{event_sen2}]"
                f"identify which sentences from the document articulate the causal relationship between events {e1} and {e2}. "
                f"( they may be clustered in the {num_sen1} and {num_sen2} sentences, but this does not preclude the need for others sentences.) "
                f"Answer Present the sentence numbers as a list, e.g., “[1,2,...]”.don't output other content.. "
                f"Document :{list(numbered_sentences)}"
            )
    else:
        if rel_type == 0:
            user = (
                f"The input document is labeled with an ordinal number before each sentence, e.g. (0,1...) , "
                f"knowing that event: {e1} and event: {e2} in the {num_sen1} sentence are causally related, "
                f"{e1} means [{event_sen1}].{e2} means[{event_sen2}]"
                f"identify which sentences from the document articulate the causal relationship between events {e2} and {e1}. "
                f"( they may be clustered in the {num_sen1} sentence, but this does not preclude the need for multiple sentences.) "
                f"Answer Present the sentence numbers as a list, e.g., “[1,2,...]”.don't output other content.. "
                f"Document :{list(numbered_sentences)}"
            )
        else:
            user = (
                f"The input document is labeled with an ordinal number before each sentence, e.g. (0,1...) , "
                f"knowing that event: {e1} and event: {e2} in the {num_sen1} and {num_sen2} sentences are causally related, "
                f"{e1} means [{event_sen1}].{e2} means[{event_sen2}]"
                f"identify which sentences from the document articulate the causal relationship between events {e2} and {e1}. "
                f"( they may be clustered in the {num_sen2} and {num_sen1} sentences, but this does not preclude the need for others sentences.) "
                f"Answer Present the sentence numbers as a list, e.g., “[1,2,...]”.don't output other content.. "
                f"Document :{list(numbered_sentences)}"
            )

    return [
        {"role": "system", "content": "You are an expert with few words and will only answer questions accurately"},
        {"role": "user", "content": user},
    ]


def _prompt_classify_direction(document_text: str, e1: str, e2: str) -> List[Dict[str, str]]:
    # Mirrors ESL_llama3.py's task: output [0]/[1]/[2]
    user = (
        f"This is a document: {document_text}, this is an event mentioning the pair: {e1},{e2}."
        f"Please determine the causal relationship between these two events according to the document and the explanation of two targets."
        f"and just output number([0]/[1]/[2]) . "
        f"if there is a NONE relationship, output [0]; if there is a PRECONDITION relationship, output [1]; if there is a FALLING_ACTION relationship, output [2]."
        f"NONE:no relationship of two events;PRECONDITION;the first event is the reason why the second event occurs;"
        f"FALLING_ACTION:the second event is reason and the first event is result"
    )
    return [
        {"role": "system", "content": "You are an expert with few words and will only answer questions accurately"},
        {"role": "user", "content": user},
    ]


def run_select_sentences(args: argparse.Namespace) -> None:
    features = _load_features(args.features_pkl)
    rows = _read_csv_rows(args.data_csv)

    # Pre-extract per-document metadata from features.
    targets_all: List[List[int]] = []
    rel_types_all: List[List[int]] = []
    node_sens_all: List[List[int]] = []
    for f in features:
        targets_all.append(list(_get_feature_attr(f, "target")))
        rel_types_all.append(list(_get_feature_attr(f, "rel_type")))
        node_event = _get_feature_attr(f, "node_event")
        node_sens_all.append([int(x[0]) for x in node_event])

    provider = _make_provider(
        ProviderConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            device_map=args.device_map,
        )
    )

    _ensure_parent_dir(args.out_csv)
    with open(args.out_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "doc_id", "text", "sentences", "event_pair", "node_event_id", "answer(id)"],
        )
        writer.writeheader()

    # Assumes data_csv rows align with features order (as in original scripts: use i-1 indexing).
    # If they don't align, the caller should pre-align them before running.
    for i in range(1, len(rows)):
        row = rows[i]
        doc_row_id = row[0]
        doc_id = row[1]
        content = row[2]
        sentences = _safe_literal_eval(row[3], default=[])
        events_list = _safe_literal_eval(row[4], default=[])
        event_sentence = _normalize_event_sentence_field(row[6]) if len(row) > 6 else []
        numbered = _build_numbered_sentences(sentences)

        a = 0
        for j in range(0, max(0, len(events_list) - 1)):
            outputs_id: List[str] = []
            node_event_id: List[List[int]] = []
            event_pair: List[List[str]] = []
            for z in range(j + 1, len(events_list)):
                node_event_id.append([j, z])
                event_pair.append([events_list[j], events_list[z]])
                if targets_all[i - 1][a] == 0:
                    outputs_id.append("None")
                else:
                    msg = _prompt_select_sentences(
                        numbered_sentences=numbered,
                        e1=events_list[j],
                        e2=events_list[z],
                        target=int(targets_all[i - 1][a]),
                        rel_type=int(rel_types_all[i - 1][a]),
                        num_sen1=int(node_sens_all[i - 1][j]),
                        num_sen2=int(node_sens_all[i - 1][z]),
                        event_sen1=str(event_sentence[j]) if j < len(event_sentence) else "",
                        event_sen2=str(event_sentence[z]) if z < len(event_sentence) else "",
                    )
                    text = provider.generate(msg)
                    lst = _parse_last_bracket_list(text)
                    outputs_id.append(lst[1:-1] if lst else "None")
                a += 1

            new_row = {
                "id": doc_row_id,
                "doc_id": doc_id,
                "text": content if j == 0 else " ",
                "sentences": numbered if j == 0 else " ",
                "event_pair": event_pair,
                "node_event_id": node_event_id,
                "answer(id)": outputs_id,
            }
            with open(args.out_csv, mode="a", newline="", encoding="utf-8") as out:
                w = csv.writer(out)
                w.writerow([new_row[k] for k in ["id", "doc_id", "text", "sentences", "event_pair", "node_event_id", "answer(id)"]])


def run_classify_direction(args: argparse.Namespace) -> None:
    features = _load_features(args.features_pkl)
    rows = _read_csv_rows(args.data_csv)

    targets_all: List[List[int]] = []
    for f in features:
        targets_all.append(list(_get_feature_attr(f, "target")))

    provider = _make_provider(
        ProviderConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            device_map=args.device_map,
        )
    )

    _ensure_parent_dir(args.out_csv)
    with open(args.out_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "text", "event_pair", "node_event_id", "answer(id)"],
        )
        writer.writeheader()

    for i in range(1, len(rows)):
        row = rows[i]
        doc_row_id = row[0]
        content = row[2]
        events_list = _safe_literal_eval(row[4], default=[])

        x = 0
        for j in range(0, len(events_list)):
            outputs_id: List[str] = []
            node_event_id: List[List[int]] = []
            event_pair: List[List[str]] = []
            for z in range(j + 1, len(events_list) - 1):
                node_event_id.append([j, z])
                event_pair.append([events_list[j], events_list[z]])
                answer = provider.generate(_prompt_classify_direction(content, events_list[j], events_list[z]))
                outputs_id.append(_parse_single_label_in_brackets(answer, default="0"))
                x += 1

            with open(args.out_csv, mode="a", newline="", encoding="utf-8") as out:
                w = csv.writer(out)
                w.writerow([doc_row_id, content, event_pair, node_event_id, outputs_id])


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified LLM data processing for CogECI_1")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--provider", required=True, choices=["openai", "qianfan", "hf_pipeline", "hf_causallm"])
        sp.add_argument("--model", default=None, help="provider model id/name (required for openai/hf*)")
        sp.add_argument("--base_url", default=None, help="OpenAI-compatible base_url (optional)")
        sp.add_argument("--api_key_env", default="OPENAI_API_KEY", help="Env var name for OpenAI-compatible api key")
        sp.add_argument("--device_map", default="auto", help="HF device_map for hf_causallm")

    sp1 = sub.add_parser("select-sentences", help="Select causal sentences for each event pair")
    sp1.add_argument("--features_pkl", required=True)
    sp1.add_argument("--data_csv", required=True)
    sp1.add_argument("--out_csv", required=True)
    add_common(sp1)
    sp1.set_defaults(func=run_select_sentences)

    sp2 = sub.add_parser("classify-direction", help="Classify direction/label for each event pair (ESL_llama3 style)")
    sp2.add_argument("--features_pkl", required=True)
    sp2.add_argument("--data_csv", required=True)
    sp2.add_argument("--out_csv", required=True)
    add_common(sp2)
    sp2.set_defaults(func=run_classify_direction)

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

