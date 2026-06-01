"""Compute sentence embeddings for the 1608+ patterns and the AdvBench goals,
saving to disk for `transition_embed.py` to consume.

Default: BAAI/bge-small-en-v1.5 (CLS pooling).
Override with --model and --pooling to test other encoders.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OUT = ROOT / "logs/transition/embeddings.npz"


def encode_normalize(model, tokenizer, texts, pooling="cls",
                     batch_size=32, device="cuda", max_len=512, query_prefix=""):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embed", mininterval=2.0):
        batch = [query_prefix + t for t in texts[i : i + batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        if pooling == "cls":
            v = out.last_hidden_state[:, 0]
        elif pooling == "mean":
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        elif pooling == "last":
            # last non-pad token (used by Qwen3-Embedding)
            attn = enc["attention_mask"]
            seq_lens = attn.sum(dim=1) - 1
            v = out.last_hidden_state[torch.arange(out.last_hidden_state.size(0),
                                                   device=out.last_hidden_state.device),
                                     seq_lens]
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        v = F.normalize(v, p=2, dim=1)
        embs.append(v.cpu())
    return torch.cat(embs, dim=0).numpy()


def main():
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pooling", default="cls", choices=["cls", "mean", "last"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--query-prefix", default="",
                    help="prefix prepended to goal text (e.g. 'query: ' for E5)")
    ap.add_argument("--passage-prefix", default="",
                    help="prefix prepended to pattern text (e.g. 'passage: ' for E5)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embed] device={device}, model={args.model}, pooling={args.pooling}")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(device).eval()
    OUT = Path(args.out)

    # patterns
    reg = json.load(open(ROOT / "logs/shared_pattern_registry.json"))
    pids = list(reg.keys())
    pat_texts = []
    for pid in pids:
        p = reg[pid]
        text = p.get("representative_template") or ""
        if not text:
            text = json.dumps(p.get("schema", {}))[:200]
        pat_texts.append(text[:1500])
    print(f"[embed] {len(pat_texts)} patterns")
    pat_emb = encode_normalize(mdl, tok, pat_texts, pooling=args.pooling,
                                device=device, query_prefix=args.passage_prefix)

    # advbench goals
    goals = json.load(open(ROOT / "data/advbench.json"))
    goal_ids = [g["id"] for g in goals]
    goal_texts = [g["goal"] for g in goals]
    print(f"[embed] {len(goal_texts)} goals")
    goal_emb = encode_normalize(mdl, tok, goal_texts, pooling=args.pooling,
                                 device=device, query_prefix=args.query_prefix)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT,
             pattern_ids=np.array(pids),
             pattern_emb=pat_emb,
             goal_ids=np.array(goal_ids),
             goal_emb=goal_emb)
    print(f"[embed] saved -> {OUT}  ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
