"""TellTale: text-only ambivalence/hesitancy recognition for the BAH dataset.

Submitted entry, 3rd A/H Video Recognition Challenge (11th ABAW Workshop, ECCV 2026).
Private test: Macro-F1 0.7364, AP 0.7940.

Three probability streams over the interview transcript, blended:
  A  multilingual-e5-large, LoRA fine-tuned, video-level MIL (LogSumExp) objective
  B  mDeBERTa-v3-base, LoRA fine-tuned, same MIL objective
  J  Qwen3-14B (4-bit, zero-shot prompt), no training
  p = 0.45 pA + 0.30 pB + 0.25 pJ, decision threshold 0.53

Subcommands: train, judge, oof, blend, predict. See README.md.
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# configuration

DATA_DIR = Path(os.environ.get("BAH_DIR", "data/BAH"))
TEST_DIR = Path(os.environ.get("BAH_TEST_DIR", "data/BAH_test"))
OUT_DIR = Path(os.environ.get("TELLTALE_OUT", "outputs"))

LSE_R = 50
MAX_CHUNKS = 64
N_FOLDS = 5
BLEND_WEIGHTS = (0.45, 0.30, 0.25)   # (A, B, J), selected on out-of-fold predictions
THRESHOLD = 0.53                     # flat-region threshold on out-of-fold predictions

STREAMS = {
    "a": {
        "model": "intfloat/multilingual-e5-large",
        "prefix": "query: ", "pool": "mean", "max_len": 256,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.1, "targets": "all-linear"},
        "chunk_pretrain": {"epochs": 6, "lr": 2e-4, "batch": 16, "patience": 3},
        "mil": {"epochs": 2, "lr": 5e-5, "patience": 1},
        "seeds": [0, 1, 2],
    },
    "b": {
        "model": "microsoft/mdeberta-v3-base",
        "prefix": "", "pool": "cls", "max_len": 128,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
                 "targets": ["query_proj", "key_proj", "value_proj", "dense"]},
        "chunk_pretrain": None,
        "mil": {"epochs": 6, "lr": 2e-5, "patience": 2},
        "seeds": [0, 1],
    },
}

JUDGE_MODEL = "mlx-community/Qwen3-14B-4bit"
JUDGE_SYSTEM = (
    "You are an expert behavioral-psychology rater. You rate interview answers "
    "for AMBIVALENCE/HESITANCY (A/H): the speaker holding conflicting attitudes, "
    "expressing uncertainty about their own answer, hedging, self-contradicting, "
    "or showing reluctance/difficulty committing to a position. You see only the "
    "transcript text. Respond with STRICT JSON only, no prose, no markdown."
)
JUDGE_CUES = ["filler_sound", "filler_word", "hedging", "positive", "negative",
              "correction", "excuse", "fail", "repetition", "stuttering", "pause",
              "slow_speech", "fast_speech", "breath", "laugh", "inconsistency"]


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# data

def sanitize_id(video_path):
    base = os.path.basename(video_path)
    return base[:-4] if base.endswith(".mp4") else base


def participant_of(video_path):
    m = re.search(r"Videos/(\d+)/", video_path)
    return m.group(1) if m else sanitize_id(video_path).split("_")[0]


def load_split(base, name):
    """Parse split/<name>.txt: video-path,label,transcript per line."""
    recs = []
    path = base / "split" / f"{name}.txt"
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split(",")
        vp = parts[0]
        try:
            label = int(parts[1]) if len(parts) > 1 and parts[1] != "" else -1
        except ValueError:
            label = -1
        recs.append({"id": sanitize_id(vp), "path": vp, "pid": participant_of(vp),
                     "label": label,
                     "yml": base / "transcription" / vp / f"{sanitize_id(vp)}.yml"})
    return recs


def read_chunks(rec):
    """Timestamped transcript chunks from the per-video yml."""
    import yaml
    if not rec["yml"].exists():
        return []
    y = yaml.load(open(rec["yml"]), Loader=yaml.UnsafeLoader)
    out = []
    for c in y.get("chunks", []):
        ts = c.get("timestamp") or (0.0, 0.0)
        out.append({"text": c.get("text", "").strip(),
                    "start": float(ts[0]), "end": float(ts[1] or ts[0])})
    return out


def _hms_to_sec(v):
    if isinstance(v, (int, float)):
        return float(v)
    parts = str(v).strip().split(":")
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))


def load_ah_segments():
    """Per-video annotated A/H time ranges, cached to json after first parse."""
    cache = OUT_DIR / "ah_segments.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import yaml
    raw = yaml.load(open(DATA_DIR / "video_annotation_transcript.yaml"),
                    Loader=yaml.UnsafeLoader)
    idx = {}
    for vpath, rec in raw.items():
        idx[sanitize_id(vpath)] = [[_hms_to_sec(a), _hms_to_sec(b)]
                                   for a, b in (rec.get("time_detailed_ah") or [])]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(idx))
    return idx


def build_labeled_pool():
    """All labeled videos (train+val+test splits pooled) with participant-grouped
    5-fold assignment. Fold assignment matches the challenge entry: GroupKFold
    over the pooled chunk sequence, grouped by participant."""
    from sklearn.model_selection import GroupKFold

    recs = []
    for split in ("train", "val", "test"):
        recs += load_split(DATA_DIR, split)
    seen, pool = set(), []
    for r in recs:
        if r["id"] not in seen:
            seen.add(r["id"])
            pool.append(r)

    segs = load_ah_segments()
    rows = []
    for r in pool:
        chunks = read_chunks(r)
        if not chunks:
            chunks = [{"text": "", "start": 0.0, "end": 0.0}]
        for i, c in enumerate(chunks):
            overlap = any(c["start"] < e and c["end"] > s
                          for s, e in segs.get(r["id"], []))
            rows.append({"video_id": r["id"], "pid": r["pid"], "label": r["label"],
                         "chunk_idx": i, "text": c["text"],
                         "chunk_label": int(r["label"] == 1 and overlap)})
    chunks_df = pd.DataFrame(rows)

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold = np.empty(len(chunks_df), dtype=int)
    for f, (_, te) in enumerate(gkf.split(chunks_df, chunks_df["label"],
                                          chunks_df["pid"])):
        fold[te] = f
    chunks_df["fold"] = fold
    return chunks_df


def load_test_pool():
    recs = load_split(TEST_DIR, "test")
    rows = []
    for r in recs:
        chunks = read_chunks(r) or [{"text": "", "start": 0.0, "end": 0.0}]
        for i, c in enumerate(chunks):
            rows.append({"video_id": r["id"], "chunk_idx": i, "text": c["text"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# models

def build_model(spec):
    import torch.nn as nn
    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model

    class ChunkClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            base = AutoModel.from_pretrained(spec["model"]).float()
            cfg = LoraConfig(r=spec["lora"]["r"], lora_alpha=spec["lora"]["alpha"],
                             lora_dropout=spec["lora"]["dropout"],
                             target_modules=spec["lora"]["targets"], bias="none")
            self.encoder = get_peft_model(base, cfg)
            dim = base.config.hidden_size
            self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.2),
                                      nn.Linear(dim, 1))

        def forward(self, enc):
            out = self.encoder(**enc).last_hidden_state
            if spec["pool"] == "cls":
                emb = out[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).float()
                emb = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return self.head(emb).squeeze(-1)

    return ChunkClassifier()


def encode(tokenizer, spec, texts, device):
    enc = tokenizer([spec["prefix"] + t for t in texts], return_tensors="pt",
                    padding=True, truncation=True, max_length=spec["max_len"])
    return {k: v.to(device) for k, v in enc.items()}


def lse_pool(logits, mask):
    import torch
    z = torch.where(mask, logits * LSE_R, torch.tensor(float("-inf"),
                    device=logits.device))
    n = mask.sum(dim=1).clamp(min=1).float()
    return (torch.logsumexp(z, dim=1) - torch.log(n)) / LSE_R


def video_logits(model, tokenizer, spec, vid_texts, video_ids, device):
    import torch
    per_video = [model(encode(tokenizer, spec, vid_texts[v], device))
                 for v in video_ids]
    T = max(l.shape[0] for l in per_video)
    logits = torch.zeros(len(per_video), T, device=device)
    mask = torch.zeros(len(per_video), T, dtype=torch.bool, device=device)
    for i, l in enumerate(per_video):
        logits[i, :l.shape[0]] = l
        mask[i, :l.shape[0]] = True
    return logits, mask


def chunk_map(df):
    return {v: list(g.sort_values("chunk_idx")["text"])[:MAX_CHUNKS]
            for v, g in df.groupby("video_id")}


def macro_f1(y, pred):
    from sklearn.metrics import f1_score
    return f1_score(y, pred, average="macro")


def best_threshold(y, p):
    ts = np.arange(0.01, 1.0, 0.01)
    f1s = [macro_f1(y, (p >= t).astype(int)) for t in ts]
    i = int(np.argmax(f1s))
    return float(ts[i]), float(f1s[i])


# ---------------------------------------------------------------------------
# training

def grouped_early_stop_split(df, seed):
    from sklearn.model_selection import GroupShuffleSplit
    vids = df["video_id"].unique()
    pids = df.drop_duplicates("video_id").set_index("video_id").loc[vids, "pid"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    itr, iva = next(gss.split(vids, np.zeros(len(vids)), pids))
    return set(vids[itr]), set(vids[iva])


def train_chunk_stage(model, tokenizer, spec, tr_df, va_df, device):
    import torch
    from tqdm import tqdm
    cfg = spec["chunk_pretrain"]
    dev = torch.device(device)
    model.to(dev)
    texts = tr_df["text"].tolist()
    y = tr_df["chunk_label"].values.astype("float32")
    pos = max(y.sum(), 1); neg = max(len(y) - y.sum(), 1)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=dev))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"])
    best = {"f1": -1, "state": None, "ep": -1}
    for ep in range(cfg["epochs"]):
        model.train()
        idx = np.random.permutation(len(texts))
        for i in tqdm(range(0, len(texts), cfg["batch"]),
                      desc=f"chunk ep{ep + 1}/{cfg['epochs']}"):
            b = idx[i:i + cfg["batch"]]
            enc = encode(tokenizer, spec, [texts[j] for j in b], dev)
            loss = lossf(model(enc), torch.tensor(y[b], device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
        f1 = eval_video_f1(model, tokenizer, spec, va_df, device)
        print(f"  chunk stage ep{ep + 1}: val Macro-F1={f1:.4f}", flush=True)
        if f1 > best["f1"]:
            best = {"f1": f1, "ep": ep,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
        elif ep - best["ep"] >= cfg["patience"]:
            break
    if best["state"]:
        model.load_state_dict(best["state"], strict=False)
    return model


def train_mil_stage(model, tokenizer, spec, tr_df, va_df, vid2label, device):
    import torch
    from tqdm import tqdm
    cfg = spec["mil"]
    dev = torch.device(device)
    model.to(dev)
    vid_texts = chunk_map(tr_df)
    tr_vids = list(vid_texts)
    y = np.array([vid2label[v] for v in tr_vids], dtype="float32")
    pos = max(y.sum(), 1); neg = max(len(y) - y.sum(), 1)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=dev))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"])
    best = {"f1": -1, "state": None, "ep": -1}
    for ep in range(cfg["epochs"]):
        model.train()
        idx = np.random.permutation(len(tr_vids))
        for i in tqdm(range(0, len(tr_vids), 4), desc=f"mil ep{ep + 1}/{cfg['epochs']}"):
            batch = [tr_vids[j] for j in idx[i:i + 4]]
            logits, mask = video_logits(model, tokenizer, spec, vid_texts, batch, dev)
            vlogit = lse_pool(logits, mask)
            yb = torch.tensor([vid2label[v] for v in batch], dtype=torch.float32,
                              device=dev)
            loss = lossf(vlogit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        f1 = eval_video_f1(model, tokenizer, spec, va_df, device)
        print(f"  mil stage ep{ep + 1}: val Macro-F1={f1:.4f}", flush=True)
        if f1 > best["f1"]:
            best = {"f1": f1, "ep": ep,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
        elif ep - best["ep"] >= cfg["patience"]:
            break
    if best["state"]:
        model.load_state_dict(best["state"], strict=False)
    return model


def predict_videos(model, tokenizer, spec, df, device):
    import torch
    dev = torch.device(device)
    model.to(dev)
    model.eval()
    vid_texts = chunk_map(df)
    vids = sorted(vid_texts)
    out = {}
    with torch.no_grad():
        for i in range(0, len(vids), 4):
            batch = vids[i:i + 4]
            logits, mask = video_logits(model, tokenizer, spec, vid_texts, batch, dev)
            prob = torch.sigmoid(lse_pool(logits, mask)).cpu().numpy()
            out.update(dict(zip(batch, map(float, prob))))
    return out


def eval_video_f1(model, tokenizer, spec, df, device):
    probs = predict_videos(model, tokenizer, spec, df, device)
    vid2label = dict(df.drop_duplicates("video_id")[["video_id", "label"]].values)
    vids = sorted(probs)
    y = np.array([vid2label[v] for v in vids])
    _, f1 = best_threshold(y, np.array([probs[v] for v in vids]))
    return f1


def save_trainable(model, path, meta):
    import torch
    path.mkdir(parents=True, exist_ok=True)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}
    torch.save(sd, path / "trainable.pt")
    (path / "meta.json").write_text(json.dumps(meta))


def load_trainable(spec, path):
    import torch
    model = build_model(spec)
    model.load_state_dict(torch.load(path / "trainable.pt", map_location="cpu"),
                          strict=False)
    return model


def cmd_train(args):
    from transformers import AutoTokenizer
    spec = STREAMS[args.stream]
    tokenizer = AutoTokenizer.from_pretrained(spec["model"])
    chunks = build_labeled_pool()
    vid2label = dict(chunks.drop_duplicates("video_id")[["video_id", "label"]].values)

    folds = [args.fold] if args.fold is not None else range(N_FOLDS)
    seeds = [args.seed] if args.seed is not None else spec["seeds"]
    for fold in folds:
        for seed in seeds:
            ckpt = OUT_DIR / f"stream_{args.stream}" / f"fold{fold}_seed{seed}"
            probs_path = ckpt / "oof_probs.csv"
            if probs_path.exists():
                print(f"[train] stream {args.stream} fold {fold} seed {seed}: cached")
                continue
            set_seed(seed)
            tr_all = chunks[chunks["fold"] != fold]
            te = chunks[chunks["fold"] == fold]
            tr_vids, va_vids = grouped_early_stop_split(tr_all, seed)
            tr = tr_all[tr_all["video_id"].isin(tr_vids)]
            va = tr_all[tr_all["video_id"].isin(va_vids)]

            model = build_model(spec)
            if spec["chunk_pretrain"]:
                model = train_chunk_stage(model, tokenizer, spec, tr, va, args.device)
            model = train_mil_stage(model, tokenizer, spec, tr, va, vid2label,
                                    args.device)
            save_trainable(model, ckpt, {"stream": args.stream, "fold": fold,
                                         "seed": seed})
            probs = predict_videos(model, tokenizer, spec, te, args.device)
            pd.DataFrame({"video_id": list(probs), "prob": list(probs.values()),
                          "fold": fold}).to_csv(probs_path, index=False)
            print(f"[train] stream {args.stream} fold {fold} seed {seed}: done")


# ---------------------------------------------------------------------------
# LLM judge

def judge_prompt(transcript):
    cues = "\n".join(f'- "{c}"' for c in JUDGE_CUES)
    return (f"Transcript of one interview answer:\n---\n{transcript}\n---\n"
            f"Rate the transcript. Output strict JSON with exactly these keys:\n"
            f'{{"score": <integer 0-100, overall ambivalence/hesitancy>,\n'
            f' "cues": {{<one boolean per cue name below>}},\n'
            f' "rationale": <one short sentence>}}\n\n'
            f"Cue names (use exactly these as JSON keys):\n{cues}\n")


def parse_judge(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        s = int(d["score"])
        assert 0 <= s <= 100
        return s
    except Exception:
        return None


def cmd_judge(args):
    from mlx_lm import load, generate
    from tqdm import tqdm

    if args.split == "labeled":
        chunks = build_labeled_pool()
        out_path = OUT_DIR / "judge_labeled.jsonl"
    else:
        chunks = load_test_pool()
        out_path = OUT_DIR / "judge_test.jsonl"

    done = set()
    if out_path.exists():
        done = {json.loads(l)["video_id"] for l in open(out_path)}
    model, tok = load(JUDGE_MODEL)
    vids = [v for v in sorted(chunks["video_id"].unique()) if v not in done]
    with open(out_path, "a") as f:
        for vid in tqdm(vids, desc="judge"):
            rows = chunks[chunks["video_id"] == vid].sort_values("chunk_idx")
            transcript = " ".join(rows["text"])[:8000]
            msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": judge_prompt(transcript)}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             enable_thinking=False)
            score = parse_judge(generate(model, tok, prompt=prompt, max_tokens=400,
                                         verbose=False))
            if score is None:
                score = parse_judge(generate(model, tok, prompt=prompt,
                                             max_tokens=400, verbose=False)) or 50
            f.write(json.dumps({"video_id": vid, "score": score}) + "\n")
            f.flush()
    print(f"[judge] wrote {out_path}")


# ---------------------------------------------------------------------------
# aggregation, blending, prediction

def stream_oof(stream):
    spec = STREAMS[stream]
    parts = []
    for fold in range(N_FOLDS):
        seed_probs = []
        for seed in spec["seeds"]:
            p = OUT_DIR / f"stream_{stream}" / f"fold{fold}_seed{seed}" / "oof_probs.csv"
            if not p.exists():
                sys.exit(f"missing {p}; run: python telltale.py train --stream {stream}")
            seed_probs.append(pd.read_csv(p).set_index("video_id")["prob"])
        avg = pd.concat(seed_probs, axis=1).mean(axis=1)
        parts.append(avg)
    return pd.concat(parts)


def judge_scores(path):
    return pd.Series({json.loads(l)["video_id"]: json.loads(l)["score"] / 100.0
                      for l in open(path)})


def assemble_oof():
    chunks = build_labeled_pool()
    meta = chunks.drop_duplicates("video_id")[["video_id", "label", "fold"]]
    df = meta.set_index("video_id")
    df["p_a"] = stream_oof("a")
    df["p_b"] = stream_oof("b")
    df["p_j"] = judge_scores(OUT_DIR / "judge_labeled.jsonl")
    if df[["p_a", "p_b", "p_j"]].isna().any().any():
        sys.exit("missing OOF probabilities for some videos; "
                 "finish train/judge runs first")
    return df.reset_index()


def cmd_oof(args):
    df = assemble_oof()
    y = df["label"].values
    for col, name in [("p_a", "stream A"), ("p_b", "stream B"), ("p_j", "judge")]:
        thr, f1 = best_threshold(y, df[col].values)
        print(f"{name:10s} OOF Macro-F1={f1:.4f} (thr={thr:.2f})")
    p = (BLEND_WEIGHTS[0] * df["p_a"] + BLEND_WEIGHTS[1] * df["p_b"]
         + BLEND_WEIGHTS[2] * df["p_j"]).values
    f1 = macro_f1(y, (p >= THRESHOLD).astype(int))
    print(f"{'blend':10s} OOF Macro-F1={f1:.4f} (weights={BLEND_WEIGHTS}, "
          f"thr={THRESHOLD})")


def cmd_blend(args):
    """Re-derive blend weights and threshold from OOF predictions."""
    df = assemble_oof()
    y = df["label"].values
    S = (df["p_a"].values, df["p_b"].values, df["p_j"].values)
    best_w, best_f1 = None, -1.0
    ticks = 20
    for a in range(ticks + 1):
        for b in range(ticks + 1 - a):
            w = (a / ticks, b / ticks, (ticks - a - b) / ticks)
            p = w[0] * S[0] + w[1] * S[1] + w[2] * S[2]
            _, f1 = best_threshold(y, p)
            if f1 > best_f1:
                best_w, best_f1 = w, f1
    p = best_w[0] * S[0] + best_w[1] * S[1] + best_w[2] * S[2]
    ts = np.arange(0.30, 0.70 + 1e-9, 0.005)
    f1s = np.array([macro_f1(y, (p >= t).astype(int)) for t in ts])
    ok = f1s >= f1s.max() - 0.005
    best_len, best_mid, i = 0, float(ts[f1s.argmax()]), 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if j - i > best_len:
                best_len, best_mid = j - i, float(ts[(i + j) // 2])
            i = j + 1
        else:
            i += 1
    print(f"weights={best_w}  flat threshold={best_mid:.3f}  OOF Macro-F1={best_f1:.4f}")


def cmd_predict(args):
    from transformers import AutoTokenizer
    test_chunks = load_test_pool()

    stream_probs = {}
    for stream in ("a", "b"):
        spec = STREAMS[stream]
        tokenizer = AutoTokenizer.from_pretrained(spec["model"])
        all_probs = []
        for fold in range(N_FOLDS):
            for seed in spec["seeds"]:
                ckpt = OUT_DIR / f"stream_{stream}" / f"fold{fold}_seed{seed}"
                model = load_trainable(spec, ckpt)
                all_probs.append(predict_videos(model, tokenizer, spec, test_chunks,
                                                args.device))
                del model
        vids = sorted(all_probs[0])
        stream_probs[stream] = {v: float(np.mean([p[v] for p in all_probs]))
                                for v in vids}

    pj = judge_scores(OUT_DIR / "judge_test.jsonl")

    order = []
    with open(TEST_DIR / "video.csv") as f:
        next(f)
        for line in f:
            order.append(line.rstrip("\n").split(",")[0])

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        for vp in order:
            vid = sanitize_id(vp)
            p1 = (BLEND_WEIGHTS[0] * stream_probs["a"][vid]
                  + BLEND_WEIGHTS[1] * stream_probs["b"][vid]
                  + BLEND_WEIGHTS[2] * pj[vid])
            p1 = round(p1, 4)
            pred = int(p1 >= THRESHOLD)
            f.write(f"{vp},{round(1 - p1, 4):.4f},{p1:.4f},{pred}\n")
    print(f"[predict] wrote {out_path} ({len(order)} rows)")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train one encoder stream (all folds/seeds)")
    t.add_argument("--stream", choices=["a", "b"], required=True)
    t.add_argument("--fold", type=int)
    t.add_argument("--seed", type=int)
    t.add_argument("--device", default="mps")
    t.set_defaults(fn=cmd_train)

    j = sub.add_parser("judge", help="run the zero-shot LLM judge (Apple Silicon)")
    j.add_argument("--split", choices=["labeled", "test"], required=True)
    j.set_defaults(fn=cmd_judge)

    o = sub.add_parser("oof", help="report out-of-fold scores per stream and blend")
    o.set_defaults(fn=cmd_oof)

    b = sub.add_parser("blend", help="re-derive blend weights/threshold from OOF")
    b.set_defaults(fn=cmd_blend)

    p = sub.add_parser("predict", help="write the test-set submission file")
    p.add_argument("--out", default="submission.txt")
    p.add_argument("--device", default="mps")
    p.set_defaults(fn=cmd_predict)

    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
