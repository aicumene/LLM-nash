"""Readers of the internal states: the Mjölnir tap (`<tap_dir>/<id>/`) and the llama.cpp dump (`nash_dump` directories).
Both give, per transcript, a dict layer -> float32 array [n_steps, hc, n_embd] (hc = the hyper-connection streams)
plus the token ids. The tap's step s holds the states after token stepped[s] (s = 0: the last prompt token); the dump's
kept rows are the last K tokens of the transcript, the states after each of them — the same object for the same tokens."""
import json, os, numpy as np


def read_tap(d):
    ix = json.load(open(f"{d}/index.json"))
    L, hc, n, S = ix["layers"], ix["hc"], ix["n_embd"], ix["n_steps"]
    raw = np.fromfile(f"{d}/tap.bin", dtype=np.float16)
    assert raw.size == S * len(L) * hc * n, (raw.size, S, L, hc, n)
    x = raw.reshape(S, len(L), hc, n).astype(np.float32)
    out = {il: x[:, j] for j, il in enumerate(L)}
    experts = None
    if ix.get("experts_used", 0) and os.path.exists(f"{d}/experts.bin"):
        experts = np.fromfile(f"{d}/experts.bin", dtype=np.int32).reshape(S, len(L), ix["experts_used"])
    return dict(states=out, layers=L, stepped=ix["stepped"], out_tokens=ix["out_tokens"], prompt_n=ix["prompt_n"], experts=experts, index=ix)


def read_dump(d):
    """A nash_dump transcript directory: index.tsv (name d streams rows_kept rows_total [type] file) + row files. llama.cpp runs a
    long transcript in micro-batches, each giving its own instance of a tensor (`l_last-6`, `l_last-6#1`, …, in order); the
    instances are concatenated and the last K rows kept (K = the largest rows_kept). `ffn_moe_argsort-N` (i32) gives the
    experts: the first 10 of each row (sorted by probability)."""
    parts, eparts, kept = {}, {}, 0
    for line in open(f"{d}/index.tsv"):
        f = line.rstrip("\n").split("\t")
        name, dim, streams, rk, rt = f[0], int(f[1]), int(f[2]), int(f[3]), int(f[4]); ty, fn = (f[5], f[6]) if len(f) == 7 else ("f16", f[5])
        il = int(name.split("-")[1].split("#")[0])
        if name.startswith("l_last-"):
            raw = np.fromfile(f"{d}/{fn}", dtype=np.float16).astype(np.float32)
            parts.setdefault(il, []).append(raw.reshape(rk, streams, dim // streams)); kept = max(kept, rk)
        elif name.startswith("ffn_moe_argsort-") and ty == "i32":
            raw = np.fromfile(f"{d}/{fn}", dtype=np.int32).reshape(rk, dim); eparts.setdefault(il, []).append(raw[:, :10])
    if os.path.exists(f"{d}/K.txt"): kept = int(open(f"{d}/K.txt").read().strip())        # the dumper's request (instances may be single tokens)
    states = {il: np.concatenate(v, 0)[-kept:] for il, v in parts.items()}
    experts = {il: np.concatenate(v, 0)[-kept:] for il, v in eparts.items()} or None
    tokens = [int(t) for t in open(f"{d}/tokens.txt").read().split()]
    return dict(states=states, layers=sorted(states), tokens=tokens, kept=kept, total=len(tokens), experts=experts)


def compare(tap, dump, layer):
    """The tap's steps against the dump's last rows for the same transcript (row s = the state after the token at position
    prompt_n − 1 + s in both): relative error per step (median, max), cosine (min), and the experts' overlap when both have them."""
    a = tap["states"][layer]; b = dump["states"][layer]; k = min(len(a), len(b))
    a, b = a[-k:].reshape(k, -1), b[-k:].reshape(k, -1)
    err = np.linalg.norm(a - b, axis=1) / (np.linalg.norm(b, axis=1) + 1e-6)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    out = dict(steps=k, rel_err_median=float(np.median(err)), rel_err_max=float(err.max()), err_argmax=int(err.argmax()), cos_min=float(cos.min()))
    if tap.get("experts") is not None and dump.get("experts") and layer in dump["experts"] and layer in tap["layers"]:
        j = tap["layers"].index(layer); ta = tap["experts"][-k:, j]; db = dump["experts"][layer][-k:]
        ov = [len(set(ta[i].tolist()) & set(db[i].tolist())) for i in range(k)]
        out["experts_overlap_mean"] = float(np.mean(ov)); out["steps_with_different_experts"] = int(sum(o < 10 for o in ov))
        out["err_on_same_experts_median"] = float(np.median(err[[o == 10 for o in ov]])) if any(o == 10 for o in ov) else None
    return out


if __name__ == "__main__":
    import sys, tempfile
    # self-test on synthetic data in the tap format
    d = tempfile.mkdtemp(); S, L, hc, n = 5, [6, 47], 4, 8
    x = np.random.default_rng(0).standard_normal((S, len(L), hc, n)).astype(np.float16); x.tofile(f"{d}/tap.bin")
    json.dump(dict(layers=L, hc=hc, n_embd=n, n_steps=S, stepped=list(range(S)), out_tokens=list(range(1, S)), prompt_n=10, experts_used=0), open(f"{d}/index.json", "w"))
    t = read_tap(d); print({k: v.shape for k, v in t["states"].items()}, "ok")
