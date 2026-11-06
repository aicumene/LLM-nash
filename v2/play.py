"""play.py — the games of the Nash paper against a local engine, logging everything the hidden-state replay needs.
Engines (--api):
  messages  Mjölnir (Anthropic Messages API, default http://127.0.0.1:8097): text, stop reason, usage; the generated
            token ids through /tokenize and the rendered prompt through /apply-template when the server gives them.
  openai    vLLM (chat completions, http://127.0.0.1:8000): seed, per-token logprobs and token ids as well.
Output: one JSON per (game, condition, seed) in --out: {"meta": …, "rounds": [...], "calls": [...]}; a file that exists is
skipped (resume), --force rewrites it.
Usage:
  python play.py pbeauty --cond p0.667 | p1.2 | surrogate_p0.667            [--seeds 0-4]
  python play.py pd      --cond loaded | neutral | script_allC | script_single_early | script_single_mid | script_single_late | script_random35
  python play.py braess  --cond easy_noTG | easy_TG | hard_noTG | hard_TG | experience_TG | surrogate_hard_TG
  python play.py smoke                                                        (one call of each kind, shapes printed)
Temperatures as in the original scripts: PD 0.7, p-beauty 0.7 / top-p 0.95, Braess 0.4; everything else = the engine's defaults."""
from __future__ import annotations
import os, sys, json, time, argparse, statistics, random, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import games as G

THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class Engine:
    def __init__(self, api, base, model, workers, tap=None):
        self.api, self.base, self.model, self.workers, self.tap = api, base.rstrip("/"), model, workers, tap
        if not model:
            d = post_get(f"{self.base}/v1/models"); self.model = d["data"][0]["id"]
        self.lock_calls = []

    def chat(self, messages, *, max_tokens, temperature, top_p=None, seed=None, retries=3):
        for attempt in range(retries):
            try:
                t0 = time.time()
                out = self._messages(messages, max_tokens, temperature, top_p, seed) if self.api == "messages" else \
                      self._openai(messages, max_tokens, temperature, top_p, seed)
                out["latency_s"] = round(time.time() - t0, 2); return out
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1: raise
                time.sleep(3 * (attempt + 1))

    def _messages(self, messages, max_tokens, temperature, top_p, seed=None):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        body = {"model": self.model, "system": system, "messages": rest, "max_tokens": max_tokens, "temperature": temperature, **THINK_OFF}
        if top_p is not None: body["top_p"] = top_p
        if seed is not None: body["seed"] = seed                              # honoured by the tap build of the server
        if self.tap: body["tap"] = self.tap
        d = post(f"{self.base}/v1/messages", body)
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        out = dict(text=text, stop_reason=d.get("stop_reason"), usage=d.get("usage"), token_ids=None, logprobs=None, prompt_text=None, tap=d.get("tap"))
        try:                                                                  # the server's own tokenisation of the reply
            t = post(f"{self.base}/tokenize", {"content": text, "add_special": False, "with_pieces": False}, timeout=60)
            toks = t.get("tokens", t) if isinstance(t, dict) else t
            out["token_ids"] = [x["id"] if isinstance(x, dict) else x for x in toks]
        except Exception as e:  # noqa: BLE001
            out["tokenize_error"] = f"{type(e).__name__}: {e}"[:120]
        try:                                                                  # the rendered prompt, for the replay
            p = post(f"{self.base}/apply-template", body, timeout=60)
            out["prompt_text"] = p.get("prompt") if isinstance(p, dict) else p
        except Exception as e:  # noqa: BLE001
            out["template_error"] = f"{type(e).__name__}: {e}"[:120]
        return out

    def _openai(self, messages, max_tokens, temperature, top_p, seed):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "logprobs": True, "return_tokens_as_token_ids": True, **THINK_OFF}
        if top_p is not None: body["top_p"] = top_p
        if seed is not None: body["seed"] = seed
        d = post(f"{self.base}/v1/chat/completions", body)
        ch = d["choices"][0]; text = ch["message"].get("content") or ""
        lp = (ch.get("logprobs") or {}).get("content") or []
        ids = []
        for x in lp:
            tok = x.get("token", "")
            ids.append(int(tok.split(":")[1]) if tok.startswith("token_id:") else tok)
        return dict(text=text, stop_reason=ch.get("finish_reason"), usage=d.get("usage"), token_ids=ids or None,
                    logprobs=[x.get("logprob") for x in lp] or None, prompt_text=None)


def post_get(url):
    with urllib.request.urlopen(url, timeout=30) as r: return json.load(r)


class Log:
    """Collects the calls and rounds of one game and writes the JSON."""
    def __init__(self, path, meta): self.path, self.meta, self.calls, self.rounds = path, meta, [], []
    def call(self, eng, messages, *, round_, agent, me, max_tokens, temperature, top_p=None, seed=None, parse, extra=None):
        out = eng.chat(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed)
        parsed = parse(out["text"])
        rec = dict(i=len(self.calls), round=round_, agent=agent, me=me, messages=messages, parsed=parsed, seed=seed,
                   temperature=temperature, top_p=top_p, **out, extra=extra or {})
        self.calls.append(rec); return parsed
    def save(self):
        json.dump(dict(meta=self.meta, rounds=self.rounds, calls=self.calls), open(self.path, "w"), indent=1)


# ======================================================================= p-beauty
def play_pbeauty(eng, cond, seed, log, N=5, R=6):
    p = {"p0.667": 2 / 3, "p1.2": 1.2}[cond.replace("surrogate_", "")]; surrogate = cond.startswith("surrogate_")
    fake = G.pb_surrogate_others(N - 1, R, seed) if surrogate else None
    history = []
    for rd in range(R):
        agents = [0] if surrogate else list(range(N))
        def one(a):
            return log.call(eng, G.pb_messages(N, p, history), round_=rd, agent=a, me=str(a + 1), max_tokens=220, temperature=0.7, top_p=0.95,
                            seed=seed * 100000 + rd * 100 + a, parse=G.pb_parse)
        with ThreadPoolExecutor(eng.workers) as ex: picks = list(ex.map(one, agents))
        picks = [round(x, 1) for x in picks]
        if surrogate: picks = picks + fake[rd]
        history.append(picks); ex_ = G.pb_exploitability(picks, p)
        log.rounds.append(dict(round=rd, picks=picks, real=agents, **ex_))
        print(f"  round {rd + 1}: picks {picks} mean {ex_['mean']:.1f} target {ex_['target']:.1f} expl {ex_['exploitability']:.1f}", flush=True)
    log.meta.update(p=p, N=N, R=R, surrogate=surrogate)


# ======================================================================= Prisoner's Dilemma
def play_pd(eng, cond, seed, log):
    if cond in ("loaded", "neutral"):
        labels, R, script = G.PD_LABELS[cond], 8, None
    else:
        kind = cond[len("script_"):]; labels = G.PD_LABELS["loaded"]; R = 24 if kind == "random35" else 14
        script = G.pd_script(kind, R, seed)
    history = []
    for rd in range(R):
        def one(me):
            return log.call(eng, G.pd_messages(labels, history, me, R), round_=rd, agent=int(me) - 1, me=me, max_tokens=160, temperature=0.7,
                            seed=seed * 100000 + rd * 100 + int(me), parse=lambda t: G.pd_parse(t, labels))
        if script is None:
            with ThreadPoolExecutor(2) as ex: m1, m2 = list(ex.map(one, ["1", "2"]))
        else:
            m1, m2 = one("1"), script[rd]
        history.append((m1, m2)); ex_ = G.pd_exploitability(m1, m2)
        log.rounds.append(dict(round=rd, moves=[m1, m2], payoff=list(G.PAY[(m1, m2)]), **ex_))
        print(f"  round {rd + 1}: {m1}{m2} payoff {G.PAY[(m1, m2)]} expl {ex_['exploitability']}", flush=True)
    log.meta.update(labels=labels, R=R, script=script, history_labels=True)


# ======================================================================= Braess
def play_braess(eng, cond, seed, log, N=6):
    with_tg = cond.endswith("_TG"); mode = cond.split("_")[0]
    if mode in ("easy", "hard"):
        routes = ["R2"] * N; passes = 4
        for ps in range(passes):
            for i in range(N):
                others = routes[:i] + routes[i + 1:]
                msgs, opt = (G.braess_messages_easy if mode == "easy" else G.braess_messages_hard)(others, with_tg)
                routes[i] = log.call(eng, msgs, round_=ps, agent=i, me=str(i + 1), max_tokens=120 if mode == "easy" else 220, temperature=0.4,
                                     seed=seed * 100000 + ps * 100 + i, parse=lambda t: G.braess_parse(t, with_tg), extra=dict(options=opt, others=others))
            ex_ = G.braess_exploitability(routes, with_tg)
            log.rounds.append(dict(round=ps, routes=list(routes), **ex_))
            print(f"  pass {ps + 1}: {ex_['counts']} avg {ex_['avg']:.1f} system {ex_['system']} expl {ex_['exploitability']}", flush=True)
        log.meta.update(mode=mode, with_tg=with_tg, passes=passes, N=N)
    elif mode == "experience":
        R = 20; hist = [[] for _ in range(N)]
        for rd in range(R):
            def one(i):
                return log.call(eng, G.braess_messages_experience(hist[i]), round_=rd, agent=i, me=str(i + 1), max_tokens=220, temperature=0.4,
                                seed=seed * 100000 + rd * 100 + i, parse=lambda t: G.braess_parse(t, True))
            with ThreadPoolExecutor(eng.workers) as ex: routes = list(ex.map(one, range(N)))
            costs = G.route_costs(routes, True)
            for i in range(N): hist[i].append((routes[i], float(costs[routes[i]])))
            ex_ = G.braess_exploitability(routes, True)
            log.rounds.append(dict(round=rd, routes=list(routes), **ex_))
            print(f"  day {rd + 1}: {ex_['counts']} avg {ex_['avg']:.1f} expl {ex_['exploitability']}", flush=True)
        log.meta.update(mode=mode, with_tg=True, R=R, N=N)
    elif mode == "surrogate":                                                 # independent one-shot decisions, fabricated others
        K = 24; regrets = []
        for k in range(K):
            others = G.braess_surrogate_others(N - 1, with_tg, seed * 1000 + k)
            msgs, opt = G.braess_messages_hard(others, with_tg)
            r = log.call(eng, msgs, round_=k, agent=0, me="1", max_tokens=220, temperature=0.4, seed=seed * 100000 + k,
                         parse=lambda t: G.braess_parse(t, with_tg), extra=dict(options=opt, others=others))
            reg = opt[r] - min(opt.values()); regrets.append(reg)
            log.rounds.append(dict(round=k, route=r, options=opt, regret=reg, exploitability=reg, normalised=reg / 100.0, others=others))
            print(f"  decision {k + 1}: others {others} options {opt} -> {r} regret {reg}", flush=True)
        log.meta.update(mode=mode, with_tg=with_tg, K=K, mean_regret=statistics.mean(regrets))
    else:
        raise ValueError(cond)


def seeds_of(s):
    if "-" in s: a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def smoke(eng):
    for name, msgs, mt, temp, tp in [("pd", G.pd_messages(G.PD_LABELS["loaded"], [], "1", 8), 160, 0.7, None),
                                     ("pbeauty", G.pb_messages(5, 2 / 3, []), 220, 0.7, 0.95),
                                     ("braess_hard", G.braess_messages_hard(["R2"] * 5, True)[0], 220, 0.4, None)]:
        out = eng.chat(msgs, max_tokens=mt, temperature=temp, top_p=tp, seed=1)
        print(f"=== {name}: {out['latency_s']} s, stop {out['stop_reason']}, usage {out['usage']}")
        print("   text:", out["text"][:300].replace("\n", " | "))
        print("   token_ids:", None if out["token_ids"] is None else (len(out["token_ids"]), out["token_ids"][:8]), "| prompt_text:",
              None if not out.get("prompt_text") else repr(out["prompt_text"][:160]), out.get("tokenize_error"), out.get("template_error"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("game"); ap.add_argument("--cond", default=""); ap.add_argument("--seeds", default="0")
    ap.add_argument("--api", default="messages"); ap.add_argument("--base", default=None); ap.add_argument("--model", default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "new_runs")); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--force", action="store_true")
    ap.add_argument("--tap", action="store_true", help="ask the server for the internal states (the tap build)")
    ap.add_argument("--tap_layers", default="6,11,18,30,42,47,-1"); ap.add_argument("--experts", action="store_true")
    A = ap.parse_args()
    base = A.base or ("http://127.0.0.1:8097" if A.api == "messages" else "http://127.0.0.1:8000")
    tap = {"layers": [int(x) for x in A.tap_layers.split(",")], "experts": A.experts} if A.tap else None
    eng = Engine(A.api, base, A.model, A.workers, tap)
    if A.game == "smoke": smoke(eng); sys.exit(0)
    os.makedirs(A.out, exist_ok=True)
    for seed in seeds_of(A.seeds):
        path = f"{A.out}/{A.game}_{A.cond}_s{seed}.json"
        if os.path.exists(path) and not A.force: print(f"skip {path} (exists)"); continue
        random.seed(seed); t0 = time.time()
        meta = dict(game=A.game, cond=A.cond, seed=seed, api=A.api, base=base, model=eng.model, tap=tap, started=time.strftime("%Y-%m-%d %H:%M:%S"))
        log = Log(path, meta); print(f"=== {A.game} {A.cond} seed {seed} ({eng.model} via {A.api})", flush=True)
        {"pbeauty": play_pbeauty, "pd": play_pd, "braess": play_braess}[A.game](eng, A.cond, seed, log)
        log.meta.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0), n_calls=len(log.calls)); log.save()
        print(f"  wrote {path}: {len(log.calls)} calls in {time.time() - t0:.0f} s", flush=True)
