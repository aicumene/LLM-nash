"""Behavioural summary of the runs in a directory: one line per (game, cond, seed) with the per-round key quantity and exploitability."""
import json, glob, os, sys, statistics
D = sys.argv[1] if len(sys.argv) > 1 else "results/runs"
rows = []
for f in sorted(glob.glob(f"{D}/*.json")):
    d = json.load(open(f)); m = d["meta"]; r = d["rounds"]; g = m["game"]
    if g == "pbeauty": s = " ".join(f"{x['mean']:.1f}/{x['exploitability']:.0f}" for x in r)
    elif g == "pd": s = "".join("".join(x["moves"]) + " " for x in r) + "| expl " + " ".join(str(x["exploitability"]) for x in r)
    elif m.get("mode") == "surrogate": s = f"mean regret {m.get('mean_regret', 0):.1f}; regrets " + " ".join(str(x["regret"]) for x in r)
    else: s = " ".join(f"{x['counts']['R1']}/{x['counts']['R2']}/{x['counts']['R3']}:{x['exploitability']}" for x in r)
    lat = statistics.mean(c.get("latency_s", 0) for c in d["calls"]) if d["calls"] else 0
    print(f"{os.path.basename(f):34s} {m.get('seconds', 0):5d}s {len(d['calls']):4d} calls {lat:4.1f}s/call | {s[:150]}")
