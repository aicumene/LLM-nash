"""Table 1 of the preprint: what the stored prompts of the second version tell an agent about the other players.
Counts the calls, the messages per call, and any word that would say what the others are; checks that the five agents of a beauty contest
receive identical messages in every round. Standard library only.
    python3 analysis/check_prompts.py"""
import collections, glob, json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = re.compile(r"\b(LLM|language model|AI|artificial|model|Qwen|Gemma|Nemotron|GPT|script|scripted|copy|clone|human|person|bot|machine|same family|identical)\b", re.I)
calls = 0; sizes = collections.Counter(); hits = collections.Counter(); rounds = distinct = 0
for d in ("runs", "runs_tap"):
    for f in glob.glob(os.path.join(ROOT, "v2", "results", d, "*.json")):
        run = json.load(open(f)); name = os.path.basename(f); by_round = collections.defaultdict(set)
        for c in run["calls"]:
            calls += 1; sizes[len(c["messages"])] += 1
            for w in WORDS.findall(" ".join(m["content"] for m in c["messages"])): hits[w.lower()] += 1
            by_round[c["round"]].add(json.dumps(c["messages"], sort_keys=True))
        if name.startswith("pbeauty_p"):
            rounds += len(by_round); distinct += sum(len(v) > 1 for v in by_round.values())
print(f"{calls} calls; messages per call: {dict(sizes)}; words that name the others: {dict(hits) or 'none'}")
print(f"beauty contest: {rounds} rounds, {distinct} of them with different messages across the five agents")
