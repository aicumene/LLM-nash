# LLM-nash

Games, transcripts and analysis for **When Do Language Models Reach Nash? Heterogeneous Cross-Family LLM Agents in Classic Economic Games**
(Alexandra Bernadotte and Ivan Menshikov, AiCumene). Preprint: [research.aicumene.com/papers/llm-nash](https://research.aicumene.com/papers/llm-nash/), DOI [10.2139/ssrn.7637178](https://doi.org/10.2139/ssrn.7637178).

Language-model agents play the p-beauty contest, the iterated Prisoner's Dilemma and a Braess routing game. They reach the equilibrium when it reduces to choosing the smallest of given payoffs, and miss it when it takes reasoning. This repository accompanies the second version of the preprint: every number below can be recomputed from the stored games without running a model.

## Check the numbers

```
pip install -r requirements.txt
python3 analysis/stats_v2.py            # exploitability, alpha and gamma, betrayal reaction, routing: writes analysis/stats_v2.json
python3 analysis/conditional_counts.py  # alpha, gamma and cooperation by own and opponent's last move (Section 4.2)
python3 analysis/check_prompts.py       # what the prompts tell an agent about the others (Table 1)
python3 analysis/equilibrium_curve.py   # the memory-one equilibria for payoffs 5,3,1,0 (Appendix B): writes analysis/numbers_curve.tex
python3 analysis/equilibrium_set_check.py  # the equilibrium sets of both payoff matrices by best-response search, with simulation
python3 analysis/make_figures.py        # the figures of the second version, into analysis/figs/
```

## Where things are

| In the preprint | Code | Data |
|---|---|---|
| 4.1 Beauty contest | `v1/pbeauty.py` | `v1/logs/pbeauty_*.txt` |
| 4.2 Prisoner's Dilemma, early games | `v1/pd.py`, `v1/xgame.py`, `v1/xgame_verbose.py` | `v1/logs/pd_selfplay.txt`, `xgame.txt`, `xgame_verbose.txt` |
| 4.2 Memory-one batch, scripted opponents | `v1/nash_recheck.py`, `v1/nash_recheck2.py` | `v1/nash_recheck_out.json`, `v1/nash_recheck2_out.json` |
| 4.3 Braess game | `v1/braess.py`, `v1/braess2.py hard` | `v1/logs/braess_explicit.txt`, `braess_formulas.txt` |
| 4.4 Mixed-family routing | `v1/braess2.py cross`, `v1/braess3.py`, `v1/braess4.py` | `v1/logs/braess_mixed_*.txt` |
| Experience-only routing, route logit | `v1/braess_experience.py`, `v1/braess_qre.py` | `v1/braess_experience_out.json`, `v1/braess_qre_out.json` |
| 4.6 Second engine and model version | `v2/games.py`, `v2/play.py`, `v2/summarize.py` | `v2/results/runs/`, `v2/results/runs_tap/` |
| 4.7 Representation space | `v2/analyze_runs.py`, `measure.py`, `export_states.py`, `probe.py`, `boid_*.py`, `signature_lm.py`, `mcssa.py`, `tapio.py` | `v2/results/measures*/`, `states_*/`, `boid_*/` |
| Appendix B | `analysis/equilibrium_curve.py` | |
| Table 1, Appendix D | `v2/games.py`, `analysis/check_prompts.py` | `calls` in `v2/results/runs*/` |

`v2/RESULTS.md` is the working log of the second version. A file in `v2/results/runs*/` holds one game: `meta`, `rounds` with moves and exploitability, and `calls` with each prompt and reply.

## What an agent knows about the others

No prompt says what the other players are: not that they are language models, of the same or another family, copies of the agent, or scripts. Every decision is requested in a fresh context that holds the rules and the record of past play, without the text the agent wrote earlier. `v2/games.py` holds the prompts. The second version of the preprint sets this out against the human study of Menshikov et al. (2017).

## Replaying the games

`v2/play.py` talks to any server with the Messages API or the OpenAI API: `python3 v2/play.py pd --cond loaded --seeds 0-4 --api openai --base http://localhost:8000 --model <name>`. The scripts in `v1/` take `QWEN_URL` and `OLLAMA_URL` from the environment and are kept as they ran, with comments and console messages in Russian. Replies are sampled, so a replay gives new games, not the stored ones.

## What is not here

- The hidden states of the tapped runs (13 GB). `v2/results/` holds the tables computed from them.
- The inference engine with the state tap (Mjölnir), which is not public, so the states cannot be recomputed from this repository alone. `--tap` in `play.py` needs that engine; the rest of `play.py` does not.
- Full transcripts of the earliest games of the first version. Those scripts printed to the console and did not save their games, so `v1/logs/` holds the console output, cut to the last lines in some cases as marked in the file.

## License

MIT, see `LICENSE`. To cite this work, see `CITATION.cff`.
