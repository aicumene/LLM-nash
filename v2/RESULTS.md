# Results log of the second version

The working log of October 2026, kept for the record. The preprint is the reference; where the two differ the preprint holds.
One reading here was withdrawn later: "reactive, memory-one-like" for the random defector. See `analysis/conditional_counts.py` and Section 4.2.

Engine: Mjölnir (4 × V100), Qwen3.8-Flash-Next UD-IQ3_XXS, Messages API; prompts of the original scripts; 5 seeds
(the sampling seed is not fixed in this first batch — the engine took no seed then). Runs: `results/runs/*.json`
(80 files, 2026-10-17), summary with `summarize.py`.

## Behaviour (first batch, Qwen3.8 against the paper's Qwen3.6)

**p-beauty, p = 2/3** (N = 5, 6 rounds): round-1 mean 25–38 (the human level-1/2 range; Qwen3.6 started at 6.1),
monotone convergence to 0.5–1.1 by round 6 in all 5 seeds; exploitability 15–40 → 0–1.
**p = 1.2** (contamination probe, NE = 100): the mean RISES, 52–74 → 88–100 in all seeds (4 of 5 reach 100 by round 4–6)
— the correct direction, where Qwen3.6 went down (44 → 20). The recall explanation of the paper does not hold for this
model: it tracks the payoff rule.
**Surrogate (one real agent, 4 fabricated non-reacting others):** the real agent's pick hovers at 20–78 with no trend,
exploitability stays 32–82: the convergence of the real game is driven by the others' picks, as a surrogate control
should show.

**Prisoner's Dilemma, 8 rounds, self-play.** Loaded labels (Cooperate/Defect): defection from round 1–2 in all 5 seeds
(player 2 defects at round 1 in every seed, player 1 follows; DD from round 2–4 on; exploitability 0 = the stage-game NE).
Neutral labels (Option A/B, the labels also in the history lines): 4 of 5 seeds cooperate on every round (exploitability 2
throughout), 1 seed defects. So for Qwen3.8 the behaviour is label-gated in the opposite sense to the paper's Qwen3.6
(100 % cooperation under both labelings): the word "Defect" triggers defection, the neutral game stays cooperative.
**Scripted opponent** (player 2 fixed): always-C → player 1 cooperates throughout in 3 seeds and defects throughout
(the best response, 5 points a round) in 2; a single defection at round 4/8/11 → immediate defection next round in 13 of
15 games (alpha = 0 on that round), followed by a long defection streak in about half of them (early: 10 consecutive D;
mid: 3–8; late: 2–3) — longer memory than Qwen3.6's "back within two rounds"; random 35 % defection → mixed play,
mostly D after an opponent's D and C after C (reactive, memory-one-like).

**Braess, 6 drivers, sequential best response, 4 passes.** Option times given ("easy"): the exact NE at every pass in every
seed, both networks (3/3 → 83; 2/2/2 → 92). Formulas ("hard"), no shortcut: oscillating 2/4 → 4/2 → 5/1 → 3/3, exploitability
0–33, no settling; with the shortcut: never 2/2/2 — the drivers pile onto R2 (0/4/2, 0/6/0; exploitability 22–55), where
Qwen3.6 piled onto the shortcut R3. Experience only (20 days): the day-1 assignment freezes for 7–12 days in every seed,
then moves; 2/2/2 is reached transiently in 3 seeds (one seed starts at it) and is not held. Surrogate one-shot decisions
with the formulas (fabricated others): mean regret 7–13 (argmin = 0) — the congestion arithmetic is done imperfectly, not
ignored.

## The tap (internal states) — validation 2026-10-17
Two tapped requests with the same seed: identical text and states. The tapped states against llama.cpp's `l_last-N` for the
same transcript (141 prompt + 37 generated tokens, rows aligned by absolute position; `nash_dump`, the same GGUF):

| layer | batched prefill (ubatch 512): median / max rel. error, min cos | token by token (ubatch 1): median / max, min cos | steps with a different expert set (of 38) | median error on the steps with the same experts |
|---|---|---|---|---|
| 6 | 2.5 % / 9.6 %, 0.997 | 1.45 % / 5.5 %, 0.9985 | 5 | 1.1 % |
| 11 | 3.5 % / 10.7 %, 0.996 | 2.0 % / 5.8 %, 0.998 | 7 | 2.0 % |
| 18 | 6.0 % / 13 %, 0.991 | 5.0 % / 11 %, 0.994 | 10 | 4.7 % |
| 30 | 8.7 % / 28 %, 0.963 | 9.4 % / 33 %, 0.948 | 23 | 6.4 % |
| 42 | 10 % / 24 %, 0.976 | 11.6 % / 21 %, 0.981 | 15 | 11.5 % |
| 47 | (truncated in the batched dump) | 9.4 % / 18 %, 0.984 | 17 | 8.5 % |

The engine author's own checks give ≪1e-3 against a CPU reference with llama.cpp's 8-bit rounding and 1e-2 → 2e-1 by
depth against f32 decode; our residual at layer 6 (1.1–1.5 %) is above the former, and the deep layers carry routing flips
(the chosen experts differ on 13–60 % of the steps). The likely remainder: the prompt is read by Mjölnir's chunked fp16
prefill and by llama.cpp token by token (q8_1), so the recurrent and KV states entering the reply already differ (the largest
error at layer 47 is at step 0, the last prompt token). For the trajectory measures this is the same object up to the two
engines' kernels; the engine's author confirms the account: its prefill (wmma, no 8-bit rounding of the activations) sits
at 2e-4–1e-3 of an f32 reference while its decode and llama.cpp's ubatch-1 decode both round activations to 8 bits (~1e-2), so the
recurrent and KV states differ by ~1 % before the first reply token; the deep layers add routing flips. Optional closing check, not
run: the server with `MJ_BY_STEPS=1` (the prompt read by decode steps of 3 tokens, bit for bit as one by one), the same tapped request,
the same comparison — ~1e-3 expected on the early layers where the experts agree. A tapped step takes 15–18 ms against 13 ms with the graph.

## Internal states of the 80 runs (llama.cpp dump of the transcripts, batched prefill path; mean over the 4 streams; per-round means over 5 seeds) — interim, 2026-10-17

Measures (`measure.py`): the agent's state in a round read at `s0` (after the last prompt token: the representation of the game
before writing), `mean` (over the reply) and `last` (after the last token written); per round the drift of each agent's state from
its previous-round state (relative to the mean norm), the agents' spread (≥ 3 agents) and PR; per reply the PR of the first 16
steps, the PR of the whole reply, the length. Tables: `results/measures2_dump/{AGGREGATE,ROUNDS}.md`.

**1. The round-to-round drift of the game representation (`drift_s0`) is the same in the real game, in the surrogate and in the
probe** — at every layer. Beauty contest, layer 18, rounds 2→6: real p = 2/3 0.40 → 0.13 → 0.10 → 0.14 → 0.10; surrogate (fabricated
non-reacting others, no behavioural convergence) 0.40 → 0.12 → 0.11 → 0.11 → 0.09; p = 1.2 0.38 → 0.14 → 0.14 → 0.09 → 0.09. Layers 6,
30, 47 give the same picture (e.g. layer 47: 0.54/0.54/0.51 → 0.17/0.17/0.18). The decline is the effect of a growing history (each
round adds one line; the state after the fixed template suffix moves less), not of the play converging: a non-interacting control
reproduces it exactly. The same in the PD: loaded 0.30 → 0.18, neutral 0.29 → 0.18 (layer 30) although one defects and the other
cooperates. Correlations of `drift_s0` with exploitability inside a condition (ρ ≈ +0.4–0.6 for the converging beauty contest) are
therefore a time confound and not evidence of internal convergence.
The concatenation of the four streams (10 240 numbers) instead of their mean gives the same picture at every layer (layer 18, rounds 2→6:
real 0.34 → 0.08, surrogate 0.34 → 0.08, p = 1.2 0.32 → 0.08; `results/measures2_all/`).

**2. What does track the play: the reply states and events.** (a) Where the outputs become identical the reply states converge
across agents: p = 1.2 `spread_last` 0.52 → 0.30 (layer 18), 0.26 → 0.14 (layer 47), the agents all writing "100"; p = 2/3 stays at
0.45–0.50 (picks 0.5/0.8/1.1 differ in their digits). (b) A scripted betrayal is an event in the state: in `script_single_early`
(defection at round 4) the next round's `drift_s0` jumps to 0.22 against 0.17 before and after, `drift_last` to 0.75 against 0.44–0.48,
and the replies get longer (25–31 → 42 tokens) with a higher first-window PR (6.0 → 7.3–7.6 in `single_mid`, layer 18). (c) In
the Braess experience game the freeze (days 2–6) shows as short replies (104 tokens against 185 on day 1) with a low across-agent
spread (0.3 against 0.8); the later exploration brings 220-token replies and a spread of 0.7. (d) In the loaded PD the first-window
PR falls 8.1 → 6.7 as DD settles (replies stereotyped); in the neutral PD it stays at 6.2–7.2.

**3. A linear probe of the round's exploitability from the state** (`export_states.py`, `probe.py`: PCA to 8 components fitted on the
training seeds, ridge, leave-one-seed-out R²; shuffle control; `results/states_dump/PROBE.md`, dump states, layers 18/30). Beauty contest
p = 2/3: this round's normalised exploitability is read from the pre-reply state `s0` with R² 0.72 (ρ 0.93) and the previous round's with
0.59–0.77; the shuffled target gives ≈ 0. But the probe trained there fails on p = 1.2 (R² −3) and on the surrogate (−17), and in the
surrogate nothing is decodable (R² ≤ 0, n = 30); in p = 1.2 only a weak rank relation remains (ρ 0.4, R² ≈ 0). So what the state carries is
the history's trend of the picks (which the exploitability of p = 2/3 follows), not a parameterisation-free "distance to equilibrium".
Prisoner's Dilemma: from the reply readouts (`act`, `mean`, `last`) the round's regret is read with R² 0.5–1.0 — these states encode the
move just written; from `s0` (before writing) R² 0.27–0.35 in the loaded game (a weak anticipation of the coming move), 0.1–0.2 in the
scripted games; the scripted conditions transfer among themselves (R² 0.6–0.9 for the reply readouts), the loaded game does not (its
equilibrium value differs). Braess with formulas and the experience game: nothing decodable (R² ≤ 0); the easy game has a constant target
(exploitability 0 throughout) and cannot be probed.

**Reading (critical):** with these readouts the representation of the game moves as any growing history moves it, and the agent's
state is pulled by what it writes; a representation of "being at equilibrium" distinct from the behaviour is not visible here. The
positive v2 statement is the control itself: the surrogate's internal dynamics equal the real game's, so the behavioural convergence
of the beauty contest is not accompanied by an internal change that a non-interacting history would not produce. Open: (i) the
same on the tap states (the decode path, reproducible seeds; running), (ii) the concatenated streams instead of their mean, (iii)
a linear probe of the round's exploitability from `s0` with cross-seed validation and the surrogate as the control, (iv) the state at
the action token rather than at the last token.

## BOID signature of the token trajectories — the null levels first (2026-10-17, `results/boid_noise/NOISE.md`)
On windows of 24–48 steps (one reply) the estimators behave as follows on controls of the same shape (N = 200, d = 2 560):
white noise — d_osc with the shuffle null 0 (no false oscillators), but ESPRIT marks every mode of the 95 % rank as sustained
(6/6, 12/12, 14/14), MC-SSA with the 1/f null still finds 2–4 "oscillator planes" (the aperiodic fit on 24–48 samples is poor), the
one-step operator explains 0.49 of the change (the AR(0) baseline: the change is −x plus noise), PR 16/27/32 (sample-limited);
AR(1), φ = 0.9 — d_osc 3/5/6 (the shuffle null counts autocorrelated directions, not oscillators), ESPRIT sustained 4/7/9 of 5/7/9,
MC-SSA 1/3/4, operator 0.01, PR 5–7. So on single replies only PR, synchrony, the speed contrast and the operator share can be read
against these baselines; the oscillator counts need longer trajectories — hence `boid_long.py`: each agent's replies of a game
concatenated in round order (240 steps) and the 200-step Braess formula replies, with controls at T = 200 and 600.

### BOID signature of single replies (`results/boid_dump/BOID.md`; T = 24 PD / 40 beauty / 48 Braess steps, all seeds and rounds pooled)
Read against the controls of the same shape: **no oscillators above the 1/f null** — MC-SSA gives 2–4 planes in every condition and layer,
exactly the white-noise level (2 at T = 24, 3 at 40, 4 at 48); ESPRIT marks every mode sustained, as on noise; d_osc with the shuffle null
is 4–12, the level of AR(1) φ = 0.9 (3–6), i.e. it counts the autocorrelation of the token trajectory, not oscillators. What is above the
null: the **speed contrast** 2–12 against 1.1 (fast and slow phases inside a reply, strongest at layer 6: 5–12), the **one-step operator**
share 0.6–0.86 against the 0.49 of white noise — the Braess formula replies are the most predictable flow (0.80–0.86 at q = 1,
0.89–0.91 with 8 steps of history), the beauty contest the least (0.50–0.62), the PD 0.60–0.74 (neutral labels 0.74 > loaded 0.62),
the experience game 0.58–0.63 < the formula games 0.81; **PR** 13–15 (T = 24), 19–24 (40), 22–29 (48) — below the sample-limited white
level (16 / 27 / 32), above AR(1) (5–7); synchrony 0.15–0.25 between the two controls. Between the real beauty contest, its surrogate
and the p = 1.2 probe the signature does not differ (PR 22.4 / 22.5 / 22.9, d_osc 6 / 5 / 7, contrast 5.2 / 5.5 / 5.3, operator 0.56 /
0.52 / 0.53 at layer 18); the q = 8 operator of the surrogate (N = 30) overfits and is not read. So on the scale of a reply the token
trajectory of this model is an aperiodic, phase-structured, largely predictable flow; the game's state does not change its BOID
signature. Long trajectories (an agent's replies over a game, 240 steps; 200-step formula replies) with controls at T = 200 / 600
are being computed (`boid_long.py`) — note that the concatenation carries the trivial period of one round (one reply ≈ one cycle).

## The tapped re-runs (Mjölnir tap = the decode-path states, request seeds 0–2; 48 games, `results/runs_tap/`, `results/measures_tap/`)
**Behaviour replicates:** beauty contest p = 2/3 33 → 0.7, p = 1.2 46–66 → 99–100 in all three seeds, the surrogate hovers at 22–77;
Braess with formulas piles onto R2 again (0/6/0), the experience game freezes for 5–10 days; PD with loaded labels defects in 2 of 3
seeds (one seed cooperates throughout), with neutral labels 1 of 3 cooperates. Pooled over both batches the label effect is a tendency,
not a fact: 1 of 8 loaded games cooperates against 5 of 8 neutral ones.
**Internal measures replicate on the decode-path states:** `drift_s0` identical in the real contest, the surrogate and the probe
(layer 18, rounds 2→6: 0.42/0.41/0.40 → 0.10/0.09/0.11; layer 30: 0.36/0.37/0.38 → 0.14/0.13/0.18); `spread_last` falls only in p = 1.2
(0.55 → 0.22 against 0.57 → 0.55 for p = 2/3); the defecting PD replies are more stereotyped (first-window PR 6.3–6.7 loaded against
7.8–8.8 neutral at layer 18). The probe and the BOID signature on these states follow (`states_tap`, `boid_tap`).

### Long trajectories and the tap states (`results/boid_long/BOID_LONG.md`, `results/boid_noise_long/`, `results/boid_tap/`, `results/states_tap/PROBE.md`)
Over 240 steps (an agent's replies of a whole game in round order) and over the 200-step Braess replies, against controls of the same
length (white: PR 158, d_osc 0, MC-SSA 4, operator 0.50; AR(1) φ = 0.9: PR 20, d_osc 22, MC-SSA 5, operator 0.01): PR 87–124, d_osc 34–51
(more smooth directions than an AR(1)), **MC-SSA 3–6 = the null level again — no oscillator above the 1/f background on the scale of a
game either**, ESPRIT's shortest periods (2.2 steps) dominate, ~20 fast phases per 240 steps (one per ~12 tokens: events inside replies,
not the round boundaries), the operator 0.46–0.69 (q = 1) and 0.5–0.86 (q = 8): formula replies the most predictable, experience and
beauty-contest replies the least; p = 2/3 and p = 1.2 identical, real and surrogate Braess identical. The BOID signature of the tap
states (decode path) equals the dump's. The probe on the tap states repeats the dump's: p = 2/3 R² 0.78 from the pre-reply state, p = 1.2
≈ 0, the surrogate nothing (n = 18), transfer p = 2/3 → p = 1.2 only ρ 0.38; the PD probes are unstable with three seeds (one loaded
seed cooperates throughout) but the loaded and neutral probes agree in rank (ρ 0.8 across the two label conditions: the same state
direction precedes cooperation or defection whatever the labels).

## Plan A/B/C/D after this run — what can be written (2026-10-18)
1. **Instrument (b), convergence in representation space, is implemented and gives a negative result with a positive control.** The
agent's pre-reply state moves from round to round exactly as it does in a non-interacting surrogate (the drift is set by the growing
history), its across-agent spread collapses only when the outputs become identical, a linear probe reads the distance to equilibrium
only where the history already carries it (p = 2/3) and does not transfer to another parameterisation or to the surrogate, and the
BOID signature of the token trajectories (no oscillators above the 1/f null; phase-structured, largely predictable aperiodic flow) is
the same whatever the game's state. Behavioural convergence is therefore not accompanied by an internal change that a non-interacting
history would not produce — consistent with the paper's thesis that these agents reach equilibrium only where it reduces to a myopic
best response over given numbers.
2. **Instrument (a), the surrogate control, is implemented for all three games** and is what turns the naive correlations (drift and
exploitability both fall with the round) into the null above.
3. **Seeds:** 5 per condition in the first batch, 3 reproducible ones (request seeds through the tap build) in the second; the two
batches agree on every behavioural and internal statement; the PD label effect is the one claim that stays a tendency (1/8 vs 5/8).
4. **Contamination probe:** Qwen3.8 passes p = 1.2 (rises to 100), where the paper's Qwen3.6 did not — a version difference to report.
5. **Events** (a scripted betrayal) are visible in the state: a jump of the pre-reply and reply states, longer and higher-dimensional
replies; the Braess freeze shows as short, similar replies — the internal measures track what the agent is doing, not where the game is.
Data: the transcripts and tables under `results/` here; the raw tap states (13 GB) are not in this repository.
