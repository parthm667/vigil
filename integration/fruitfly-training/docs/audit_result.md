# G0 connectome audit result

Plan: `docs/DRONE_RL_PLAN.md` 4.3. Run on 2026-09-26, Apple M5, MaleCNS v1.0 (pairs with at least 3 synapses).

## Decision

**G0 = full controller (yaw and forward from the brain), trained on `data/brains/pursuit_core1.npz`.**

- Structural: pass. Each side's LC10a reaches DNa02 on both sides at 2 synapses, as a push-pull: excitation to the same-side DNa02 (via AOTU025, AOTU012, AOTU016_a, all cholinergic) and inhibition of the opposite DNa02 (via the GABAergic AOTU019). No direct LC10a to steering-DN synapses.
- A (bearing): pass on core1, with arousal on and off, when LC10a is driven at plan rates (alone, or with LC9/LC11 at 10 Hz, which is the CLI default). It fails when LC9/LC11 are driven at 50 Hz. The response is a strongly biased, saturating curve rather than a clean proportional one (see caveats).
- B (size): pass on core1 in every encoder variant; total DN activity rises with s.
- Pass rules in `audit.py`: A needs (1) the two ends significant (more than 2 SE) and of opposite sign, (2) no reversal between neighboring bearings larger than 2 SE, and (3) a swing across +/-30 deg larger than one per-tick SD. B needs a monotone and significant change that is larger than 10 %. Rules (3) and the 10 % were added after core2 met the looser rules with a 40 Hz swing, which a 50 ms readout cannot use.
- C (arousal on and off): the result does not depend on arousal; P1 adds only a small tonic DN baseline.
- **core2 fails functionally**: its LIF dynamics run away into a self-sustained state whatever the spot bearing. Do not train on it.

Sign relation (the readout must use this sign or learn it): **when the spot is on the right, DNa02_R fires and DNa02_L goes silent** (at +30 deg: R 313 Hz, L 0 Hz). Spot on the left gives the mirror image (at -30 deg: L 162 Hz, R 0 Hz). So DNa02 (R - L) and the pooled (R - L) have the same sign as the bearing: `yaw = +w * (R - L)` with w > 0 turns toward the target (yaw > 0 is clockwise, to the right). This matches DNa02 being ipsiversive.

## Sizes and speed

| Brain | Neurons | Connections | ms per 50 ms tick (mean / p95) |
|---|---|---|---|
| Full MaleCNS | 166,700 | 10,520,377 (59,262 inhibitory neurons) | not benchmarked |
| **pursuit_core1** (hops=1, at most 2-synapse paths) | 1,446 | 40,586 | **2.0 to 2.3 / 2.4** |
| pursuit_core2 (hops=2, at most 4-synapse paths) | 50,195 | 3,826,555 | 19 to 22 / 23 |

Bench: `python -m flyfollow.brain.build bench`, single process, 100 LIF steps at dt 0.5 ms per tick, right-side spot at +15 deg (LC10a up to 150 Hz, LC9/LC11 50 Hz) with P1 on. Two runs gave 2.26 and 2.03 ms (core1) and 22.2 and 19.2 ms (core2).

Group sizes (identical in the full brain and in both cores; sides from rootSide, falling back to somaSide):

| Group | L | R | Notes |
|---|---|---|---|
| LC10a | 135 | 140 | 8 azimuth bins per side: L 17 x 7 + 16, R 18 x 4 + 17 x 4 |
| LC9 | 104 | 115 | |
| LC11 | 68 | 75 | |
| AROUSAL (P1) | 43 | 43 | 25 pC1 types, see below |
| DNa02, DNa01, DNb05, DNg13, DNb06 | 1 each | 1 each | one cell per side, as expected |
| audit only: DNp09, AOTU019 | 1 | 1 | |
| audit only: LC10b / LC10d / LC10e | 47 / 108 / 62 | 48 / 106 / 48 | MaleCNS has no type named LC10c |
| audit only: pC1x (Sten 2025) | 4 | 4 | |

## Structural audit (full brain)

Path sums are the sum over all paths of the product of signed synapse counts (`data/audit/audit_structural.json`).

| LC10a side to DN | Direct | 2-hop | Main 2-hop relay |
|---|---|---|---|
| L to DNa02_L / DNa02_R | 0 / 0 | **+1.84M / -1.74M** | AOTU025_L +0.68M, AOTU012_L +0.43M / AOTU019_L -1.91M |
| R to DNa02_R / DNa02_L | 0 / 0 | **+2.01M / -2.05M** | AOTU025_R +0.73M, AOTU012_R +0.39M / AOTU019_R -2.25M |
| L to DNa01_L / DNa01_R | 0 / 0 | +3k / -234k | AOTU019_L inhibits DNa01_R |
| R to DNa01_R / DNa01_L | 0 / 0 | -23k / -161k | AOTU019_R |
| L to DNg13_L / DNg13_R | 0 / 0 | +82k / +69k | AOTU002, AOTU016 |
| R to DNg13_R / DNg13_L | 0 / 0 | +116k / +16k | AOTU016_c |
| L, R to DNb05 | 0 | at most 1.5k | negligible |
| L, R to DNb06_L | 0 | +339k, +347k | AOTU023 of both sides drive only DNb06_L |
| L, R to DNb06_R | 0 | +10k, 0 | DNb06_R has almost no LC10a pathway |

- Ipsi vs contra (2-hop): DNa02 ipsi +3.85M, contra -3.79M; DNg13 ipsi +198k, contra +85k; DNa01 ipsi -20k, contra -395k.
- Turn-toward index (sum over types of turn sense x (same-side minus opposite-side path)) is positive for both sides at 2 and 3 hops: each side's LC10a drives turning toward its own visual field.
- Top intermediates (both sides alike): AOTU019 (about 2.2M to 2.5M), AOTU025, AOTU012, AOTU023, AOTU016_a, AOTU005, LAL028.
- Only direct LC10a to DN synapses: DNa10 762, DNp11 158, DNa16 156, DNp63 108, DNp09_R 53. None of the readout DNs.
- LC9 and LC11 have much weaker 2-hop paths to the readout DNs (LC9_L to DNa02_L 156k; LC11 at most 42k), mostly not lateralized the LC10a way.

## Functional audit (LIF, `python -m flyfollow.audit.audit`)

Protocol: 5 repeats per condition (independent noise seeds), 300 ms warm-up, then 1 s of spot; the first 200 ms are dropped and DN rates are counted per 50 ms tick. Pooled asymmetry = sum over the 5 types of (R - L) in Hz. SNR = mean / SD of the 1 s means across repeats. Per-tick SD is the SD of the single 50 ms tick value, which is what the readout sees before its low-pass filter. Audit encoder: 8 Gaussian bins per side, right bin k centered at -overlap + k x 100/7 deg (left mirrored), width sigma x s^0.6, gain 1.5 s/(s + 0.5); P1 at 20 Hz when arousal is on.

### Core1, LC10a-only drive, plan rates (r_max 150 Hz, sigma 10 deg, overlap 10 deg): A pass, B pass

| Bearing | Pooled R-L (Hz), arousal on | SNR | Per-tick SD | DNa02 R-L | Pooled R-L, arousal off |
|---|---|---|---|---|---|
| -30 | -244 +- 10 | 24 | 43 | -162 | -228 |
| -15 | +44 +- 15 | 3.0 | 36 | +113 | +53 |
| 0 | +156 +- 6 | 26 | 34 | +240 | +147 |
| +15 | +202 +- 7 | 27 | 34 | +269 | +199 |
| +30 | +288 +- 5 | 58 | 35 | +313 | +316 |

B (spot at 0 deg), total DN activity: s = 0.5 / 1 / 2 gives 445 / 714 / 930 Hz (arousal on), 386 / 700 / 904 Hz (off). Pass.

### Core1 variants (sensitivity)

- **CLI default: plan encoder plus LC9 and LC11 at 10 Hz** (`python -m flyfollow.audit.audit --brain data/brains/pursuit_core1.npz`): pooled -268, +30, +274, +350, +412 Hz (arousal on; SNR 57, 2.5, 16, 21, 127; per-tick SD 42 Hz). A pass, B pass (426 / 871 / 1120 Hz). Arousal off gives the same verdict.
- **LC9/LC11 at 20 Hz**: A fails, because -15 and 0 deg both give about +240 Hz.
- **Plan encoder plus LC9 and LC11 at 50 Hz** (all cells of the spot side): pooled -503, +20, +6, +42, +399 Hz. A fails on the monotone test only (a 14 Hz dip from -15 to 0 deg, just past 2 SE); the sign still flips. With the whole LC9/LC11 population driven, the output becomes a bang-bang switch with a flat middle between -15 and +15 deg, and DNs saturate at 300 to 500 Hz. B passes (1073 / 1317 / 1428 Hz).
- **Graded encoder** (LC10a only, r_max 50 Hz, sigma 8 deg, no overlap): pooled -76, -103, +105, +208, +182 Hz. The zero crossing moves to about 0 deg, but A fails the monotone test at the lateral ends: +/-30 deg drives the DNs less than +/-15 deg. With rank bins, each bin is an arbitrary set of about 17 cells, so bin-to-bin differences in connectivity show up as bearing wiggles. The trained per-bin gains (0.5 to 2) are meant to fix exactly this. B passes (216 / 295 / 390 Hz).
- **Degree-preserving shuffles of core1** (LC10a-only, plan rates): 5 of the 6 shuffle and arousal combinations fail A; all pass B. For example, shuf1 stays between -48 and -59 Hz at every bearing, and shuf3 goes -90, -123, -123, -128, +28. The exception is shuf2 with arousal off, which passes A with the **reversed** sign (+59, +41, +36, -79, -32 Hz; a 91 Hz swing against 544 Hz for the real core). So the large, correctly signed bearing signal in core1 depends on the actual fly wiring. That is good news for the FLY-SHUF contrast, although a trained readout could still use shuf2's weak reversed signal.

### Core2: fails

With P1 at 20 Hz and no spot, core2 already sits in a self-sustained state (DNb05_L/R 370/400 Hz, DNa02_L 72 Hz, DNa02_R 0 Hz). With arousal off and no spot it is silent, but any spot sets off the same state. Even a drive of P1 at 2 Hz or LC10a at 10 Hz does it. Pooled R-L across -30 to +30 deg stays in a 40 Hz band with SNR below 3.6, and DNa02 R-L never changes sign. B total is flat (about 900 Hz). Every variant (CLI default, LC10a-only, LC9/LC11 at 50 Hz, low-rate probe) fails A and B under the final rules. The 50k-neuron subgraph keeps a lot of recurrent excitation. Making it usable would need a change to the fixed LIF parameters (for example a lower w_syn), which is out of scope tonight.

## P1 / arousal decision

- There is no MaleCNS type named P1. P1 (Kimura 2008) = pMP-e (Cachero 2010) = pMP4 (Yu 2010). **AROUSAL_L/R = the 25 pC1_* types whose `synonyms` list "Cachero 2010: pMP-e; Yu 2010: pMP4"** (all fru/dsx co-expressing, mostly male-specific; 43 cells per side). pC1x (Sten 2025) and the dsx-only pC1 types are excluded. This identification is by name correspondence only.
- Structurally, P1 has no direct (at least 3 synapses) connection onto LC10a or AOTU019. LC10a feeds P1 (263 synapses per side), not the other way round. P1's 2-hop drive to DNa02 is about 24k to 34k, roughly 1.5 % of LC10a's.
- Functionally in core1, P1 at 20 Hz gives a small tonic DN baseline (DNa02 L/R 6/12 Hz, DNa01_R 13, DNg13_L 17) and leaves A and B essentially unchanged. The arousal gating of the LC10a pathway seen in real flies is not reproduced here.
- **Recommendation:** keep the AROUSAL groups in the files (they are frozen interface names and non-empty), but start the encoder's P1 rate at 0 with a low bound (at most about 10 Hz), and use the plan's fallback `arousal_gain` on LC10a as the arousal knob. Never drive P1 on core2.

## Shuffles (FLY-SHUF, `python -m flyfollow.brain.shuffle --hops 1`)

These are vectorized double-edge swaps within the excitatory and inhibitory edge sets separately. Per-sign in-degree and out-degree are kept for every neuron (so Dale's law holds), there are no self-loops or duplicate pairs, weights are permuted within each sign set, and groups and meta (including azimuth_bins) are unchanged. About 24 swaps per edge, 0.2 s each.

| File | Seed | Walks of length at most 2, input to output (1-step + 2-step) | Input-output pairs reachable |
|---|---|---|---|
| real core1 | | 15,090 (24 + 15,066) | 4,056 |
| pursuit_core1_shuf1.npz | 1001 | 20,060 (547 + 19,513) PASS | 5,507 |
| pursuit_core1_shuf2.npz | 1002 | 19,160 (530 + 18,630) PASS | 5,429 |
| pursuit_core1_shuf3.npz | 1003 | 19,970 (529 + 19,441) PASS | 5,464 |

Note: the shuffles have about 22 times more direct input-to-DN edges than the real core. That is because inputs have high out-degree and DNs high in-degree. So FLY-SHUF is not handicapped on path count; if anything it gets shortcuts.

**No core2 shuffles.** A plain shuffle reaches only about 54 % of the real core2 walk count (length at most 4), because core2 is enriched for input-to-output paths. A layered shuffle (swaps restricted to edges between the same distance-from-input and distance-to-output layers) reaches about 90 %. Neither passes the "at least as many" rule. `--mode auto` would try both and save nothing. core2 is not recommended anyway.

## Which core to train: core1

- It is the only core with a bearing-dependent steering signal (core2 runs away).
- It costs 2 ms per 50 ms tick against 20 ms. A 60 s episode (1200 ticks) is about 2.5 s of brain compute on core1 and about 25 s on core2.
- The fidelity cost: core1 is a 2-synapse relay (LC10a to AOTU to DN). It has no recurrence beyond what sits inside that relay, so any "brain" dynamics are mostly the single-cell LIF filtering plus the AOTU019 push-pull. We should say this in the demo.

## Caveats (read before training)

1. **Right bias.** At 0 deg the pooled R-L is +150 Hz and the zero crossing sits near -15 deg (plan encoder). The right LC10a has 140 cells vs 135 on the left, and its pathway is stronger. The yaw bias b_yaw or the per-bin gains must absorb this; initialize b_yaw to cancel the 0 deg asymmetry measured at hand calibration.
2. **Saturation and step-like response.** DNa02 saturates near 300 to 360 Hz, and the input-to-yaw curve is steep near the crossing and flat beyond about 20 deg. Expect a bang-bang-like yaw. LC9/LC11 driving whole populations makes it worse, so start the LC9 and LC11 gains low (about 10 Hz or less).
3. **Readout noise.** With one cell per side, the pooled per-tick SD is 30 to 45 Hz against a slope of about 5 Hz per deg near the crossing (DNa02 alone: per-tick SD 14 to 30 Hz). That is roughly 7 deg of bearing noise per 50 ms tick before the low-pass filter. As the plan predicted, this is the main reason the fly arm may lose to NOBRAIN.
4. **Useless or constant readout DNs.** DNb06_R is silent in every condition (no pathway), so DNb06 R-L is a constant offset. DNb05 is silent without LC9/LC11 drive. DNa01 and DNg13 carry weak, non-monotone asymmetry. Effectively the yaw signal is DNa02, with DNg13 and DNa01 as minor contributors. The readout weights will sort this out; no interface change is needed.
5. **Azimuth bins are the rank fallback** (`meta["azimuth_bins_method"] = "rank"`). They split each side by core index, so bin 0 is not really frontal. MaleCNS has no hex coordinates for LC10a, and I did not attempt the partner-hex retinotopy (it was over the time box, and the frontal-vs-lateral orientation of the hex axes could not be verified offline). The trained per-bin gains are the mitigation.
6. **The B pass is partly trivial.** A bigger spot means more LC10a spikes in, so more DN spikes out. It shows that size information reaches the DNs; it does not show a distance-specific circuit.
7. The audit encoder is ours (Gaussian bins, fixed overlap), not the trained TargetEncoder. The functional numbers depend on its rates.

## Reproduce

```
scripts/setup_data.sh                                       # download (resumable) + build + core1 shuffles + bench
.venv/bin/python -m flyfollow.brain.build [--force]         # full brain, core1, core2
.venv/bin/python -m flyfollow.brain.build bench
.venv/bin/python -m flyfollow.audit.audit --brain data/brains/pursuit_core1.npz data/brains/pursuit_core2.npz   # structural + functional, LC9/LC11 at 10 Hz
.venv/bin/python -m flyfollow.audit.audit --functional-only --brain data/brains/pursuit_core1.npz data/brains/pursuit_core2.npz --lc10-only --tag default
.venv/bin/python -m flyfollow.audit.audit --functional-only --brain data/brains/pursuit_core1.npz --aux-rate 50 --tag aux50
.venv/bin/python -m flyfollow.audit.audit --functional-only --brain data/brains/pursuit_core1.npz --lc10-only --rmax 50 --sigma 8 --overlap 0 --tag graded
.venv/bin/python -m flyfollow.brain.shuffle --hops 1
```

Environment gotcha: the venv's editable-install `.pth` files carry the macOS "hidden" flag, which something keeps re-applying, and Python 3.12 skips hidden `.pth` files. When that happens, `import flydrones` fails even though it is installed. The fix is `export PYTHONPATH=$PWD:$PWD/third_party/FlyDrones/src`. Alternatively, call `flyfollow.brain.build.ensure_flydrones()` before importing flydrones (all modules listed here already do this).

JSON outputs are in `data/audit/`. The full functional audit takes about 1 s on core1 and about 12 s on core2 with 8 worker processes.
