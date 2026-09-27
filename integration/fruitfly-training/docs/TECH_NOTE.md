# What the fruit fly brain does and what the RL trains

Technical note for the team and judges, 2026-09-26. Sources: `docs/audit_result.md`, `docs/RUNBOOK.md` Section 7,
`docs/DRONE_RL_PLAN.md` Sections 1 and 4, `configs/`, `flyfollow/`.

## 1. Summary

A subgraph of a real fruit fly connectome (fixed wiring, simulated as spiking leaky integrate-and-fire neurons)
steers a DJI Tello toward a person or an object. The connectome is never trained. Only a small interface around
it is trained: an encoder (camera box to input spikes) and a readout (output spikes to the yaw stick). Training
runs in simulation with CMA-ES on Modal. Distance keeping, perception, safety and mission logic are conventional
code (the team's ReachGlass stack); the fly does one job, steering (yaw).

## 2. The fly brain

**Connectome.** MaleCNS v1.0: a male fly, full central nervous system, 166,700 neurons and 10.5 M connections
after keeping neuron pairs with at least 3 synapses. Each connection is signed by its neurotransmitter
(excitatory or inhibitory).

**The pursuit circuit** (`data/brains/pursuit_core1.npz`): every neuron on a path of at most 2 synapses from the
inputs to the outputs. 1,446 neurons, 40,586 connections.

| Role | Cell types | Cells (L / R) |
|---|---|---|
| Inputs (visual) | LC10a (main target input), LC9, LC11 | 135 / 140, 104 / 115, 68 / 75 |
| Inputs (arousal) | P1 candidates (25 pC1 types) | 43 / 43 |
| Outputs (steering descending neurons) | DNa02, DNa01, DNb05, DNg13, DNb06 | one cell per side each |

**Mechanism: a push-pull on DNa02.** No LC10a cell synapses directly onto a readout neuron. Each side's LC10a
reaches both DNa02 cells through one relay layer in the AOTU:

```
     target on the RIGHT of the image
                   |
           LC10a (right side)
            /               \
  AOTU025_R, AOTU012_R     AOTU019_R
  (excitatory)             (GABAergic, inhibitory)
            |               |
            v               v
  DNa02_R: fires           DNa02_L: silenced
  (313 Hz at +30 deg)      (0 Hz)
            \               /
    readout: yaw = +w (R - L)  ->  turn right
```

The left side is the mirror image. This matches DNa02's known role (it turns the fly toward its own side).

**Model.** LIF neurons with the Shiu et al. (Nature 2024) parameters, integrated at dt 0.5 ms. One control tick
simulates 50 ms of brain time (20 Hz), which costs about 2 ms of CPU on a laptop (2.0 to 2.3 ms in the audit
bench, 1.7 ms on the M5).

**What the audit showed** (`docs/audit_result.md`):
- The bearing sign flips correctly: spot at +30 deg gives DNa02 R 313 Hz, L 0 Hz; at -30 deg, L 162 Hz, R 0 Hz.
- The response is biased to the right (zero crossing near -15 deg) and saturating, close to bang-bang.
- Readout noise: with one cell per side, a single 50 ms tick carries roughly 7 deg of bearing noise before filtering.
- P1 adds only a small tonic baseline; the arousal gating seen in real flies is not reproduced.
- The larger 4-synapse core (50,195 neurons) is unusable: it runs away into self-sustained activity whatever the target does.
- Degree-preserving shuffles of the wiring lose the clean bearing signal (5 of 6 shuffle conditions fail the bearing test; one passes with the reversed sign).

## 3. What is trained and what is not

| Part | Trained? | What it does |
|---|---|---|
| Connectome weights, cell types, signs, LIF parameters | **Never** | The fixed fly circuit |
| Encoder (`flyfollow/senses/target.py`), 25 parameters | Yes | Target bearing and normalized size (box height relative to its height at the standoff) become Poisson spike rates into 8 azimuth bins of LC10a per side, plus LC9, LC11 and arousal. Parameters: peak rate, tuning width, left/right overlap, 2 size exponents, velocity gain, 16 per-bin gains (bounded 0.5 to 2), arousal, LC9 and LC11 rates (capped at 10 Hz) |
| Readout (`flyfollow/pilot/pursuit_decoder.py`), 9 parameters in yaw-only | Yes | `yaw = 60 x G_yaw x tanh(sum over DN types of w x (R - L) + b_yaw)`, after a low-pass filter (at most 150 ms) and a deadzone. Parameters: 5 weights, bias, gain, time constant, deadzone |

**Total: 34 parameters** in the yaw-only setup (`flyfollow/rl/params.py`); 47 in the full setup, where the readout
also drives forward speed (13 more).

**Bypass rule.** The encoder writes only into the designated input neurons; the readout reads only the 10 steering
DN rates and never sees the box. Any steering has to pass through the connectome. The bias `b_yaw` could carry some
behavior on its own, so the final evaluation audits it against the DN-driven term and runs a lesion test (DN rates
clamped to their episode mean).

**Why yaw-only.** The plan's G0 fallback allows "brain controls yaw, PID's range loop sets forward". A teammate's
400-generation local run showed the full fly fails on range (13 % of time in the distance band, hanging back
0.67 m) while the yaw-only fly matched PID and no-brain. So in all yaw-only arms, forward speed comes from the same
fixed hand-tuned PID range loop, and the comparison isolates steering.

## 4. How it is trained

**Simulator** (`flyfollow/rl/env.py`). No pixels: kinematics projected to a detector box.
- Drone dynamics centered on the measured Tello lag test (at stick 30): yaw 55 deg/s per 100 stick, 0.18 s dead
  time; forward 0.96 m/s per 100, 0.47 s dead time, tau 0.45 s. Randomized about +-30 % per episode.
- Camera and detector: focal length 650 to 950 px, video latency 0.15 to 0.45 s plus detector time, 8 to 30 Hz
  detections, box noise, random and burst dropouts, false boxes.
- Targets: a walking person (straights, turns, stops, steps toward the drone) for 60 s follow episodes, and static
  objects of uncertain size for 25 s approach episodes.

**Reward** (per tick, from true state; the controller sees only the noisy, delayed box): penalties for distance
outside the band (+-15 % of the standoff), bearing error, stick jerk, safety interventions, target lost, and too
close; events for loss (-10), too close (-20), collision (-200 and episode ends) and approach success (+20). Weights
were rescaled against the hand-tuned PID so each main term is 10 to 40 % of its cost, then frozen for all arms.

**CMA-ES** (`configs/train.yaml`, changes from the plan in `docs/RUNBOOK.md` Section 7):

| Setting | Value | Why |
|---|---|---|
| Population | 32 | About twice the pycma default (15 for 47 parameters), for noisy fitness |
| Episodes per candidate | K = 32 (20 follow + 12 approach), shared seeds per generation | Rank reliability of candidate scores 0.80 -> 0.91 (no brain), 0.55 -> 0.75 (PID) vs K = 8 |
| Generations | 150 | |
| Step size | sigma0 0.05; readout weights x0.08, biases x0.2; max std 0.3 | Readout bounds are about 10x their useful scale; at the plan's 0.2 almost every candidate was worse than the start |
| Score per episode | return / max(abs(hand-tuned PID return on the same seed), 20), clipped to [-4, 2] | Hand-tuned PID = -1; floor and clip stop near-zero denominators and single crashes from dominating |
| Fitness | 0.5 x follow mean + 0.5 x approach mean | Approach counts as much as follow |

**Modal.** 12 runs (4 arms x 3 seeds) fan out episodes over CPU containers (8 cores, 16 episode workers each),
80 evaluation containers under the Starter plan's 100-container cap. Projected about $70 and 1.4 h; launched 10:16 PT.
CMA state is pickled to a Modal Volume every generation, so a relaunch with the same tag resumes. Every 10 generations
the distribution mean is scored on 64 fixed selection seeds; the best one is saved as `best.json`.

## 5. Honesty controls and final results

Final evaluation, 2026-09-26 11:50 PT: best checkpoint of each of 3 training seeds per arm, 200 held-out test seeds
(125 follow, 75 approach) never used in training or selection, per set. Values are the mean across seeds, with the
range across seeds in brackets. Score: higher is better, -1 is about the hand-tuned PID (see Section 4). Full tables:
`docs/results/final_eval.md`; chart: `docs/results/final_chart.png`.

All arms share the same simulator, seeds, governor, box filter and CMA-ES budget (150 generations x 32 candidates x
32 episodes). In the "steers" arms the controller sets yaw only and the same hand-tuned PID sets forward speed.

| Arm | Test score | Steering error (RMS bearing) | Demo set (measured Tello) score / bearing | Stress set score |
|---|---|---|---|---|
| **Fly steers, trained** (FLY-YAW) | **-0.634** [-0.640, -0.628] | **6.6 deg** [6.4, 6.9] | -0.697 / 4.6 deg | -0.588 |
| Fly steers, untrained (hand calibration) | -0.859 | 8.9 deg | -0.987 / 8.2 deg | -0.729 |
| Shuffled fly steers (FLY-SHUF-YAW) | -0.813 [-0.880, -0.775] | 11.0 deg [10.0, 12.4] | -0.823 / 6.5 deg | -0.630 |
| No brain steers (NOBRAIN-YAW) | -0.613 [-0.619, -0.600] | 5.5 deg [5.0, 5.8] | -0.687 / 4.8 deg | -0.593 |
| PID, tuned by CMA-ES (PID-CMA) | -0.642 [-0.651, -0.627] | 5.4 deg [5.2, 5.7] | -0.704 / 4.6 deg | -0.622 |
| PID, hand-tuned | -0.683 | 6.3 deg | -0.708 / 4.7 deg | -0.679 |
| Full fly (also sets forward), untrained | -2.129 | 9.1 deg | -2.113 / 7.9 deg | -1.174 |

- **Training works on the fly:** the trained fly steers with 6.6 deg error vs 8.9 deg untrained, and its score beats
  the hand-tuned PID and matches the CMA-tuned PID (within the seed range). On the measured-Tello demo profile it ties
  for the best bearing error (4.6 deg).
- **The real wiring beats degree-preserving shuffled wiring:** 6.6 vs 11.0 deg, and the score ranges across seeds do not
  overlap (-0.640 to -0.628 vs -0.880 to -0.775). The specific connectome, not just its degree statistics, carries the
  steering signal (consistent with the audit: shuffles lose the push-pull bearing signal).
- **The brain does the steering:** clamping every DN readout channel to its episode mean (lesion) raises bearing error
  from about 9 deg to about 29.5 deg and cuts time in view from 0.97 to 0.80 (test subset). The yaw bias does less than
  the DN-driven term (|b_yaw| / DN drive = 0.51, 0.03, 0.38 across seeds); it mostly cancels the circuit's right bias.
- **As predicted before training, the no-brain control is at least as good** (5.5 deg, score -0.613): the spiking,
  single-cell readout adds noise. The honest claim is "a fixed real fly circuit steers as well as classical control",
  not "better".
- **Distance is shared:** time in band is about 0.42 for every steering arm on the test set; it is limited by the shared
  PID range loop, latency and the Tello's measured speed, not by steering. On the real drone the ReachGlass stack owns
  distance.
- **Obstacles:** with obstacles in the follow path, all arms degrade about equally (about 7 points of time in view,
  about 4.5 deg), and the brake plus sidestep layer cuts obstacle collisions from 15 to 19 % of episodes to about 4 %
  (48 seeds). Braking behind tall obstacles can hide the user long enough to trigger the lost-target landing.
- The plan's absolute gate G2 (70 % in band on the demo set) is not met by any arm for the distance reason above; the
  relative version (fly within the tuned PID's range, no collisions) is met.

## 6. Biology vs engineering

| Biology (from the connectome and model) | Engineering (our choices) |
|---|---|
| Which neurons exist, cell types, sides | Camera-to-neuron encoding (bearing and size to LC10a bins) |
| Wiring and synapse counts | Readout from DN rates to the yaw stick |
| Neurotransmitter signs (excitatory / inhibitory) | PID distance loop (forward speed) |
| Spiking LIF dynamics (Shiu et al. 2024) | Safety governor, obstacle avoidance |
| The LC10a -> AOTU -> DNa02 push-pull pathway | Fly body animation on the demo screen |

The fly body on screen is TuragaLab's flybody model of a **female** fly, posed each frame from the brain's outputs
(on-screen label: "Animated from the fly brain's live motor outputs; not a physics simulation"). The connectome is male.

## 7. How it plugs into the drone

- `flyfollow.steer.FlySteer` is a drop-in yaw controller for the ReachGlass stack (patch:
  `docs/integration/reachglass_flysteer.patch`, guide: `docs/integration/REACHGLASS.md`). The host passes the target
  bearing (and range) each loop and gets back one yaw stick.
- ReachGlass keeps perception, target lock, distance, altitude, safety, FIND and approach moves; the fly sets yaw.
  `steering: pid` restores the original yaw law, and the stack falls back to it if the fly fails to load or raises.
- The trained interface is one file: the run's `best.json` (set `fly.params_path`).
- Before flying: run the Tello I/O in dry run (the default, sends no rc but shows what it would send) and check the
  sign: a person on the right of the image must give a positive (clockwise) yaw.

## 8. Limitations and next steps

- **Sim-to-real.** Tello dynamics were measured only at stick 30; larger sticks are extrapolated. The camera FOV
  still needs confirming at R0. All results above are simulation-only until flight tests.
- **Readout noise.** One DN cell per side gives about 7 deg of bearing noise per tick; this is the main cost versus the
  no-brain controller.
- **Azimuth bins are by rank.** MaleCNS has no retinotopic coordinates for LC10a, so the 8 bins per side are rank
  splits, not real positions on the eye; the trained per-bin gains compensate.
- **No rear sensing.** The drone cannot see a person stepping back into it; the reverse speed cap was raised (stick 40)
  and needs a spotter at first flights.
- **Circuit depth.** The usable core is a 2-synapse relay; the deeper core runs away without changing the fixed LIF parameters.
