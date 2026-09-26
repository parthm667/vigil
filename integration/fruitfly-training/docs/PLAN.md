# ReachGlass Scout Drone: Learned Object Search with a Fly-Wired Policy

**Planning document** · Hackathon build, 30-hour window · Draft for team review · 2026-09-26

> **Superseded.** The team changed scope: search is now a scripted algorithm, not RL, and the RL work is the fly-brain follow and approach controller. The current plan is [DRONE_RL_PLAN.md](DRONE_RL_PLAN.md). This file is kept for history.

Background research: [research/fruitfly-brain-rl.md](../research/fruitfly-brain-rl.md)

---

## 0. Summary

ReachGlass uses a DJI Tello to survey a room so that a visually impaired user can ask for an object and be guided to it by tactile cues on a pair of sunglasses. In the current design, the Tello flies a preset path and the vision system labels whatever the camera happens to see. However, the Tello has roughly 13 minutes of flight time per battery, and a fixed sweep spends most of that time looking at walls and empty floor.

In this project, we replace the preset path with a learned search policy that decides where the drone should look next. We build the core of that policy from the mushroom body of the fruit fly, the brain structure that learns which odors predict reward, using the real wiring from the FlyWire connectome. We train the policy in a simple 2D room simulator on Modal. We then compare it against hand-coded search and against matched networks that do not use the fly wiring. Finally, we run it on the real drone behind a hard-coded safety layer.

---

## 1. What the RL is for

### 1.1 The problem with the preset path

The preset path is the obvious first design because it is simple and it guarantees coverage. However, it ignores everything the drone learns while flying. If the user asks for a water bottle and the first thing the drone sees is a kitchen counter, a sweep still finishes the far wall before it comes back. With about 13 minutes per battery and several searches needed per demo, that wasted flight time matters.

### 1.2 Why search is the part to learn

The search problem has structure that is hard to hand-code but easy to learn from many simulated rooms. Water bottles are usually on tables, desks, and counters. Keys are usually on tables or near doors. Backpacks are usually on the floor or on chairs.

A good searcher uses these priors, avoids viewpoints it has already covered, and trades off how far away a viewpoint is against how likely it is to show the target. This is also a sequential decision, since the best next viewpoint depends on what has already been seen. That is what separates it from a lookup table and makes it an RL problem.

Chaplot et al. [1] showed that a specific split works well for household object search. In that split, a learned policy picks long-term goals on a semantic map, and a classical planner handles the motion between them. We use the same split.

### 1.3 What is not RL

RL makes exactly one decision in this system: **which viewpoint the drone visits next**. Everything around it is classical code, and that is deliberate. It keeps the learned part small enough to train in hours, and it lets us swap in a heuristic if the policy fails.

| Component | Approach | Reason |
|---|---|---|
| Position hold, low-level flight | Tello's built-in downward vision positioning | Already stable; nothing to learn |
| Motion between viewpoints | A* on the grid + Tello SDK moves | Deterministic and easy to debug |
| Object detection | Pretrained detector (CV team) | Supervised problem, not RL |
| Request parsing | Grok | Language problem |
| Obstacle stop on the glasses | ToF distance threshold | A blind user needs predictable safety behavior |
| Drone safety | Hard-coded shield (Section 7.3) | A learned policy should never be the only thing keeping a drone away from a person |

---

## 2. Where the policy sits

```
User speech ──> Grok ──────────────> target class ("bottle")
                                               │
Tello video ──> Detector ──> detections ──┐    │
Tello state ──> Pose tracker ─────────────┤    │
Room outline (measured) ──────────────────┤    │
                                          ▼    ▼
                               Semantic map (2D grid)
                                          │
                              Candidate viewpoint generator
                                          │
                        Search policy (RL, fly-wired core)
                                          │
                                   Safety shield
                                          │
                         Planner ──> Tello SDK commands
                                          │
              found ──> target location (room frame) ──> glasses guidance
```

### 2.1 Interfaces

We propose these message formats so that each team can build against them independently.

**Detections** (CV → map builder), one message per frame:

```json
{"t": 1727380000.12, "frame_id": 812,
 "detections": [{"cls": "table", "conf": 0.87, "bbox": [412, 300, 690, 455]}]}
```

**Pose** (pose tracker → map builder): `{"t", "x_m", "y_m", "z_m", "yaw_deg"}`

**Target** (Grok → policy): `{"cls": "bottle"}`

**Result** (policy → glasses guidance): `{"cls": "bottle", "x_m": 2.4, "y_m": 1.1, "conf": 0.91, "t": ...}`

**Room frame:**
- The origin is the takeoff point.
- +x is the direction the drone faces at takeoff, and +y is to its left.
- Units are meters.

We own the map builder on the RL side, so the CV team only has to deliver detections. If the CV team would rather deliver a finished map, the policy does not change, because it only ever reads the map.

### 2.2 The Wi-Fi problem

The standard Tello creates its own Wi-Fi network, and the laptop has to join that network to send commands and receive video. However, while the laptop is on the Tello's network it has no internet, which means it cannot reach the Grok API or any Modal endpoint. This affects the whole team, not just the RL workstream.

There are three ways around it:
1. **A second network adapter.** Use a USB Wi-Fi dongle, or a USB or Ethernet tether to a phone. One interface talks to the Tello and the other reaches the internet.
2. **A Tello EDU.** It can join an existing router in station mode.
3. **Run everything offline.** Everything the live demo needs would run locally on the laptop.

We plan for option 1. We also design the system so that the search policy runs locally no matter what, since the policy network is small enough to run on a laptop CPU. Modal is only used for training. **This needs to be tested in the first hour.**

---

## 3. The search task

### 3.1 Simulator

The obvious simulator would be Habitat, since that is where most object-search results come from. However, Habitat takes hours to set up and is slow per step compared to a grid. It also produces rendered images, and our policy never sees images. The policy only sees a semantic map, so the simulator only has to produce realistic semantic maps. A 2D top-down grid does that, and it runs thousands of rooms in parallel.

**Rooms:**
- Each side is sampled from 4–10 m, on a 0.25 m grid.
- Each room gets 3–8 furniture items from {table, chair, couch, counter, desk, shelf, bed, door}, each with a realistic footprint.
- 1–3 target objects are placed using class-conditional priors, plus distractor objects.
- The user is placed at a random free cell and gets a keep-out radius.

**Placement priors** (sampling weights):

| Target | Where it is placed |
|---|---|
| Water bottle | table 0.4, counter 0.3, desk 0.2, floor 0.1 |
| Mug | table 0.4, desk 0.3, counter 0.3 |
| Keys | table 0.3, near door 0.3, counter 0.3, floor 0.1 |
| Phone | couch 0.3, table 0.3, desk 0.3, bed 0.1 |
| Backpack | floor 0.5, chair 0.3, couch 0.2 |

These priors are our estimates, not measured household statistics. Since the policy learns whatever priors the simulator encodes, the demo room has to be set up roughly consistently with this table. Otherwise the learned policy has no advantage over plain exploration. This is a limitation we should state in the presentation rather than hide.

**Sensor model:**
- Horizontal field of view: 82.6° (the Tello camera spec).
- Visibility comes from ray casting on the grid, with occlusion by walls and tall furniture.
- Detection probability is high within 1.5 m and falls linearly to zero at a maximum range. That range is sampled per episode, 3–5 m for furniture and 2–3.5 m for small targets.
- The model includes a small false-positive rate.

These detection numbers are placeholders. Between T5 and T10, we will measure the CV team's actual detection rates on Tello video and replace them. This is the main sim-to-real adjustment, because the policy never sees raw pixels.

### 3.2 Decisions, observations, and rewards

At each decision, a generator proposes up to K = 16 candidate viewpoints, each a position plus a heading. The candidates come from two sources:
- **Frontier viewpoints**, facing the boundary between seen and unseen free space.
- **Furniture viewpoints**, 1–2 m from each detected furniture item and facing it.

The policy picks one candidate. The planner flies there with A* on the grid. The drone then captures three frames at headings of −30°, 0°, and +30° relative to the viewpoint heading, which covers about 140°.

**Per-candidate features** (about 40 values):
- target class (one-hot);
- path distance from the drone;
- unseen area that would become visible from the viewpoint;
- counts of each furniture class near the viewed region;
- how many times that region has already been viewed;
- distance to the user and to the nearest wall.

**Global features:** decisions used and remaining budget.

**Reward:**
- +10 when the target is confirmed, meaning detected in at least 2 of the 3 frames.
- −0.1 per meter flown and −0.2 per decision, to represent time and battery cost.
- −10 if the budget runs out without finding the target.

Unsafe candidates are masked out by the shield, so there is no collision penalty to tune.

**Episode length:** an episode ends when the target is found or the budget runs out. The budget is 25 decisions or 40 m of flight, which is roughly 3 minutes of Tello time.

### 3.3 Why candidate selection instead of raw moves

We considered letting the policy output raw moves (forward 0.5 m, turn ±30°). However, the horizon then becomes hundreds of steps and credit assignment gets much harder. Every motion error in the sim-to-real transfer would also compound.

With candidate selection, episodes are 5–25 decisions long, and the policy's job is exactly the question we care about: where to look next. The motion comes from a planner that we can test separately.

The cost is that the policy can only pick from what the generator proposes, so a bad generator caps performance. We check this with an oracle that always picks the candidate that finds the target soonest. If the oracle does badly, the problem is the generator, not the policy.

---

## 4. Policy architectures

All four networks take the same inputs (the candidate feature vectors) and produce the same outputs: a score per candidate for the actor, and a value for the critic. They differ only in the network in between.

The idea we borrow from the fly is simple. In the mushroom body, about 50 olfactory input channels (glomeruli) are expanded onto about 2,000 Kenyon cells (KCs) per hemisphere. Each KC samples about 7 inputs [6]. Only about 5% of KCs fire for any given odor [5]. Output neurons (MBONs) then learn which KC patterns predict reward. Scoring a candidate viewpoint is the same kind of problem: take a description of a place and decide how likely it is to pay off.

| ID | Network | Input → hidden | Hidden → output |
|---|---|---|---|
| **A** | MLP | Two dense hidden layers, width chosen so the trainable parameter count matches C | Dense |
| **B** | MB-random | Features → ~50 "PN" units (trainable, ReLU) → ~2,000 KCs through a **fixed random sparse** matrix (7 inputs per KC) → keep top 5% | Trainable KC→MBON (~35 units) → score and value |
| **C** | MB-FlyWire | Same as B, but PN→KC is the **real FlyWire wiring** (glomerulus × KC synapse counts, one hemisphere) | KC→MBON masked to the **real** compartment connectivity |
| **D** | MB-shuffled | C's matrices, rewired so each neuron keeps its in-degree and out-degree | Same shuffle applied to the mask |

The pairs answer different questions:
- **B vs A** tests the motif, meaning whether sparse expansion helps at all.
- **C vs D** tests whether the fly's actual wiring adds anything beyond its degree statistics. Dhiman [11] found this control missing from most connectome-constrained studies, and found that the connectome's advantage disappeared once it was added.
- **C vs B** compares real wiring with purely random wiring.

**Our prediction is that B, C, and D will perform about the same.** There are two reasons:
1. PN→KC wiring in the fly is close to random [7], with only some structured biases found in EM data [8].
2. Our input channels do not mean what glomeruli mean to a fly. The PN layer learns an arbitrary assignment of map features to "glomeruli."

So if C beats D, that is interesting. If it does not, that is the honest answer, and it is still a result we can show. We are stating this prediction before training so that we do not fit the story to the chart afterward.

### 4.1 Extracting the mushroom body from FlyWire

1. Download the FlyWire v783 proofread connections from Zenodo (`proofread_connections_783.feather`, about 852 MB, CC-BY 4.0) and the neuron annotation table from `flyconnectome/flywire_annotations`.
2. Select one hemisphere and pull out four groups:
   - uniglomerular antennal lobe projection neurons, grouped by glomerulus;
   - Kenyon cells, split by subtype (γ, α/β, α′/β′);
   - MBONs, by type;
   - DANs (PAM and PPL1), used only by the stretch variants.
3. Build the glomerulus × KC matrix from synapse counts. Start with a threshold of 5 synapses and check that the results are not sensitive to it.
4. Build the KC → MBON connectivity mask from the same table.
5. Save both as sparse `.npz` files in `data/` and run a sanity check that prints:
   - the number of KCs (should be about 2,000 for one hemisphere);
   - the number of glomeruli (about 50);
   - the mean glomerular inputs per KC (should be near 7).

The exact class names in the annotation file have to be confirmed on first load. The sanity check numbers are how we know the extraction is correct.

### 4.2 Training

All variants use PPO [14]. The actor is a masked softmax over candidate scores. To keep the comparison fair, every architecture gets the same hyperparameter search budget: 8 configurations × 3 seeds, varying the learning rate and entropy coefficient. Each architecture's best configuration is then retrained with 10 fresh seeds. Without equal tuning, the comparison would mostly measure which network we spent more time tuning.

### 4.3 Fly-inspired extras, in priority order

1. **Novelty bonus.** This is modeled on MBON-α′3, a mushroom body output neuron that responds to new odors and habituates with repetition [9].
   - We hash each viewed region through the KC code and count how many times each hash has been seen.
   - The intrinsic reward decays with the count.
   - We test it on and off for the best architecture.
2. **Compartment critics.** The fly's mushroom body has parallel compartments with different learning and forgetting rates [10]. We mimic this with three value heads using different discount factors (γ = 0.8, 0.95, 0.99), combined by their mean. This is a stretch item.
3. **Local three-factor learning.** After PPO pretraining, freeze everything except KC → MBON and train it with the rule Δw = η · δ · KC_active instead of backpropagation. This is closer to how dopamine-gated plasticity works in the fly. It is a stretch item and mostly adds to the story.
4. **Whole-brain policy.** Use the full FlyWire graph as a message-passing network, following FlyGM [12]. This is the riskiest item and only happens if checkpoint C3 passes with time to spare.

---

## 5. Training on Modal

**Setup:**
- One Modal app built on an image with `torch`, `numpy`, `pandas`, and `pyarrow`.
- A Modal volume holds the FlyWire files, checkpoints, and results.
- A `train(config)` function is launched in parallel with `.map` over a list of configs.
- Results are written to the volume as JSON and CSV, and a local script makes the plots.

**Throughput benchmark (T3):** we measure decisions per second for the simulator on CPU and on GPU before choosing hardware. The networks are small (under about 100k trainable parameters), so the simulator is likely to be the bottleneck rather than the GPU. In that case, many cheap CPU containers would beat a few H100s. We decide from the benchmark instead of assuming.

**Budget:**

| Item | Runs | Estimated cost |
|---|---|---|
| Debugging and smoke tests | n/a | $100 |
| Hyperparameter search (4 architectures × 8 configs × 3 seeds) | 96 | $300 |
| Main comparison (4 architectures × 10 seeds) | 40 | $150 |
| Ablations (novelty bonus, compartment critics) | 40 | $150 |
| Detector fine-tuning and serving (CV team) | n/a | $300 |
| Whole-brain stretch | ~10 | $1,000 |
| **Total planned** | | **~$2,000** |

These estimates assume roughly $4 per H100-hour and 30-minute runs, and they will be corrected after the T3 benchmark. Even with a large error, the plan uses about 20% of the $10,000 in credits. So compute is not our constraint; the 30 hours are. Wherever possible, we spend compute to save people's time.

In practice, that means running larger sweeps instead of hand-tuning. It also means scheduling the main sweep so that it runs while the team sleeps.

---

## 6. Evaluation

### 6.1 Hypotheses

- **H1.** The learned policy finds the target with higher SPL than the preset sweep and frontier exploration on held-out rooms.
- **H2.** The learned policy beats the semantic greedy baseline (Section 6.2). This is the hard baseline. If RL cannot beat it, the RL is not earning its place over a lookup table.
- **H3.** The MB-random network (B) is more sample-efficient than the MLP (A) at the same number of trainable parameters.
- **H4.** The real FlyWire wiring (C) versus the degree-preserving shuffle (D). Our prediction is no meaningful difference (Section 4).

### 6.2 Non-learned baselines

1. **Preset sweep.** The team's current plan: a lawnmower path with a look at each waypoint.
2. **Frontier exploration** [3]. Always go to the nearest frontier.
3. **Semantic greedy.** Rank candidates by the prior probability of the target near the furniture in view, divided by distance.
4. **Oracle.** Pick the candidate that finds the target soonest. This only exists in simulation and gives an upper bound.

### 6.3 Metrics

- **Success rate** within the budget.
- **SPL** [2]: success weighted by the ratio of shortest path length to actual path length. It rewards finding the target and doing it efficiently.
- **Distance and decisions to find** the target.
- **Shield interventions.** These should be zero, because unsafe candidates are masked. A nonzero count means a bug.
- **Sample efficiency.** Success rate plotted against environment decisions during training.

### 6.4 Test sets

- **Held-out rooms:** 1,000 procedurally generated rooms from a fixed seed that is never used in training.
- **Demo-room replica:** a copy of the real demo room, measured with a tape measure, with the real target placements.
- **Stress tests:** the same held-out rooms with the detector miss rate doubled and the pose noise increased.

**Statistics:** each architecture gets 10 seeds, and we report the mean with a 95% bootstrap confidence interval across seeds. With 10 seeds we can only detect fairly large differences, so we will say that plainly instead of calling small gaps significant.

### 6.5 Real-world trials

In the demo room, we fly 3 target placements × 2 methods (RL policy and preset sweep) × 2 trials each, for 12 flights. For each flight we record time to find, distance flown, and success. At about 3 minutes per search this fits in 3–4 batteries, plus setup. With 12 flights, we report the raw times and do not claim statistical significance.

### 6.6 Success criteria

- **Must have (by T20):**
  - A non-learned search (semantic greedy or frontier) runs safely end to end on the real drone and hands off to the glasses.
  - The RL policy is trained in simulation, and the A/B/C/D comparison chart is done.
- **Target:** the RL policy flies the real search and finds the target faster than the preset sweep in at least 4 of 6 paired trials.
- **Stretch:** the novelty and compartment-critic ablations, and the whole-brain policy in simulation.

---

## 7. Real-world deployment and safety

### 7.1 Pose

The Tello does not report its absolute position. Instead, we dead-reckon from the commands we send, such as "forward 50 cm," and correct yaw and height using the Tello state stream on UDP port 8890.

Dead reckoning drifts, so each search has to stay short (about 3 minutes), and we reset to zero at every takeoff. If the team has a Tello EDU, its mission pads can give absolute position fixes.

### 7.2 Map builder

**Room outline.** We measure the room outline and preload it instead of detecting walls. This takes wall detection off the critical path. Wall detection from the CV team can replace the preloaded outline later if there is time.

**Furniture.** We project the bottom-center of each bounding box onto the floor. This uses the camera intrinsics, the drone's height h, and a flat-floor assumption:

  d = h / tan(α)

Here, α is the angle below the horizon of the bottom pixel.

**Targets on furniture.** The flat-floor projection overestimates the distance to a bottle on a table, since the bottom of its bounding box sits on the table, not the floor. So when a target's box sits inside or on top of a furniture box, we place the target at that furniture's map position instead.

**Calibration.** We calibrate the camera intrinsics with a checkerboard and OpenCV, which takes about 20 minutes. During the same step, we measure the camera's pitch offset.

### 7.3 Safety shield (hard-coded, not learned)

1. **Geofence.** Every viewpoint and planner waypoint must be at least 0.5 m inside the measured room outline.
2. **User keep-out.** No waypoint within 1.5 m of the user, and no path segment that crosses that circle.
3. **Fixed altitude.** Fly at about 1.0 m, which is high enough to see tabletops and below a standing person's head.
4. **Low speed.** Use a slow SDK speed setting, starting at `speed 30` (cm/s).
5. **Watchdogs.**
   - If video or state is lost for more than 1 s, stop sending moves and hover.
   - If it is lost for more than 5 s, land.
   - The Tello also lands on its own after about 15 s without a command, so the runtime sends keep-alive commands while the policy is thinking.
6. **Battery.** Below 25%, land.
7. **Human kill switch.**
   - A teammate holds a dedicated key that sends `land`.
   - A second key sends `emergency`, which cuts the motors, as a last resort.
8. **Takeoff position.** The drone takes off at least 2 m from the user and never flies toward the user's face.

Before any indoor flight, we check the venue's drone rules and get permission from the organizers.

### 7.4 Runtime loop

1. Grok returns the target class.
2. The drone takes off, and the pose resets to the origin.
3. The drone does an initial three-frame look, and the map is updated.
4. The generator proposes candidates, and the shield masks out unsafe ones.
5. The policy scores the candidates and picks one.
6. The planner flies to it in segments of 50 cm or less and turns to the viewpoint heading.
7. The drone does a three-frame look, and the map is updated.
8. If the target is confirmed in at least 2 of the 3 frames, the drone publishes the result to the glasses and lands. Otherwise, the loop returns to step 4.

The drone is hovering whenever a decision is made, so video latency of a few hundred milliseconds does not affect control. It only adds a small delay per decision.

The operator screen shows the live map, the candidates, and the policy's score for each one. That display is our main debugging tool, and it also lets judges see the decisions being made.

---

## 8. Schedule

T0 is the start of the build. The RL workstream is 1–2 people working in parallel with the hardware and CV teams.

| Time | RL and simulator | Real-drone integration | Checkpoint |
|---|---|---|---|
| T0–T1 | Finalize plan and interfaces, Modal login, repo scaffold | Test the Wi-Fi and internet setup with the Tello; ask organizers about drone rules | Interfaces agreed |
| T1–T5 | Simulator, candidate generator, baselines 1–4, PPO with the MLP on CPU; T3 throughput benchmark | `djitellopy` wrapper, state stream, kill switch | **C1** |
| T5–T10 | FlyWire extraction, MB networks B–D, Modal sweep scripts; launch the hyperparameter search | Camera calibration, pose tracker, map builder tested on recorded Tello video; measure detection rates | |
| T10–T14 | Main sweep running; analysis and plot scripts | Shield and planner; fly semantic greedy in the real room | **C2** |
| T14–T20 | Results, novelty ablation, demo-room replica evaluation | Deploy the RL policy on the drone; tune detector thresholds | **C3** go/no-go |
| T20–T24 | Stretch items, only if C3 passed | Harden the fallback; repeat trials | |
| T24–T27 | Code freeze; final figures | Record a backup demo video | Freeze |
| T27–T30 | Slides, rehearsal, buffer | | |

**Checkpoint C1 (T5).**
- *Pass:* the simulator runs, PPO with the MLP clearly beats random selection, and semantic greedy works.
- *Fail:* simplify the simulator by using fewer classes and smaller rooms before adding the fly networks.

**Checkpoint C2 (T14).**
- *Pass:* the drone completes a safe end-to-end search in the real room using a non-learned baseline. This is the most important checkpoint, because it proves the pipeline works without any RL.
- *Fail:* everyone who can help moves to integration, and the RL results become simulation-only.

**Checkpoint C3 (T20).**
- *Go:* the RL policy has completed at least 3 safe real flights and found the target. The RL policy flies in the live demo.
- *No-go:* semantic greedy flies the live demo, and the RL results are shown from simulation and replays.

Whenever the team sleeps, a sweep should be running.

---

## 9. Demo narrative

1. The user asks for their water bottle.
2. The drone launches. The screen shows the live map, the candidate viewpoints, and the policy's scores. The drone goes to the table first instead of sweeping the room.
3. The drone finds the bottle, and the glasses guide the user to it.
4. **Slide:** the simulation comparison of preset sweep, frontier, semantic greedy, MLP, MB-random, MB-FlyWire, and MB-shuffled, with the honest result for FlyWire versus shuffled.
5. **Slide:** the FlyWire mushroom body wiring that sits inside the policy.

---

## 10. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No internet while on Tello Wi-Fi | High | Blocks Grok and Modal during the demo | Second network adapter; test at T0 |
| Detector unreliable on Tello video (blur, compression, lighting) | High | Wrong map, bad decisions | Measure detection rates at T5–T10 and put them into the simulator; fine-tune on demo-room images |
| Pose drift | Medium | Map errors that grow over a flight | Short searches, slow speed, reset at every takeoff |
| RL does not beat semantic greedy | Medium | Weaker story | Report it honestly; semantic greedy still flies the demo |
| Simulator priors do not match the demo room | Medium | Learned priors do not help | Place demo objects plausibly; evaluate on the replica first |
| FlyWire extraction takes longer than planned | Low–Medium | C and D arrive late | B does not depend on FlyWire, so the pipeline can run without it |
| Not enough batteries | Medium | Fewer real trials | Plan a charging schedule; test the pipeline on recorded video |
| Venue drone rules | Unknown | Could block live flight | Ask organizers at T0; the backup video covers the worst case |

---

## 11. Repository layout

```
fruitfly-training/
├── research/fruitfly-brain-rl.md   background research
├── docs/PLAN.md                    this document
├── sim/          room generator, sensor model, candidate generator, batched env
├── baselines/    preset sweep, frontier, semantic greedy, oracle
├── policies/     mlp.py, mushroom_body.py, flywire_loader.py, nulls.py
├── train/        ppo.py, modal_app.py, configs/
├── eval/         evaluate.py, plots.py
├── deploy/       tello_io.py, pose.py, map_builder.py, shield.py, planner.py, run_search.py
└── data/         FlyWire cache, calibration files (gitignored)
```

---

## 12. Open decisions for the team

1. Is the drone a standard Tello or a Tello EDU? The EDU adds station mode and mission pads.
2. Do we have a second network adapter or a phone we can tether?
3. Which detector will we use: a local YOLO model, or an open-vocabulary model served from Modal? Which target classes will the demo use?
4. Does the proposed detections JSON work for the CV team, or would they rather deliver a map?
5. Can we measure the demo room and fly in it, and have the organizers approved indoor flight?
6. How many batteries do we have?
7. Who owns each workstream?

---

## References

[1] D. S. Chaplot, D. Gandhi, A. Gupta, R. Salakhutdinov, "Object Goal Navigation using Goal-Oriented Semantic Exploration," *NeurIPS*, 2020.
[2] P. Anderson et al., "On Evaluation of Embodied Navigation Agents," arXiv:1807.06757, 2018.
[3] B. Yamauchi, "A Frontier-Based Approach for Autonomous Exploration," *IEEE CIRA*, 1997.
[4] S. Dorkenwald et al., "Neuronal wiring diagram of an adult brain," *Nature* 634, 2024; P. Schlegel et al., "Whole-brain annotation and multi-connectome cell typing of Drosophila," *Nature* 634, 2024. Data: https://zenodo.org/records/10676866
[5] S. Dasgupta, C. F. Stevens, S. Navlakha, "A neural algorithm for a fundamental computing problem," *Science* 358, 2017.
[6] A. Litwin-Kumar, K. D. Harris, R. Axel, H. Sompolinsky, L. F. Abbott, "Optimal Degrees of Synaptic Connectivity," *Neuron* 93, 2017.
[7] S. J. C. Caron, V. Ruta, L. F. Abbott, R. Axel, "Random convergence of olfactory inputs in the Drosophila mushroom body," *Nature* 497, 2013.
[8] Z. Zheng et al., "Structured sampling of olfactory input by the fly mushroom body," *Current Biology* 32, 2022.
[9] D. Hattori et al., "Representations of Novelty and Familiarity in a Mushroom Body Compartment," *Cell* 169, 2017.
[10] Y. Aso, G. M. Rubin, "Dopaminergic neurons write and update memories with cell-type-specific rules," *eLife* 5:e16135, 2016.
[11] N. Dhiman, "Topological Sensitivity in Connectome-Constrained Neural Networks," arXiv:2604.04033, 2026.
[12] Jin et al., "Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly," arXiv:2602.17997, 2026.
[13] Lu, Webb, "Insect-inspired Visual Point-goal Navigation," arXiv:2601.16806, 2026.
[14] J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
[15] Ryze Tech, *Tello SDK 2.0 User Guide*.
