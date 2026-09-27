# Fruit fly brain × RL — research notes (as of 2026-09-26)

Compiled from web research. Items marked **[unverified]** could not be confirmed against a primary source.

## 1. Connectome data

| Dataset | Scope | Size | Access | License |
|---|---|---|---|---|
| **FlyWire / FAFB v783** (Dorkenwald+ 2024, Schlegel+ 2024, Nature) | Female whole brain | 139,255 neurons, ~50M synapses, 8,453 cell types, NT predictions (Eckstein+ 2024 Cell) | [Codex](https://codex.flywire.ai), [Zenodo 10676866](https://zenodo.org/records/10676866) (`proofread_connections_783.feather`, 852 MB), [annotations](https://github.com/flyconnectome/flywire_annotations), `fafbseg`, `caveclient`, `navis` | CC-BY 4.0 (Zenodo) |
| **Hemibrain v1.2.1** (Scheffer+ 2020 eLife) | Female central brain (partial) | ~25k neurons | [neuPrint](https://neuprint.janelia.org), `neuprint-python` | open |
| **MANC v1.2.3** (Takemura/Marin+ 2024 eLife) | Male VNC | ~15.8k neurons | neuPrint | open |
| **Male optic lobe** (Nern+ 2025 Nature) | Male optic lobe | — | neuPrint `optic-lobe:v1.1` | open |
| **Male CNS v1.0** (Berg+ 2026 Cell, doi:10.1016/j.cell.2026.08.015) | Male brain + optic lobes + VNC, neck intact | 166,691 neurons, ~11.7k types | neuPrint `male-cns:v1.0`, [male-cns.janelia.org](https://male-cns.janelia.org) | CC-BY 4.0 |
| **BANC** (Bates+ 2026 Nature, doi:10.1038/s41586-026-10735-w) | Female brain + nerve cord | ~188k neurons (some sources say 142k proofread) **[check]** | Codex, [Dataverse](https://doi.org/10.7910/DVN/7WTH1N), `pip install banc`, [repo](https://github.com/htem/BANC-project) | CC-BY 4.0 |

Ecosystem overview: https://flyconnecto.me/2026/09/04/the-adult-drosophila-connectome-ecosystem/

**Practical:** The FlyWire edge list fits comfortably on a laptop as a sparse 139k×139k matrix. The full synapse table (9.5 GB) needs 32 GB+ RAM.

## 2. Whole-brain / circuit simulation

- **Shiu+ 2024 Nature**: LIF model of all FlyWire neurons. Sign comes from NT and weight from synapse count; there are **no fitted params**. It predicted feeding and grooming circuits (~91% agreement with experiments). Code: [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (Brian2, MIT). GPU ports: [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) (Brian2CUDA / PyTorch / GeNN / NEST GPU), [annel0/flybrain](https://github.com/annel0/flybrain) (Triton).
- **Lappalainen+ 2024 Nature (flyvis)**: connectome-constrained optic-lobe model with 64 cell types and ~45k neurons. The wiring is fixed and the cell/synapse params are trained by backprop. Its T4/T5 direction selectivity emerges from training. `pip install flyvis` ([TuragaLab/flyvis](https://github.com/TuragaLab/flyvis)), with pretrained ensembles.
- Review: *State of Brain Emulation Report 2025*, arXiv:2510.15745.

## 3. Fly bodies (simulators)

| | NeuroMechFly v2 / **FlyGym** | **flybody** |
|---|---|---|
| Paper | Wang-Chen+ 2024 Nat Methods | Vaxenburg+ 2025 Nature 643:1312 |
| Engine | MuJoCo, Gymnasium API | MuJoCo / dm_control |
| Senses | Compound eyes (~721 ommatidia/eye), odor plumes, mechanosensory | Vision (low-res) |
| Behaviors | Walking (CPG + feedback), odor tracking, path integration, multimodal nav | Walking, flight (with fluid model), vision-guided flight |
| RL used | SAC (SB3) for the high-level decision MLP | DMPO (Acme/TF + Ray), imitation of real fly kinematics |
| Connectome | flyvis visual system run in closed loop (frozen) | None |
| Compute | CPU ~2× realtime; GPU (MJWarp) ~60× realtime | ~1e8–1e9 env steps, many CPU actors + 1 GPU |
| Repo | [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym) (Apache-2.0) | [TuragaLab/flybody](https://github.com/TuragaLab/flybody) (Apache-2.0) |

⚠️ FlyGym 2.x (Mar 2026) is a rewrite that is **not backward compatible**. The old Gymnasium API now lives in `flygym-gymnasium`.

## 4. How people combine the connectome with RL

The approaches fall into three buckets:

**A. Fixed wiring, trainable node or readout params, trained with RL**
- **FlyGM** (Jin+ 2026, arXiv:2602.17997): FlyWire v783 serves as a directed message-passing graph, with signs taken from NT and a trainable per-neuron descriptor. It uses imitation pretraining followed by PPO on the flybody body for walking, turning and flight. It reports better sample efficiency than ER, degree-preserving rewired, MLP, GCN and GAT baselines. It was trained on an A100, and **no code is public**.
- **FLYNN** (Wang & Chen 2026, arXiv:2607.00025): FlyWire as a sparse RNN trained with DAgger (imitation) to drive a wheeled robot from vision. It is more robust to OOD input than CNNs.
- Hobbyist: [flydoom](https://github.com/eganeganegan/flydoom) (MaleCNS subgraphs + PPO or three-factor rule on VizDoom, with ER and degree-preserving baselines; ~12 GB GPU).

**B. Fixed-weight emulation with no learning, and hand-mapped motor outputs**
- **Eon Systems** (Mar 2026): Shiu LIF + flyvis in a NeuroMechFly v2 body. It shows grooming, feeding and foraging, with a handful of descending neurons **mapped by hand** to imitation-trained leg controllers. Eon itself says this does not show that structure alone is sufficient. Embodied code not public **[unverified]**.
- Hobbyist: [Fly-Brain-AI](https://github.com/neilt93/Fly-Brain-AI), [erojasoficial-byte/fly-brain](https://github.com/erojasoficial-byte/fly-brain) (claims unreviewed).

**C. Connectome as a weak structural prior**
- FlyCNS (arXiv:2609.28816): BANC ascending/descending structure → SVD allocation bias for a quadruped PPO controller.

**Counter-evidence to take seriously**
- **Dhiman 2026** (arXiv:2604.04033, [code](https://github.com/nalin-dhiman/Connectome-Constrained-Neural-Networks)): "connectome beats random" advantages disappear once you control for **degree-preserving nulls + matched initialization**. The task was supervised, not RL, but any connectome-RL claim should be tested against those controls.

No formal benchmark or competition for connectome + RL + body exists yet.

## 5. Mushroom body (MB) ≈ the fly's RL machinery

**Circuit:** ~2,000 Kenyon cells (KCs) per hemisphere form a sparse, expanded odor code. KC→MBON synapses are tiled into ~15 compartments, and each compartment has its own dopamine neurons (DANs): PAM for reward, PPL1 for punishment. Coincident KC activity and DA **depresses** KC→MBON synapses. MBON→DAN feedback closes the loop.

**Key findings**
- **Aso & Rubin 2016 eLife**: compartments behave like parallel learners, each with its own learning and forgetting rate.
- **Handler+ 2019 Cell**: the sign of plasticity depends on timing. Odor→DA gives LTD (DopR1) and DA→odor gives LTP (DopR2), which works like a signed eligibility trace.
- **Hattori+ 2017 Cell**: MBON-α'3 is a built-in novelty detector that habituates with exposure.
- **Bennett, Philippides & Nowotny 2021 Nat Commun**: a DAN = reinforcement − MBON-prediction RPE model fits 439 experimental outcomes. It is Rescorla-Wagner-level, **not full TD**. [code](https://github.com/BrainsOnBoard/paper_RPEs_in_drosophila_mb)
- **Jiang & Litwin-Kumar 2021 PLOS CB**: meta-learns the MBON→DAN feedback in an outer loop, with only local DA-gated plasticity in the inner loop. RPE emerges as the dominant population mode. [code](https://github.com/alitwinkumar/jiang_litwin-kumar_mb_rnn)
- **Eschbach+ 2020 Nat Neurosci**: the larval connectome shows extensive MBON→DAN feedback motifs, which support prediction-error-like computation.
- **Yamada+ 2023 eLife**: second-order conditioning (a TD signature) works through a slow compartment "teaching" fast compartments.
- **Brembs & Plendl 2008 Curr Biol**: pure operant (action→outcome) learning uses PKC and **not** the rutabaga/cAMP MB pathway. So the MB is closer to a critic or associative memory than the whole agent.
- **Verdict:** DA signals in flies look like RPEs. Continuous-time TD is **plausible but unestablished**.

## 6. Fly-inspired ML algorithms

- **FlyHash**: Dasgupta, Stevens & Navlakha 2017 Science. Sparse random 40× expansion + 5% k-WTA gives an LSH that beats SimHash. Learned variant: BioHash (ICML 2020).
- **FlyBloomFilter**: Dasgupta+ 2018 PNAS. Novelty detection via KC→MBON depression.
- **FlyModel continual learning**: Shen, Dasgupta & Navlakha 2023, *Neural Computation* 35(11). A sparse KC code where only synapses from active KCs are updated. It beats EWC, GEM and BI-R on class-incremental learning.
- **FlyVec**: Liang+ ICLR 2021. Sparse binary word embeddings. [code](https://github.com/bhoov/flyvec)
- Related sparse-expansion work:
  - Litwin-Kumar+ 2017 Neuron: ~7 inputs per KC maximizes the dimension of the representation.
  - Bricken+ ICLR 2023: Sparse Distributed Memory is a continual learner.
  - Liu+ ICLR 2024: sparse random features for online Dyna world models.
  - Lan & Mahmood: Elephant activations.
- MB modules in RL agents (few, and mostly small-scale):
  - **Wei, Guo & Webb 2024 PLOS CB**: KC gap-junction value propagation. Solves Taxi-v3 in hundreds of episodes. [code](https://github.com/InsectRobotics/DynamicRoutingPublish)
  - **Lu & Webb 2026** (arXiv:2601.16806): MB + central complex for Habitat PointNav at very low compute.
  - **Staples 2026** (arXiv:2604.22081): insect-modular architecture on a toy foraging task. Single author, no code.
  - **FlyPrompt** (ICLR 2026): MB-inspired routing for continual learning. Supervised, not RL.
- **Open gap:** no peer-reviewed paper yet adds an MB module (FlyHash encoder, compartmentalized critics, or α'3 novelty bonus) to a standard deep RL agent on Atari, DMC or Procgen against strong baselines.

## 7. Design patterns to borrow for RL

1. **KC encoder**: fixed sparse random projection (5–10 inputs per unit, 10–40× expansion) followed by k-WTA at ~5%, placed on top of the obs encoder.
2. **Local gated value updates**: Δw = η·δ·KC_active, updating only the weights of active units. This gives cheap forgetting resistance.
3. **Compartmentalized critics**: several value heads with different learning and decay rates, split into appetitive and aversive channels, with slow heads bootstrapping the fast ones.
4. **MBON→DAN feedback as the TD baseline**: δ = r − V⁺ + V⁻, keeping the positive and negative streams separate.
5. **Timing-signed eligibility traces.**
6. **α'3-style novelty head**: its weights depress on each visit, which gives a similarity-aware, count-like intrinsic reward.
7. **Meta-learn the teacher, not the learner**: an outer loop learns the error-signal circuit, and the inner loop runs local three-factor plasticity.
8. **Keep the actor separate**: the MB acts as critic or memory, and policy learning runs through a separate pathway.

## 8. Candidate project directions

- **(a) MB-module deep RL (open gap, cheap):** add a KC encoder, compartmentalized critics and an α'3 novelty bonus to PPO or DQN. Run on continual or sparse-reward benchmarks (Procgen, MiniGrid, Continual World) with a proper ablation. Runs on a single GPU.
- **(b) Connectome-as-policy, done rigorously:** reproduce a FlyGM-style controller on flybody or FlyGym. Include degree-preserving and ER nulls with matched init (Dhiman's critique) and report whether topology actually helps. FlyGM's code isn't public, so this would be a from-scratch build. It needs a real GPU.
- **(c) Extract the MB circuit from FlyWire or MaleCNS** (KC→MBON, DAN→compartment, MBON→DAN) and use it as the wiring of a plastic critic inside an RL agent. This is a middle ground between (a) and (b).
- **(d) Closed-loop FlyGym + flyvis + an RL-trained descending interface**, replacing Eon's hand-mapped descending neurons with a learned mapping.
