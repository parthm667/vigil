# Visualization: the fly's brain and body, live

While the drone flies (or in simulation, or on replay), the laptop shows the fly's brain lighting up
and an anatomically detailed fly body animated from the brain's motor outputs. Concept: FLYGUIDE_SPEC
Section 6.8. Code: `flyfollow/viz/`.

What is on screen:

| Panel | What it shows | Source |
|---|---|---|
| Fly brain | 20k MaleCNS brain somata (faint cloud) and the 1,446-neuron pursuit circuit at real soma positions, glowing with live spikes. The LC10a -> AOTU025 / AOTU012 / AOTU019 -> DNa02 push-pull is drawn from the connectome's strongest synapses (amber = excitatory route, violet = the GABAergic AOTU019 route), with travelling pulses and live rates on the labels. | `brain_view.py` |
| Fly body | TuragaLab's flybody fly (MuJoCo), posed every frame from the brain's motor outputs. Label: "Animated from the fly brain's live motor outputs; not a physics simulation." | `fly_body.py` |
| Drone camera | The Tello frame with the tracked head box, or in simulation a synthetic scene with the filtered box. | `panels.py` |
| Fly's-eye view | LC10a spikes per azimuth bin (left eye row, right eye row), the camera bearing (green) and the LC10a activity centroid (white). | `panels.py` |
| Traces | DNa02 L / R rates and the yaw stick sent; in Rerun also DNa01, DNg13, forward stick and bearing. | `panels.py`, `live.py` |
| Top view | Simulation only: drone, camera wedge, person, trails and the standoff ring. | `panels.py` |

The demo story: a target on the right drives LC10a R; AOTU025/012 R excite DNa02 R while AOTU019 R
inhibits DNa02 L; DNa02 R fires, DNa02 L goes silent, the readout yaws the drone right, and the fly
body beats its left wing harder and banks right.

## Setup

```
scripts/setup_data.sh        # brains + MaleCNS annotations (needed for soma positions), if not done yet
scripts/setup_viz.sh         # flybody model -> data/flybody/ (about 140 MB, gitignored, pinned commit)
```

`setup_viz.sh` does a sparse, blob-filtered fetch of TuragaLab/flybody at commit `d015e9b` and copies only
`fruitfly.xml`, the 85 meshes it references and the Apache-2.0 `LICENSE` (plus `SOURCE.txt`) into
`data/flybody/`. Use `FLYBODY_SRC=/path/to/flybody` to copy from an existing clone, `FORCE=1` to refresh,
`FLYFOLLOW_FLYBODY=/dir` to install elsewhere (the viz reads the same variable). Never commit these files.

Python packages: the `viz` extra in `pyproject.toml` (mujoco, rerun-sdk, imageio, imageio-ffmpeg, pyzmq);
PIL, scipy, pandas and pyarrow come with the base install. No dm_control, no `mjpython`: the body is
rendered offscreen with `mujoco.Renderer` (the default macOS GL backend works; `MUJOCO_GL` does not need
to be set) and shown in Rerun or written to video.

The macOS hidden-flag issue on the venv's `.pth` files is handled by `scripts/setup_env.sh` (it installs a
`sitecustomize.py`), so no `PYTHONPATH` is needed.

## Run

```
# live: a simulated FOLLOW episode (PursuitEnv + FLY-HAND) streamed to the Rerun viewer in real time
.venv/bin/python -m flyfollow.viz.demo --brain data/brains/pursuit_core1.npz --seed 1000 --kind follow

# trained readout instead of hand calibration (a trainer best.json with "x" or "params"; its "arm" wins)
.venv/bin/python -m flyfollow.viz.demo --params runs/<run>/best.json --seed 1000

# record a slide-ready 1920x1080 30 fps MP4 composite (rendered after the episode, about 11 frames/s)
.venv/bin/python -m flyfollow.viz.demo --seed 1000 --seconds 30 --no-live --record runs/viz/demo.mp4 --stills runs/viz

# both at once: watch it live, then the MP4 is rendered from the same frames
.venv/bin/python -m flyfollow.viz.demo --seed 1000 --record runs/viz/demo.mp4

# save a Rerun recording instead of opening the viewer window (open later with: .venv/bin/rerun file.rrd)
.venv/bin/python -m flyfollow.viz.demo --seed 1000 --save-rrd runs/viz/demo.rrd

# no simulator: a scripted spot drives the subgraph LIF directly (sweep | left_right | hold)
.venv/bin/python -m flyfollow.viz.demo --synthetic --script left_right --record runs/viz/synthetic.mp4 --no-live

# component checks
.venv/bin/python -m flyfollow.viz.fly_body --out runs/viz_check/body_test.mp4   # scripted body clip + stills
.venv/bin/python -m flyfollow.viz.brain_view --out runs/viz_check              # brain stills, spot right / left
.venv/bin/python -m pytest tests/test_viz.py
```

Other options: `--kind approach`, `--profile train|demo|stress`, `--arm FLY-SHUF --brain data/brains/pursuit_core1_shuf1.npz`,
`--speed 2` (live playback speed), `--address tcp://127.0.0.1:5557`.

## Hooking the live drone runtime

The control loop publishes one frame per brain tick (20 Hz) through `VizSink`; the viewer is a separate
process. `publish()` never blocks: it pickles the frame and sends it on a ZeroMQ PUB socket bound to
loopback with a high-water mark of 4, so when the viewer is slow or absent the frame is dropped. Measured:
6 us mean per publish without an image, 0.5 ms with a 960x720 camera frame (downscaled to 480 px wide
inside the sink), p99 under 1 ms with a stalled subscriber.

```python
from flyfollow.viz.frames import frame_from_controller
from flyfollow.viz.live import VizSink

sink = VizSink()   # tcp://127.0.0.1:5557; loopback only (frames are pickled)
...
yaw, fb = controller.act(box, settings, DT)          # the FlyController (C's hooks are read, never modified)
go = governor.filter(yaw, fb, box, settings, DT, ...)
sink.publish(frame_from_controller(
    controller, t, tick=i, box=box, sticks=(go.yaw, go.fb),
    target={"bearing_deg": math.degrees(features.theta), "range_m": est_range},   # optional
    image=frame_rgb,                                                              # optional, RGB uint8
    meta={"source": "drone"}))
...
sink.close()       # sends an end marker; the viewer prints stats and exits (the Rerun window stays)
```

Then, in another terminal (or let a launcher call `flyfollow.viz.live.spawn_viewer(brain)`):

```
.venv/bin/python -m flyfollow.viz.live --brain data/brains/pursuit_core1.npz
```

The viewer drains every queued frame each loop (so the spike traces integrate all of them), logs the
brain and panels for the newest frame only, and renders the body on its own 30 fps clock. It needs only
the frame dict (`flyfollow/viz/frames.py` documents every key; all but `t` are optional). A controller
without spike counts still works: inputs and DNs are lit from rates.

Hooks used from the fly controller (all present in `FlyController`): `last_counts` (spikes per neuron per
tick), `last_channels` (22 encoder rates), `last_input_rates`, `last_dn_rates`, `last_sticks`, `brain_path`,
`encoder.centers` (bin azimuths), and read-only `decoder.wy / f_yaw / b_yaw` (the readout's yaw drive, see
the body mapping below). Missing hooks are skipped, not fatal.

## Biology versus engineering

| Biology (from the connectome and the LIF model) | Engineering (our choices) |
|---|---|
| Which neurons exist, their cell types and sides | Which 20k somata are drawn as context, colors, glow, point sizes |
| Soma positions (MaleCNS `somaLocation`, 8 nm voxels, shown in um) | Arc shapes of the pathway lines (drawn as curves between somata, not the real axon paths) |
| The pathway edges and their signs and synapse counts (strongest 14 LC10a inputs per relay neuron, all relay to DNa02 edges) | Which edges are drawn (strongest only) and the travelling-pulse animation |
| Spike counts per neuron per tick (Shiu et al. 2024 LIF on the 1,446-neuron core) | Decay of the glow (tau 0.2 s) and the rate readouts on labels |
| | The whole fly body animation (below), the camera-to-LC10a encoding, the readout to sticks, the governor |

The fly body is TuragaLab's flybody model of a **female** Drosophila melanogaster; our connectome is the
**male** CNS. The body is a kinematic puppet: joint angles are set directly each frame; there is no
physics, no aerodynamics and no trained flight controller.

## Body mapping (engineering)

| Pose channel | Driven by | Map |
|---|---|---|
| Left / right wing stroke amplitude | steering DN asymmetry and forward drive | `steer = tanh(readout yaw drive)`; `amp_L = 1 + 0.22 steer`, `amp_R = 1 - 0.22 steer` (times `1 + 0.08 drive`). A right turn gives a larger LEFT stroke, as in real flies. The difference is exaggerated for visibility. |
| Body bank | same `steer` | up to 22 deg, right wing down for a right turn |
| Body yaw | final yaw stick | up to 32 deg offset toward the turn (the camera does not follow) |
| Body pitch | forward stick | 42 deg nose-up hover posture, 12 deg lower at full forward drive |
| Head yaw | LC10a activity centroid (spike-weighted bin azimuths) | toward the target side, clipped to 18 deg |
| Wingbeat | fixed pattern (flybody's approximate base stroke) | shown at 3.2 Hz (25 % faster at full forward drive); the real fly beats at about 200 Hz, which video cannot show. Three translucent ghost wings trail the stroke as a motion sweep. |
| Legs | none | flybody's retracted flight posture (joint springrefs) |
| Smoothing | all channels | critically damped second-order filters (0.1 s wings, 0.12 s head, 0.25 to 0.45 s body) |

Why the readout yaw drive and not raw DNa02 R - L: core1 has a right bias (audit caveat 1; at rest DNa02 R
fires more than DNa02 L). The readout normalizes each side and type separately (hand calibration: DNa02_R
241 Hz, DNa02_L 123 Hz) and weighs five DN types, so "R above L in Hz" does not mean "turn right". In a
30 s FLY-HAND episode the sign of raw DNa02 R - L matched the yaw command only 62 % of the time; the readout
yaw drive (`sum_t w_t (R_t/norm_Rt - L_t/norm_Lt) + b_yaw`, filtered, the quantity that sets the yaw stick
before the governor) matched 100 %. So the wings use the readout drive, and the brain panel and traces
show the raw DNa02 rates. Frames without a readout (the synthetic stream) fall back to
`tanh((DNa02_R - DNa02_L) / 150 Hz)`.

## Performance (Apple M5, measured)

| Step | Cost |
|---|---|
| flybody model load (85 meshes, 818k vertices) | 0.85 s once |
| Body pose + render, 640x480 | 11 ms (render alone about 3 ms without shadows; the rest is text overlay) |
| Body pose + render, 960x720 | about 20 ms |
| Brain geometry load (feather join, 20k context somata, edges) | 0.5 to 0.9 s once |
| Brain numpy render with bloom, 960x720 (MP4 only; Rerun draws the 3D brain natively) | 32 ms |
| Composite 1920x1080 frame (body + brain + 4 panels) | about 89 ms on an idle machine (a 30 s clip in about 80 s); 179 ms while training jobs loaded the CPU |
| Viewer: log brain + panels per frame / body per frame | about 17 to 21 ms / 21 to 28 ms; live runs drew 152 of 160 and 116 of 120 frames |
| `VizSink.publish` | 6 us mean without image, 0.5 ms with a camera frame; in live demo runs 0.1 ms mean, 0.5 ms max; never blocks |

## Caveats

- LC10a azimuth bins are the rank fallback (`meta["azimuth_bins_method"] = "rank"`), not retinotopy: the
  fly's-eye cell positions are the encoder's bin assignment, so the LC10a centroid can differ from the
  camera bearing by several degrees.
- The pathway arcs connect somata; real LC10a to AOTU synapses sit in the anterior optic tubercle, not
  between cell bodies.
- In the Rerun 3D view the context cloud may render opaque (alpha on points depends on the viewer build).
- The Rerun path was verified by a live run (viewer spawned, 120 of 120 frames received, 4 not drawn) and by
  recording to `.rrd` (all entities and the blueprint present), but not by a screenshot of the viewer window,
  so the 3D eye position and panel proportions may want a tweak on the demo laptop (drag, then save the
  blueprint from the viewer).
- The Rerun viewer's own gRPC server listens on port 9876 (Rerun's default); our VizSink is loopback only.
