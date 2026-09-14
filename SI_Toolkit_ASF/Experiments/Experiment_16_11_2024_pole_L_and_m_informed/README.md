# L4DC 2025 A-NC — pole length and pole mass

This folder is the **pole-length / pole-mass** Adaptive Neural Controller from
[A-NC (L4DC 2025, PMLR 283:987–998)](https://proceedings.mlr.press/v283/paluch25a.html).
It is **not** the angle-sensor-offset experiment (that is a different dataset
and a different net).

Privileged parameters the GRU must infer from history:

- **L** — full pole length, one draw per 360 s file in **[0.05, 0.8] m**
- **m_pole** — resampled every **2.6 s** in **[0.015, 0.15] kg**

The net never sees L or m as inputs.

## Give this to a customer

Branch: `L4DC25-adaptive-neural-control-recreated`  
Data: CartPoleSimulation Release [`l4dc25-pole-L-and-m-data`](https://github.com/SensorsINI/CartPoleSimulation/releases/tag/l4dc25-pole-L-and-m-data)

```bash
git clone --recurse-submodules -b L4DC25-adaptive-neural-control-recreated \
  https://github.com/SensorsINI/physical-cartpole.git
cd physical-cartpole
git submodule update --init --recursive
# Recordings (~301 MB). Needs zstd (e.g. brew install zstd).
./Driver/CartPoleSimulation/others/EulerClusterScripts/fetch_l4dc_pole_L_and_m_recordings.sh
```

Standalone simulator (no robot repo):

```bash
git clone --recurse-submodules -b L4DC25-adaptive-neural-control-recreated \
  https://github.com/SensorsINI/CartPoleSimulation.git
cd CartPoleSimulation
git submodule update --init --recursive
./others/EulerClusterScripts/fetch_l4dc_pole_L_and_m_recordings.sh
```

Do **not** run `git submodule update --remote` — that would move Control_Toolkit
off the Driver-cwd optimizer fix.

## What is in git vs the Release artifact

| Path | Where |
|---|---|
| `Models/GRU-8IN-64H1-64H2-1OUT-2/` | **This repo** — paper A-NC weights + training yaml |
| `NormalizationInfo/NI_2024-11-16_20-04-12.csv` | **This repo** |
| `Recordings/Train` (172 files, 3.096 M rows) | **GitHub Release** `l4dc25-pole-L-and-m-data` (~301 MB `tar.zst`) |
| `Recordings/Validate` (10 files, 180 k) | same artifact |
| `Recordings/Test` (10 files, 180 k) | same artifact |

Paper text said “5 M / ~28 h” train. The archived set is **3.10 M / 17.2 h**.
Validation **180 k** matches. Treat the files, not the prose, as ground truth.

SHA256 of `Experiment_16_11_2024_pole_L_and_m_informed_recordings.tar.zst`:

`c6613c058a0fbe6d7f111d691113861f2fa7f5813ba9dc79b6d79b8e20ea5787`

Also in [CHECKSUMS.sha256](CHECKSUMS.sha256).

```bash
./others/EulerClusterScripts/fetch_l4dc_pole_L_and_m_recordings.sh
# or, if you already have the archive:
./others/EulerClusterScripts/fetch_l4dc_pole_L_and_m_recordings.sh /path/to/the.tar.zst
```

Inputs: `Q_applied_-1`, `angleD`, `angle_cos`, `angle_sin`, `position`,
`positionD`, `target_equilibrium`, `target_position`.  
Label: `Q_calculated_informed`.

Teacher for a new collect/relabel on this branch is **`rpgd_c_l4dc`**
(16 rollouts, 64 outer iterations, warmup on). The C cost is not bit-identical
to 2024 `rpgd-tf` + Python `quadratic_boundary_grad`; weights are mapped in
`optimizer_rpgd_c_l4dc.py`.

## Retry the published net

Default `cartpole_physical_parameters.yml` is the **clean test plant**
(paper Fig. 3: `L=0.15`, `m_pole=0.050`, control noise OFF, informer OFF).
The collection recipe is `cartpole_physical_parameters_l4dc_collect.yml`
(noise 0.5, random L / m_pole, informer 4.5 s ON / 1.5 s OFF).

From the physical-cartpole repo, Python 3.9 + TensorFlow (`CPS39`):

```bash
python Driver/CartPoleSimulation/run_cartpole_gui.py
```

`neural-imitator` already points at `GRU-8IN-64H1-64H2-1OUT-2`
(Driver-relative `PATH_TO_MODELS`). Keep `second_derivatives_mode: ODE`.
Leave the informer OFF so the GRU has to infer L and m. To compare RPGD
on this same plant, set informer `ON` so the teacher sees the true parameters.

## Retrain

Unpack the Recordings artifact first, then:

```bash
python SI_Toolkit_ASF/Run/A1_Create_Normalization_File.py
python SI_Toolkit_ASF/Run/A2_Train_Network.py
```

`SI_Toolkit_ASF/config_training.yml` already points here (`Q_calculated_informed`,
wash 15 + 10, batch 256, seed 1873, no angle-shift aug).

## Recollect (cluster)

Copy the collection plant over the default yaml first:

```bash
cp cartpole_physical_parameters_l4dc_collect.yml cartpole_physical_parameters.yml
```

1. `python run_data_generator.py -i $i` for `i = 1 … 192`
2. `python SI_Toolkit_ASF/Run/PreprocessData_Add_Control_Along_Trajectories.py -i $i`  
   (uses `rpgd_c_l4dc`, writes `Q_calculated_informed` from true `L` and `m_pole`)
3. Split: Test `001–005`+`183–187`, Val `006–010`+`188–192`, Train `011–182`

RPGD seed is null — labels will not bit-match.

## Publish / fetch the dataset artifact

Pack (from CartPoleSimulation root):

```bash
./others/EulerClusterScripts/pack_l4dc_pole_L_and_m_artifact.sh
```

After this branch is on GitHub (`gh auth login` if needed):

```bash
./others/EulerClusterScripts/publish_l4dc_pole_L_and_m_release.sh
```
