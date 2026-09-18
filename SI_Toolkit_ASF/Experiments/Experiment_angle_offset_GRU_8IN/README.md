# Angle-offset A-NC — GRU-8IN-64-0 (April 2025)

This folder is a **vertical-angle (0-angle) offset** Adaptive Neural Controller.
It is **not** the L4DC 2025 paper experiment (that one infers pole length and
pole mass: `Experiment_16_11_2024_pole_L_and_m_informed`).

Privileged parameter the net must infer from history:

- **vertical_angle_offset** — sensor-zero bias on the measured angle only
  (not `angleD`). Training draws one constant offset per wash-out window in
  **[−π, π]**. Feature noise is **0.1**.

The net never sees the offset as an input.

## Selected net

`GRU-8IN-64H1-64H2-1OUT-0` (Keras GRU, trained 6 April 2025, CPS git
`5b47d272`). SHA256 of the `.keras` file:

`858c73c72354ad21025d722b3a0c5feaab70d22cf10ca0264edf1b67a71d484a`

| | |
|---|---|
| Inputs | `Q_applied_-1`, `angleD`, `angle_cos`, `angle_sin`, `position`, `positionD`, `target_equilibrium`, `target_position` |
| Label | `Q_calculated_offline_slow_swing` |
| Wash / post-wash | 100 + 20 (label shift 1) |
| Best val | **0.02780** at epoch 52 (seed 1873, batch 1024, LR 2e-3) |
| Feature noise | **0.1** |
| Series mode | `train_for_random_vertical_angle_shift` |

Closed-loop on a long pole (`L=0.395` m, `m_pole=0.087` kg, dt 0.002 / 0.01,
**control noise OFF**, informer OFF), hang start, 12 s:

- Swings up and **holds at 0°, 90°, and 180°** sensor offset.

This is a Keras `GRU`, not `GRU_mask0` / CustomGRU.

## What is shipped vs what we cannot promise

The original training path was
`SI_Toolkit_ASF/Experiments/Experiment_3_04_2025/Recordings/Train`.
That `Train/` folder is gone. The CSVs here are the surviving sim pool from
`Other/Experiment_3_04_2025` (`Train_all` 172 + `Validate` 9 + `Test` 11).
They have the swing teacher and match the NI used next to these weights.
A **different** keras sitting in that old folder is **not** this net, so we
cannot claim a bit-identical retrain.

Collection was **simulation only** (360 s, 100 Hz, `rpgd-tf`). RPGD seeds were
unset. Recollecting will not reproduce these files.

## Give this to a customer

Branch: `angle-adaptive-gru-recreated`  
Data: CartPoleSimulation Release
[`angle-offset-gru-8in-data`](https://github.com/SensorsINI/CartPoleSimulation/releases/tag/angle-offset-gru-8in-data)

```bash
git clone --recurse-submodules -b angle-adaptive-gru-recreated \
  https://github.com/SensorsINI/physical-cartpole.git
cd physical-cartpole
git submodule update --init --recursive
./Driver/CartPoleSimulation/others/EulerClusterScripts/fetch_angle_offset_recordings.sh
```

Standalone simulator:

```bash
git clone --recurse-submodules -b angle-adaptive-gru-recreated \
  https://github.com/SensorsINI/CartPoleSimulation.git
cd CartPoleSimulation
git submodule update --init --recursive
./others/EulerClusterScripts/fetch_angle_offset_recordings.sh
```

Do **not** run `git submodule update --remote`.

For the **paper** (pole length and pole mass), use
`L4DC25-adaptive-neural-control-recreated` and Release `l4dc25-pole-L-and-m-data`
instead.

## What is in git vs the Release artifact

| Path | Where |
|---|---|
| `Models/GRU-8IN-64H1-64H2-1OUT-0/` | **This repo** |
| `NormalizationInfo/NI_2025-04-04_06-57-28.csv` | **This repo** |
| `Recordings/Train` (172 files) | **Release** `angle-offset-gru-8in-data` |
| `Recordings/Validate` (9 files) | same |
| `Recordings/Test` (11 files) | same |

```bash
./others/EulerClusterScripts/fetch_angle_offset_recordings.sh
```

## Retrain (from the archived CSVs)

`SI_Toolkit_ASF/config_training.yml` on this branch already points here
(`Q_calculated_offline_slow_swing`, wash 100 + 20, batch 1024, seed 1873,
angle-shift, feature noise 0.1).

```bash
# from CartPoleSimulation root, after fetch_angle_offset_recordings.sh
python SI_Toolkit_ASF/Run/A2_Train_Network.py
```

New weights will not match the published keras.

## Recollect (optional, not bit-identical)

Copy the collection plant over the default yaml first:

```bash
cp cartpole_physical_parameters_angle_offset_collect.yml cartpole_physical_parameters.yml
```

The archived files were recorded with (see CSV headers):

- long pole `L=0.395` m, `m_pole=0.087` kg
- additive control noise **0.5**
- 0-angle `random` in ±π every **3 s**
- informer `switching_regular` (1.5 s off / 4 s on)
- 360 s, dt 0.001 / 0.01 / 0.01, controller `mpc` / `rpgd-tf`

Then relabel with
`SI_Toolkit_ASF/Run/PreprocessData_Add_Control_Along_Trajectories.py`
(`Q_calculated_offline_slow_swing`). The script defaults are placeholders;
point `-i` at each CSV. RPGD seed is null.

## Test the published net

Default `cartpole_physical_parameters.yml` on commit 2 of this branch is the
**clean test plant** (long pole, control noise OFF, informer OFF, 0-angle
redraw every 4 s). Collection recipe stays in
`cartpole_physical_parameters_angle_offset_collect.yml`.

GUI default controller is `neural-imitator` → this GRU. Start from hang to
see swing-up; start upright to watch 0-angle jumps.
