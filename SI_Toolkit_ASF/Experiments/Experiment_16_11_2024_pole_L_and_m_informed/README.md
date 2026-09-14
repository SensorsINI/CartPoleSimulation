# L4DC 2025 A-NC — pole length and pole mass

This folder is the **pole-length / pole-mass** Adaptive Neural Controller from
[A-NC (L4DC 2025, PMLR 283:987–998)](https://proceedings.mlr.press/v283/paluch25a.html).
It is **not** the angle-sensor-offset experiment (that is a different dataset
and a different net).

Privileged parameters the GRU must infer from history:

- **L** — full pole length, one draw per 360 s file in **[0.05, 0.8] m**
- **m_pole** — resampled every **2.6 s** in **[0.015, 0.15] kg**

The net never sees L or m as inputs.

## What is in git vs the Release artifact

| Path | Where |
|---|---|
| `Models/GRU-8IN-64H1-64H2-1OUT-2/` | **This repo** — paper A-NC weights + training yaml |
| `NormalizationInfo/NI_2024-11-16_20-04-12.csv` | **This repo** |
| `Recordings/Train` (172 files, 3.096 M rows) | **GitHub Release artifact** (~835 MB) |
| `Recordings/Validate` (10 files, 180 k) | same artifact |
| `Recordings/Test` (10 files, 180 k) | same artifact |

Paper text said “5 M / ~28 h” train. The archived set is **3.10 M / 17.2 h**.
Validation **180 k** matches. Treat the files, not the prose, as ground truth.

After you attach the artifact (see below):

```bash
./others/EulerClusterScripts/pack_l4dc_pole_L_and_m_artifact.sh --extract \
  /path/to/Experiment_16_11_2024_pole_L_and_m_informed_recordings.tar.zst
```

Inputs: `Q_applied_-1`, `angleD`, `angle_cos`, `angle_sin`, `position`,
`positionD`, `target_equilibrium`, `target_position`.  
Label: `Q_calculated_informed`.

Teacher for a new collect/relabel on this branch is **`rpgd_c_l4dc`**
(16 rollouts, 64 outer iterations, warmup on). The C cost is not bit-identical
to 2024 `rpgd-tf` + Python `quadratic_boundary_grad`; weights are mapped in
`optimizer_rpgd_c_l4dc.py`.

## Retry the published net

From this repository root, Python 3.9 + TensorFlow (`CPS39`):

```bash
# cartpole_physical_parameters.yml: informer OFF, L and m_pole constant
# Example paper Fig. 3: L: 0.15, m_pole: 0.050
python run_cartpole_gui.py
```

`neural-imitator` already points at `GRU-8IN-64H1-64H2-1OUT-2`.
Keep `second_derivatives_mode: ODE`. Do not turn the informer ON.

## Retrain

Unpack the Recordings artifact first, then:

```bash
python SI_Toolkit_ASF/Run/A1_Create_Normalization_File.py
python SI_Toolkit_ASF/Run/A2_Train_Network.py
```

`SI_Toolkit_ASF/config_training.yml` already points here (`Q_calculated_informed`,
wash 15 + 10, batch 256, seed 1873, no angle-shift aug).

## Recollect (cluster)

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

Attach to a GitHub Release on this branch (after the branch is pushed):

```bash
gh release create l4dc25-pole-L-and-m-data \
  --title "L4DC 2025 A-NC pole length and pole mass — recordings" \
  --notes "Train/Val/Test CSVs for Experiment_16_11_2024_pole_L_and_m_informed (not angle offset)." \
  Experiment_16_11_2024_pole_L_and_m_informed_recordings.tar.zst
```
