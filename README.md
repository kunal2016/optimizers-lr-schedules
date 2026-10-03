# Optimizers and LR Schedules

Experiments on Adam's internals, bias correction, warmup, cosine vs WSD schedules, and how the optimal learning rate changes with model width.

## Files

| File | What it is |
|---|---|
| `Optimizers_LR_Schedules_Run.ipynb` | The executed notebook, with all outputs and conclusions |
| `Optimizers_LR_Schedules.ipynb` | The same notebook without outputs, ready to rerun |
| `outputs/` | Every figure (PNG), table (CSV) and `results_summary.txt` from the run |

## How to run

1. Open `Optimizers_LR_Schedules.ipynb` in Google Colab.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Run all**, and approve the Google Drive pop-up. A full run takes about 15–25 minutes, and outputs are saved to `My Drive/TheSchoolofAI/optimizers-lr-schedules/outputs/`.

Set `QUICK = True` in the first cell for a quick test run (about 2 minutes). It saves to a separate `outputs_quick` folder, so it never overwrites the real results.

GPU kernels are not bit-exact, so a rerun can differ slightly from the numbers below.

## Setup

- **Parts 1–2:** a single scalar weight, Adam with lr 0.1, β = (0.9, 0.999), ε = 1e-8.
- **Parts 3–6:** a 2-layer character-level GPT trained on Tiny Shakespeare:
  - head size 64, sequence length 128, batch 32
  - AdamW with β = (0.9, 0.95), no weight decay, gradient clipping at 1.0
  - runs with the same seed start from the same weights and see the same batches
  - validation loss is measured on a fixed set of 20 batches

## Results

### 1. Adam by hand
I computed m, v, m̂, v̂, the step and the new weight in plain Python for one weight and five gradients, then ran `torch.optim.Adam` on the same inputs in float64. The largest difference is **3.5 × 10⁻¹⁷**, so the two agree to about 16 decimal places.

### 2. Bias correction off
For the same gradients, the uncorrected step is the corrected step times r(t) = (1 − β₁ᵗ) / √(1 − β₂ᵗ).

- **β₂ = 0.999:** over the first 20 steps the uncorrected steps are 3–6× too large, because v is underestimated far more than m. The difference stays within 1% only after **3,925 steps** (within 5% after 2,375).
- **β₂ = 0.95** (typical for LLMs): the uncorrected steps start too small, and the difference stops mattering after **76 steps**.

### 3. Update-to-weight ratio and warmup
I logged ‖ΔW‖ / ‖W‖ for every weight matrix at every step, in two runs from the same starting weights: one with a 100-step linear warmup and one with none.

Warmup ends at step 100, but the two runs' ratios only come within 10% of each other at about **step 177** (median over the 8 matrices that converge; range 109–287). The warmed-up ratios level off at about step 219. Three MLP matrices never converge within 300 steps, and their ratio stays 13–25% higher in the warmed-up run. So warmup changes the path the weights take, not just the first few updates.

### 4. Cosine vs WSD, scheduled for 300 steps, stopped at step 200
Each schedule had its own 5-point LR sweep. Both picked LR 4e-3, which is inside the sweep range.

| | Val loss @ 200 | LR @ 200 | + 30-step cooldown | Run to 300 |
|---|---|---|---|---|
| Cosine | 2.0244 | 1.5e-3 | — | 1.9322 |
| WSD | 2.0294 | 4.0e-3 | 1.9438 | **1.8740** |

At step 200 the two models are tied (see Part 6). **I would keep the WSD model**:
- It reaches the same loss while still at full learning rate; cosine has already cut its LR to about 37% of peak.
- A 30-step cooldown takes WSD to 1.944, within 0.012 of what cosine reaches after all 300 steps.
- When both run the full 300 steps, WSD finishes 0.058 lower.

### 5. LR sweep across widths

| Width | 256 | 512 | 1024 |
|---|---|---|---|
| Best LR | 5.76e-3 | 2.63e-3 | 1.38e-3 |

The best LR falls as **width^−1.03**, almost exactly the width^−1 that standard parameterization predicts: each doubling of width roughly halves it.

**At width 4096 I would use about 3e-4.** The fit gives 3.2e-4, and extrapolating from each pair of widths gives 2.5e-4 to 3.8e-4.

**Confidence:** moderate, within about 1.5× either way. The minima line up well and match theory, but the estimate goes two doublings past the widest model I measured, uses single-seed 300-step sweeps, and shares one LR across embeddings and hidden layers. Before a long run I would check 2e-4, 3e-4 and 4.5e-4. μP would make this extrapolation unnecessary.

### 6. Fair comparisons
- Each schedule and each width got its own LR sweep, and comparisons always used each one's best LR. An LR minimum at the edge of the sweep is flagged.
- All compared runs share the same initial weights and batches.
- The Part 4 comparison was repeated with 3 seeds:

| | Cosine | WSD | Gap | 2 × std. error | Verdict |
|---|---|---|---|---|---|
| Val loss @ 200 | 2.019 ± 0.014 | 2.009 ± 0.016 | 0.010 for WSD | 0.025 | tie |
| Run to 300 | 1.925 ± 0.011 | 1.864 ± 0.010 | 0.061 for WSD | 0.017 | **WSD better** |

At step 200 the winner depends on the seed (cosine on seed 0, WSD on seeds 1 and 2), so a single run would not have been enough to call it. At step 300 WSD wins on every seed by more than 3× the noise level.

## Outputs

| Type | Files |
|---|---|
| Figures | `p2_bias_correction.png`, `p3_update_ratio_per_layer.png`, `p3_warmup_vs_nowarmup_ratio.png`, `p4_cosine_vs_wsd.png`, `p5_lr_sweep_by_width.png`, `p5_lr_extrapolation.png` |
| Tables | `p1_adam_by_hand.csv`, `p1_adam_pytorch.csv`, `p2_bias_correction_steps.csv`, `p3_merge_step_per_layer.csv`, `p4_lr_sweep_val_at_stop.csv`, `p4_cosine_vs_wsd.csv`, `p5_lr_sweep_by_width.csv`, `p6_seed_repeats.csv`, `p6_seed_summary.csv` |
| Summary | `results_summary.txt` |
