# Training Objective for Supervised Manifold-Based ECoG Movement Decoder

## Goal

Train the decoder so that it:

1. predicts the correct movement class from labeled ECoG data,
2. learns a latent representation that is compact within class and separated across classes,
3. enforces temporal coherence for nearby samples belonging to the same movement,
4. makes the **low-dimensional manifold itself** discriminative, not only the higher-dimensional hidden embedding.

This objective is supervised. Reinforcement learning, if retained, should only be used later for online adaptation or calibration, not as the main training signal.

---

## Current Model Assumption

The current pipeline appears to be:

- input ECoG features or time window -> encoder / trunk
- penultimate embedding `z` (currently the main decoding space, e.g. 32D)
- projection head `m` (candidate low-dimensional manifold, e.g. 2D or 3D)
- classifier on `z`
- optional classifier on `m`

Important implication:

- if the classifier only reads from `z`, then `m` may become visually smooth but not necessarily useful for decoding,
- therefore `m` must be explicitly supervised.

---

## Notation

Let:

- `x_t` = ECoG input at time `t`
- `y_t` = ground-truth movement label at time `t`
- `z_t` = latent embedding produced by the model
- `m_t` = low-dimensional manifold projection produced by the model
- `p_t = softmax(logits_z_t)` = class probabilities from `z_t`
- `p_m_t = softmax(logits_m_t)` = optional class probabilities from `m_t`

Batch size: `N`

Number of classes: `K`

---

## Total Loss

The full training loss should be:

```text
L_total =
    lambda_cls       * L_cls
  + lambda_compact   * L_compact
  + lambda_sep       * L_sep
  + lambda_temp      * L_temp
  + lambda_proj_cls  * L_proj_cls
  + lambda_proj_comp * L_proj_compact
  + lambda_proj_sep  * L_proj_sep
  + lambda_proj_temp * L_proj_temp
```

Recommended philosophy:

- classification loss should remain the strongest term,
- manifold compactness and separation should be meaningful but not dominant,
- temporal regularization should be weaker than classification,
- projection supervision must be present if the low-dimensional manifold is part of the scientific objective.

---

## 1. Classification Loss on Latent Embedding

This is the main supervised objective.

### Definition

```text
L_cls = CrossEntropy(logits_z, y)
```

If classes are imbalanced, replace with one of:

### Weighted Cross Entropy

```text
L_cls = WeightedCrossEntropy(logits_z, y, class_weights)
```

### Focal Loss

```text
L_cls = FocalLoss(logits_z, y, gamma)
```

### When to use focal loss

Use focal loss if:

- some classes are systematically harder than others,
- the model becomes overconfident on easy classes,
- confusion is concentrated in a few class pairs.

---

## 2. Compactness Loss in Latent Space

This forces samples of the same class to cluster together in `z`.

### Definition

For each class `k`, define a centroid `c_k` in latent space.

```text
L_compact = (1 / N) * sum_t || z_t - c_{y_t} ||^2
```

### Practical notes

Possible implementations:

- maintain class centers as moving averages,
- recompute centers from each batch,
- or use a center-loss style module.

### Purpose

This reduces intra-class spread and makes the latent representation more stable.

---

## 3. Separation Loss in Latent Space

This forces different classes to remain separated.

### Definition

Let `c_i` and `c_j` be class centroids in latent space. For a desired margin `delta`:

```text
L_sep = sum_{i != j} max(0, delta - ||c_i - c_j||)^2
```

### Alternative implementation

Instead of centroid separation, you can use:

- supervised contrastive loss,
- triplet loss,
- pairwise margin loss.

### Recommendation

If two or more movement classes are frequently confused, supervised contrastive loss is usually better than a simple centroid-margin loss.

---

## 4. Temporal Consistency Loss in Latent Space

The latent representation should evolve smoothly over time **within the same class**, but should not over-smooth real transitions.

### Definition

Only penalize temporal jumps when adjacent samples share the same label:

```text
L_temp = (1 / (N - 1)) * sum_{t=2..N} 1[y_t == y_{t-1}] * ||z_t - z_{t-1}||^2
```

### Important

Do not apply global smoothness blindly across all timesteps, otherwise class transitions will be blurred.

### Purpose

This encourages stable trajectories for the same movement while preserving sharp class boundaries.

---

## 5. Projection Classification Loss

If the scientific goal is to identify a useful low-dimensional manifold, the projection `m` must also be directly predictive of the label.

### Definition

Add a small classifier on top of `m` and optimize:

```text
L_proj_cls = CrossEntropy(logits_m, y)
```

### Why this is necessary

Without this term:

- `z` can remain fully discriminative,
- `m` can remain only visually smooth,
- the low-dimensional manifold may fail to carry the class information needed for decoding.

This is one of the most important additions if the project wants the manifold itself to be meaningful.

---

## 6. Compactness Loss on the Projection

The low-dimensional projection should also form class-specific clusters.

### Definition

For each class `k`, define projection centroid `mu_k`.

```text
L_proj_compact = (1 / N) * sum_t || m_t - mu_{y_t} ||^2
```

### Purpose

This encourages compact class structure directly in low dimension.

---

## 7. Separation Loss on the Projection

Different classes should also be separated in the projected space.

### Definition

```text
L_proj_sep = sum_{i != j} max(0, delta_m - ||mu_i - mu_j||)^2
```

### Purpose

This prevents the projection from collapsing different movements into overlapping regions.

---

## 8. Temporal Consistency Loss on the Projection

Temporal smoothness can also be applied directly on the manifold.

### Definition

```text
L_proj_temp = (1 / (N - 1)) * sum_{t=2..N} 1[y_t == y_{t-1}] * ||m_t - m_{t-1}||^2
```

### Purpose

This helps produce stable low-dimensional trajectories, which is useful both scientifically and visually.

---

## 9. Recommended Default Weighting

A practical starting point:

```text
lambda_cls       = 1.0
lambda_compact   = 0.1
lambda_sep       = 0.1
lambda_temp      = 0.05
lambda_proj_cls  = 0.5
lambda_proj_comp = 0.05
lambda_proj_sep  = 0.05
lambda_proj_temp = 0.02
```

### Notes

- `lambda_cls` should usually remain the strongest term.
- `lambda_proj_cls` should be substantial if you want `m` to become a real decoding manifold.
- `lambda_temp` and `lambda_proj_temp` should stay relatively small to avoid over-smoothing.
- if the projection is still not discriminative, increase `lambda_proj_cls` first before increasing geometric penalties.

---

## 10. Recommended Metrics to Track During Training

Track these metrics on both training and validation sets.

### Decoding metrics

- balanced accuracy
- macro F1
- per-class recall
- per-class precision
- confusion matrix
- top-2 accuracy
- negative log-likelihood
- expected calibration error (ECE)
- Brier score

### Latent-space geometry metrics on `z`

- within-class variance
- between-class variance
- Fisher ratio
- silhouette score
- centroid distances

### Projection geometry metrics on `m`

- within-class variance
- between-class variance
- Fisher ratio
- silhouette score
- centroid distances
- neighborhood preservation / trustworthiness if needed

### Temporal metrics

- within-class temporal smoothness
- transition sharpness
- class-switch lag
- mean latent velocity
- mean projection velocity

---

## 11. Handling Hard-to-Discriminate Classes

If some classes remain difficult to separate, apply the following changes.

### A. Use focal loss or class-weighted CE

This helps when difficult classes are underrepresented or systematically confused.

### B. Hard negative mining

When sampling batches for contrastive or triplet loss, deliberately include examples from frequently confused classes.

### C. Pairwise confusion-aware weighting

If classes `i` and `j` are often confused, increase the weight of those mistakes.

Example idea:

```text
L_cls_weighted = pair_weight[y_true, y_pred] * CrossEntropy(...)
```

### D. Hierarchical classification

If appropriate, first classify a coarse family of movements, then a more specific subclass.

### E. Temporal disambiguation

If single-frame class separation is poor, classify from a short temporal window rather than from one isolated timestep.

---

## 12. Implementation Priorities

If code changes must be staged, implement in this order:

### Priority 1
Add direct projection supervision:

```text
L_proj_cls
```

### Priority 2
Add latent compactness:

```text
L_compact
```

### Priority 3
Add latent separation:

```text
L_sep
```

### Priority 4
Add class-aware temporal consistency:

```text
L_temp
```

### Priority 5
Mirror compactness / separation / temporal terms on the projection:

```text
L_proj_compact
L_proj_sep
L_proj_temp
```

This order gives the most immediate gain toward making the manifold truly useful.

---

## 13. Final Objective Summary

The model should not only classify correctly from a hidden embedding. It should also learn a low-dimensional manifold that is:

- discriminative,
- compact within class,
- separated across class,
- temporally coherent.

The recommended training objective is therefore:

```text
L_total =
    lambda_cls       * L_cls
  + lambda_compact   * L_compact
  + lambda_sep       * L_sep
  + lambda_temp      * L_temp
  + lambda_proj_cls  * L_proj_cls
  + lambda_proj_comp * L_proj_compact
  + lambda_proj_sep  * L_proj_sep
  + lambda_proj_temp * L_proj_temp
```

This formulation is supervised, stable, and directly aligned with the scientific objective of identifying an interpretable low-dimensional neural manifold for movement decoding.

---

## 14. Optional Next Step for Copilot

Ask Copilot to:

1. add the new loss terms in the loss module,
2. add a classifier head on the projection if missing,
3. compute class centers / projection centers per batch,
4. log manifold metrics for both `z` and `m`,
5. expose the lambda weights in the config file,
6. keep RL-related code isolated from the main supervised training path unless explicitly enabled.
