from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import ModelConfig
from .losses import (
    centroid_separation_loss,
    class_conditional_temporal_loss,
    compactness_loss,
    entropy_regularization,
    geometry_statistics,
    reward_weighted_policy_loss,
    supervised_classification_loss,
    temporal_smoothness_loss,
)
from .reward import RewardProvider
from .stream import ExperienceReplayBuffer
from .types import Experience, InferenceStep, StreamSample, TrainingMetrics


class OnlineTrainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: ModelConfig,
        reward_provider: RewardProvider,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.reward_provider = reward_provider
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.buffer = ExperienceReplayBuffer(cfg.buffer_size)
        self.step_count = 0
        self.num_updates = 0
        self.labeled_seen = 0

        self.reward_baseline = 0.0
        self._baseline_initialized = False

    def process_sample(self, sample: StreamSample) -> InferenceStep:
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(sample.embedding).to(self.device, dtype=torch.float32).unsqueeze(0)
            out = self.model(x)

        probs = out.probs.squeeze(0).cpu().numpy()
        adjust_fn = getattr(self.reward_provider, "adjust_probabilities", None)
        if callable(adjust_fn):
            adjusted = np.asarray(adjust_fn(sample, probs), dtype=float)
            if adjusted.shape == probs.shape and np.all(np.isfinite(adjusted)):
                mass = float(np.sum(adjusted))
                if mass > 0.0:
                    probs = adjusted / mass

        pred = int(np.argmax(probs))
        confidence = float(probs[pred])
        reward = self.reward_provider.compute(sample, probs)
        game_feedback = getattr(self.reward_provider, "last_feedback", None)

        self._update_reward_baseline(reward)

        label_value = int(sample.label) if sample.label is not None else -1
        if label_value >= 0:
            self.labeled_seen += 1

        self.buffer.append(
            Experience(
                sample_idx=sample.sample_idx,
                timestamp=sample.timestamp,
                embedding=sample.embedding.copy(),
                label=label_value,
                action=pred,
                reward=reward,
            )
        )

        self.step_count += 1
        train_metrics: TrainingMetrics | None = None
        if self._should_update():
            train_metrics = self._update_model()

        return InferenceStep(
            sample_idx=sample.sample_idx,
            label=sample.label,
            predicted_class=pred,
            confidence=confidence,
            reward=reward,
            probabilities=probs.copy(),
            penultimate=out.penultimate.squeeze(0).cpu().numpy().copy(),
            projection=out.projection.squeeze(0).cpu().numpy().copy(),
            training=train_metrics,
            game_prompt_id=getattr(game_feedback, "prompt_id", None),
            game_target_class=getattr(game_feedback, "target_class", None),
            game_in_window=bool(getattr(game_feedback, "in_window", False)),
            game_hit=bool(getattr(game_feedback, "hit", False)),
            game_label_correct=bool(getattr(game_feedback, "label_correct", False)),
            game_timing_hit=bool(getattr(game_feedback, "timing_hit", False)),
            game_timing_error_s=getattr(game_feedback, "timing_error_s", None),
            game_seconds_to_window_start=getattr(game_feedback, "seconds_to_window_start", None),
            game_next_target_class=getattr(game_feedback, "next_target_class", None),
            game_seconds_to_next_prompt_start=getattr(
                game_feedback,
                "seconds_to_next_prompt_start",
                None,
            ),
            game_prompt_progress=getattr(game_feedback, "prompt_progress", None),
            game_margin=getattr(game_feedback, "margin", None),
            game_level=getattr(game_feedback, "level", None),
            game_streak=getattr(game_feedback, "streak", None),
            game_reward_components=(
                dict(getattr(game_feedback, "components", {}))
                if game_feedback is not None
                else None
            ),
        )

    def _should_update(self) -> bool:
        if self.step_count % self.cfg.update_every != 0:
            return False
        if len(self.buffer) < self.cfg.min_buffer_before_updates:
            return False
        return True

    def _update_model(self) -> TrainingMetrics:
        batch = self.buffer.sample_recent(self.cfg.batch_size)
        if not batch:
            return _empty_metrics(rl_enabled=False)

        x_np = np.stack([exp.embedding for exp in batch], axis=0)
        y_np = np.array([exp.label for exp in batch], dtype=np.int64)
        a_np = np.array([exp.action for exp in batch], dtype=np.int64)
        r_np = np.array([exp.reward for exp in batch], dtype=np.float32)

        x = torch.from_numpy(x_np).to(self.device, dtype=torch.float32)
        y = torch.from_numpy(y_np).to(self.device)
        actions = torch.from_numpy(a_np).to(self.device)
        rewards = torch.from_numpy(r_np).to(self.device)

        self.model.train()
        out = self.model(x)

        class_weights = self.cfg.class_weight_tensor(self.device)
        supervised = supervised_classification_loss(
            out.logits,
            y,
            class_weights=class_weights,
            focal_gamma=self.cfg.classification_focal_gamma,
        )
        projection_supervised = supervised_classification_loss(
            out.projection_logits,
            y,
            class_weights=class_weights,
            focal_gamma=self.cfg.classification_focal_gamma,
        )

        compact = compactness_loss(out.penultimate, y, n_classes=self.cfg.n_classes)
        sep = centroid_separation_loss(
            out.penultimate,
            y,
            margin=self.cfg.latent_sep_margin,
            n_classes=self.cfg.n_classes,
        )
        temporal = class_conditional_temporal_loss(out.penultimate, y)

        proj_compact = compactness_loss(out.projection, y, n_classes=self.cfg.n_classes)
        proj_sep = centroid_separation_loss(
            out.projection,
            y,
            margin=self.cfg.projection_sep_margin,
            n_classes=self.cfg.n_classes,
        )
        proj_temporal = class_conditional_temporal_loss(out.projection, y)

        manifold_supervised = (
            self.cfg.lambda_cls * supervised
            + self.cfg.lambda_compact * compact
            + self.cfg.lambda_sep * sep
            + self.cfg.lambda_temp * temporal
            + self.cfg.lambda_proj_cls * projection_supervised
            + self.cfg.lambda_proj_compact * proj_compact
            + self.cfg.lambda_proj_sep * proj_sep
            + self.cfg.lambda_proj_temp * proj_temporal
        )

        labeled_in_batch = int((y >= 0).sum().item())

        rl_enabled = self.labeled_seen >= self.cfg.warmup_labeled_samples
        if rl_enabled:
            advantages = rewards - float(self.reward_baseline)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
            policy = reward_weighted_policy_loss(out.logits, actions, advantages)
        else:
            policy = out.logits.sum() * 0.0

        entropy = entropy_regularization(out.logits)
        smoothness = temporal_smoothness_loss(out.projection)
        auxiliary_loss = out.logits.sum() * 0.0
        aux_loss_fn = getattr(self.model, "auxiliary_loss", None)
        if callable(aux_loss_fn):
            auxiliary_loss = aux_loss_fn(out, y, self.cfg)

        if labeled_in_batch == 0 and not rl_enabled:
            return _empty_metrics(rl_enabled=False)

        total = (
            self.cfg.supervised_weight * manifold_supervised
            + self.cfg.policy_weight * policy
            + self.cfg.contrastive_weight * auxiliary_loss
            - self.cfg.entropy_weight * entropy
            + self.cfg.smoothness_weight * smoothness
        )

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        self.num_updates += 1

        with torch.no_grad():
            within_z, between_z, fisher_z = geometry_statistics(
                out.penultimate.detach(), y, n_classes=self.cfg.n_classes
            )
            within_m, between_m, fisher_m = geometry_statistics(
                out.projection.detach(), y, n_classes=self.cfg.n_classes
            )
            bal_acc, macro_f1, top2, nll, brier, ece = _decoding_metrics(
                out.logits.detach(),
                y,
                self.cfg.n_classes,
            )

        return TrainingMetrics(
            update_applied=True,
            total_loss=float(total.detach().cpu().item()),
            supervised_loss=float(manifold_supervised.detach().cpu().item()),
            policy_loss=float(policy.detach().cpu().item()),
            auxiliary_loss=float(auxiliary_loss.detach().cpu().item()),
            entropy=float(entropy.detach().cpu().item()),
            smoothness_loss=float(smoothness.detach().cpu().item()),
            labeled_in_batch=labeled_in_batch,
            rl_enabled=rl_enabled,
            manifold_supervised_loss=float(manifold_supervised.detach().cpu().item()),
            projection_supervised_loss=float(projection_supervised.detach().cpu().item()),
            compactness_loss=float(compact.detach().cpu().item()),
            separation_loss=float(sep.detach().cpu().item()),
            temporal_consistency_loss=float(temporal.detach().cpu().item()),
            projection_compactness_loss=float(proj_compact.detach().cpu().item()),
            projection_separation_loss=float(proj_sep.detach().cpu().item()),
            projection_temporal_loss=float(proj_temporal.detach().cpu().item()),
            within_class_var_z=float(within_z.detach().cpu().item()),
            between_class_var_z=float(between_z.detach().cpu().item()),
            fisher_ratio_z=float(fisher_z.detach().cpu().item()),
            within_class_var_m=float(within_m.detach().cpu().item()),
            between_class_var_m=float(between_m.detach().cpu().item()),
            fisher_ratio_m=float(fisher_m.detach().cpu().item()),
            balanced_accuracy=bal_acc,
            macro_f1=macro_f1,
            top2_accuracy=top2,
            negative_log_likelihood=nll,
            brier_score=brier,
            expected_calibration_error=ece,
        )

    def _update_reward_baseline(self, reward: float) -> None:
        alpha = self.cfg.reward_baseline_alpha
        if not self._baseline_initialized:
            self.reward_baseline = reward
            self._baseline_initialized = True
            return
        self.reward_baseline = (1.0 - alpha) * self.reward_baseline + alpha * reward


def _empty_metrics(rl_enabled: bool) -> TrainingMetrics:
    return TrainingMetrics(
        update_applied=False,
        total_loss=0.0,
        supervised_loss=0.0,
        policy_loss=0.0,
        auxiliary_loss=0.0,
        entropy=0.0,
        smoothness_loss=0.0,
        labeled_in_batch=0,
        rl_enabled=rl_enabled,
    )


def _decoding_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    ece_bins: int = 10,
) -> tuple[float, float, float, float, float, float]:
    mask = labels >= 0
    if not torch.any(mask):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    logits_l = logits[mask]
    labels_l = labels[mask]
    probs = torch.softmax(logits_l, dim=-1)
    log_probs = torch.log_softmax(logits_l, dim=-1)
    preds = torch.argmax(probs, dim=-1)

    # Balanced accuracy and macro F1 over classes with support in this batch.
    recalls: list[float] = []
    f1s: list[float] = []
    for cls in range(int(n_classes)):
        y_true = labels_l == cls
        support = int(y_true.sum().item())
        if support == 0:
            continue

        y_pred = preds == cls
        tp = int((y_true & y_pred).sum().item())
        fn = int((y_true & ~y_pred).sum().item())
        fp = int((~y_true & y_pred).sum().item())

        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        f1 = 2.0 * precision * recall / max(1e-8, precision + recall)

        recalls.append(recall)
        f1s.append(f1)

    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    topk = min(2, probs.shape[-1])
    top2_idx = torch.topk(probs, k=topk, dim=-1).indices
    top2_match = (top2_idx == labels_l.unsqueeze(1)).any(dim=1)
    top2_accuracy = float(top2_match.float().mean().item())

    nll = float((-log_probs.gather(1, labels_l.unsqueeze(1)).mean()).item())

    one_hot = torch.nn.functional.one_hot(labels_l, num_classes=probs.shape[-1]).to(probs.dtype)
    brier = float(((probs - one_hot).pow(2).sum(dim=-1).mean()).item())

    conf, _ = probs.max(dim=-1)
    acc = (preds == labels_l).to(probs.dtype)
    ece = 0.0
    for i in range(max(1, int(ece_bins))):
        lo = i / ece_bins
        hi = (i + 1) / ece_bins
        in_bin = (conf >= lo) & (conf < hi if i < ece_bins - 1 else conf <= hi)
        if not torch.any(in_bin):
            continue
        frac = float(in_bin.float().mean().item())
        avg_conf = float(conf[in_bin].mean().item())
        avg_acc = float(acc[in_bin].mean().item())
        ece += frac * abs(avg_conf - avg_acc)

    return balanced_accuracy, macro_f1, top2_accuracy, nll, brier, float(ece)
