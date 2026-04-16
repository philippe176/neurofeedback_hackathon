from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import ModelConfig
from .losses import (
    entropy_regularization,
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
        pred = int(np.argmax(probs))
        confidence = float(probs[pred])
        reward = self.reward_provider.compute(sample, probs)

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

        supervised = supervised_classification_loss(out.logits, y)
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

        if labeled_in_batch == 0 and not rl_enabled:
            return _empty_metrics(rl_enabled=False)

        total = (
            self.cfg.supervised_weight * supervised
            + self.cfg.policy_weight * policy
            - self.cfg.entropy_weight * entropy
            + self.cfg.smoothness_weight * smoothness
        )

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        self.num_updates += 1

        return TrainingMetrics(
            update_applied=True,
            total_loss=float(total.detach().cpu().item()),
            supervised_loss=float(supervised.detach().cpu().item()),
            policy_loss=float(policy.detach().cpu().item()),
            entropy=float(entropy.detach().cpu().item()),
            smoothness_loss=float(smoothness.detach().cpu().item()),
            labeled_in_batch=labeled_in_batch,
            rl_enabled=rl_enabled,
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
        entropy=0.0,
        smoothness_loss=0.0,
        labeled_in_batch=0,
        rl_enabled=rl_enabled,
    )
