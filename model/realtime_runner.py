from __future__ import annotations

import argparse
import time

from .config import ModelConfig
from .network import MovementDecoder
from .reward import ProgrammaticReward
from .stream import ZMQEmbeddingReceiver
from .trainer import OnlineTrainer
from .types import StreamSample
from .viz import RealtimeManifoldVisualizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time movement decoder with manifold visualization and online adaptation"
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--embedding-key", type=str, default="data")
    parser.add_argument("--input-dim", type=int, default=None)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, choices=[2, 3], default=2)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=4000)
    parser.add_argument("--update-every", type=int, default=10)
    parser.add_argument("--warmup-labeled", type=int, default=200)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--heartbeat-every", type=int, default=20)
    parser.add_argument("--no-viz", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = ModelConfig(
        host=args.host,
        port=args.port,
        embedding_key=args.embedding_key,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        update_every=args.update_every,
        warmup_labeled_samples=args.warmup_labeled,
        heartbeat_every=args.heartbeat_every,
        viz_enabled=not args.no_viz,
        device=args.device,
    )

    receiver = ZMQEmbeddingReceiver(
        host=cfg.host,
        port=cfg.port,
        embedding_key=cfg.embedding_key,
        queue_capacity=cfg.queue_capacity,
        receiver_timeout_ms=cfg.receiver_timeout_ms,
    )
    receiver.start()

    print(f"Receiver connected to tcp://{cfg.host}:{cfg.port} using key='{cfg.embedding_key}'")
    print("Waiting for first sample...")

    first_sample = _wait_for_first_sample(receiver)
    if first_sample is None:
        raise RuntimeError("No samples received from stream")

    inferred_input_dim = int(first_sample.embedding.shape[0])
    if cfg.input_dim is not None and cfg.input_dim != inferred_input_dim:
        raise ValueError(
            f"Configured input_dim={cfg.input_dim} does not match stream dim={inferred_input_dim}"
        )
    cfg.input_dim = inferred_input_dim

    device = cfg.resolve_device()
    model = MovementDecoder(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        embedding_dim=cfg.embedding_dim,
        n_classes=cfg.n_classes,
        projection_dim=cfg.projection_dim,
        dropout=cfg.dropout,
    )

    reward_provider = ProgrammaticReward(cfg)
    trainer = OnlineTrainer(model=model, cfg=cfg, reward_provider=reward_provider, device=device)

    visualizer = None
    if cfg.viz_enabled:
        visualizer = RealtimeManifoldVisualizer(
            projection_dim=cfg.projection_dim,
            history_len=cfg.viz_history,
            ema_alpha=cfg.viz_ema_alpha,
            draw_every=cfg.viz_draw_every,
        )

    print(
        f"Model ready: input_dim={cfg.input_dim}, embedding_dim={cfg.embedding_dim}, "
        f"projection_dim={cfg.projection_dim}, device={device}"
    )
    print("Streaming loop started. Press Ctrl-C to stop.")

    processed = 0
    pending: list[StreamSample] = [first_sample]

    try:
        while True:
            sample = pending.pop() if pending else receiver.get(timeout=1.0)
            if sample is None:
                continue

            if sample.embedding.shape[0] != cfg.input_dim:
                continue

            step = trainer.process_sample(sample)
            processed += 1

            if visualizer is not None:
                visualizer.update(step)

            if processed % cfg.heartbeat_every == 0:
                _print_heartbeat(step, trainer)

    except KeyboardInterrupt:
        print("Stopping real-time runner...")
    finally:
        receiver.stop()
        if visualizer is not None:
            visualizer.close()


def _wait_for_first_sample(receiver: ZMQEmbeddingReceiver, timeout: float = 10.0) -> StreamSample | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        sample = receiver.get(timeout=0.25)
        if sample is not None:
            return sample
    return None


def _print_heartbeat(step, trainer: OnlineTrainer) -> None:
    train_str = ""
    if step.training is not None and step.training.update_applied:
        train_str = (
            f" | loss={step.training.total_loss:.4f}"
            f" sup={step.training.supervised_loss:.4f}"
            f" pol={step.training.policy_loss:.4f}"
            f" rl={'on' if step.training.rl_enabled else 'off'}"
        )

    print(
        f"[{step.sample_idx:6d}] pred={step.predicted_class} "
        f"conf={step.confidence:.2f} reward={step.reward:.2f} "
        f"baseline={trainer.reward_baseline:.2f} "
        f"labeled_seen={trainer.labeled_seen} updates={trainer.num_updates}{train_str}"
    )


if __name__ == "__main__":
    main()
