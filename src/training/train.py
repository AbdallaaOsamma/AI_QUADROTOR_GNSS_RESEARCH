"""
Train a PPO agent on the AirSim Quadrotor environment.

Usage:
    python -m src.training.train --config configs/train_ppo.yaml
    python -m src.training.train --config configs/train_ppo.yaml --total_timesteps 4096
"""

import argparse
import copy
import os
import socket
import time
from datetime import datetime

# Default AirSim shortcut — auto-launched if AirSim is not already running.
_DEFAULT_AIRSIM_PATH = (
    r"C:\Users\bedor\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\AirSimNH.lnk"
)
_AIRSIM_STARTUP_TIMEOUT_S = 90


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if something is listening on host:port."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _is_airsim_api_ready(port: int, host: str = "127.0.0.1", timeout: float = 5.0) -> bool:
    """Return True if AirSim's TCP port is accepting connections.

    Note: We only check TCP reachability (not confirmConnection) because
    msgpackrpc's handshake can hang on some Python/tornado versions.
    The environment constructor will surface any deeper API errors.
    """
    return _is_port_open(port, host)


def _is_airsim_error(exc: Exception) -> bool:
    """Return True if the exception looks like an AirSim / msgpack-rpc connection failure."""
    try:
        import msgpackrpc.error
        if isinstance(exc, (msgpackrpc.error.TimeoutError, msgpackrpc.error.TransportError)):
            return True
    except ImportError:
        pass
    return isinstance(exc, (ConnectionRefusedError, ConnectionResetError,
                             BrokenPipeError, OSError))


def launch_airsim_if_needed(port: int, shortcut_path: str) -> None:
    """Launch AirSim via Windows shortcut if not already running, then wait for API."""
    # Fast path: port open AND API responding — nothing to do.
    if _is_port_open(port) and _is_airsim_api_ready(port):
        print(f"[airsim] Already running on port {port}.", flush=True)
        return

    if _is_port_open(port):
        # Port is open but API not yet responding — AirSim may be starting up.
        print(f"[airsim] Port {port} open but API not ready — waiting...", flush=True)
    elif not shortcut_path:
        raise RuntimeError(
            f"AirSim is not running on port {port} and no --airsim_path was given. "
            "Start AirSim manually or pass --airsim_path <path>."
        )
    else:
        print(f"[airsim] Not detected on port {port} — launching AirSimNH...", flush=True)
        print(f"[airsim] Shortcut: {shortcut_path}", flush=True)
        os.startfile(shortcut_path)  # Windows shell open — same as double-clicking the .lnk

    # Wait for BOTH port open AND API ready (not just port open).
    print(f"[airsim] Waiting up to {_AIRSIM_STARTUP_TIMEOUT_S}s for AirSim API...", flush=True)
    t0 = time.time()
    deadline = t0 + _AIRSIM_STARTUP_TIMEOUT_S
    while time.time() < deadline:
        if _is_port_open(port) and _is_airsim_api_ready(port):
            elapsed = int(time.time() - t0)
            print(f"[airsim] AirSim API ready after {elapsed}s.", flush=True)
            return
        elapsed = int(time.time() - t0)
        print(f"[airsim] Waiting... ({elapsed}s)", flush=True)
        time.sleep(3.0)

    raise TimeoutError(
        f"AirSim did not become API-ready on port {port} within {_AIRSIM_STARTUP_TIMEOUT_S}s. "
        "Try starting AirSim manually before running training."
    )


import yaml  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback  # noqa: E402
from stable_baselines3.common.utils import get_schedule_fn  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack  # noqa: E402

from src.environments.airsim_env import AirSimDroneEnv  # noqa: E402
from src.training.callbacks import RewardLoggingCallback  # noqa: E402
from src.training.env_scheduler import EnvironmentScheduler  # noqa: E402


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in-place."""
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def make_env(cfg: dict, port: int | None = None):
    """Return a factory that creates a Monitored AirSimDroneEnv.

    If port is given, override the env.port in a deep-copied config so
    each subprocess gets its own AirSim instance.
    """
    if port is not None:
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("env", {})["port"] = port

    def _init():
        return Monitor(AirSimDroneEnv(cfg))
    return _init


def make_vec_env(cfg: dict, num_envs: int, base_port: int):
    """Create a vectorized environment with N parallel AirSim instances.

    - num_envs=1: DummyVecEnv (single process, backward compatible)
    - num_envs>1: SubprocVecEnv (each env on base_port + i)

    Uses start_method="spawn" for SubprocVecEnv to avoid CUDA fork issues.
    """
    if num_envs == 1:
        return DummyVecEnv([make_env(cfg, port=base_port)])

    env_fns = [make_env(cfg, port=base_port + i) for i in range(num_envs)]
    return SubprocVecEnv(env_fns, start_method="spawn")


def build_model(
    algo: str,
    env,
    cfg: dict,
    log_dir: str,
    resume: str | None = None,
):
    """Factory that constructs or resumes a PPO or SAC model.

    Parameters
    ----------
    algo:
        Algorithm name — ``'ppo'`` or ``'sac'`` (case-insensitive).
    env:
        Vectorised Gymnasium environment passed to the model constructor.
    cfg:
        Flat dict of algorithm hyperparameters (keys match SB3 constructor
        kwargs, plus ``total_timesteps`` which is handled by the caller).
    log_dir:
        TensorBoard log directory for the run.
    resume:
        Optional path to a ``.zip`` checkpoint from which to resume
        training.  When given the matching SB3 class is used to load the
        file; ``env`` and ``tensorboard_log`` are rebound to the new run.
    """
    algo = algo.lower()

    if algo == "ppo":
        cls = PPO
    elif algo == "sac":
        cls = SAC
    else:
        raise ValueError(
            f"Unknown algorithm: {algo!r}. Choose 'ppo' or 'sac'."
        )

    if resume:
        model = cls.load(resume, env=env, tensorboard_log=log_dir)
        # Apply YAML hyperparams so fine-tuning configs (e.g. parking_finetune)
        # take effect. n_steps and batch_size are excluded because the rollout
        # buffer was already allocated at the checkpoint's original size.
        # learning_rate and clip_range are SB3 schedules (callables); wrap any
        # raw float with get_schedule_fn so SB3 can still call them with
        # progress_remaining — otherwise train() raises TypeError.
        # SB3 uses `lr_schedule` internally (not `learning_rate`) — setting
        # `learning_rate` only shadows the attribute, the optimizer still reads
        # `lr_schedule`.  Map the YAML key to the correct internal attribute.
        _internal_key = {
            "learning_rate": "lr_schedule",   # SB3 reads this in _update_learning_rate
            "clip_range":    "clip_range",     # PPO reads this directly as a callable
        }
        _schedule_keys = {"learning_rate", "clip_range"}
        _override_keys = {
            "learning_rate", "clip_range", "ent_coef", "vf_coef",
            "max_grad_norm", "n_epochs", "gamma", "gae_lambda",
        }
        for key, val in cfg.items():
            if key not in _override_keys:
                continue
            attr = _internal_key.get(key, key)
            if not hasattr(model, attr):
                continue
            if key in _schedule_keys:
                setattr(model, attr, get_schedule_fn(val))
            else:
                setattr(model, attr, val)
        return model

    # Build kwargs from cfg — exclude non-SB3 keys handled by the caller.
    _caller_keys = {"total_timesteps"}
    kwargs = {k: v for k, v in cfg.items() if k not in _caller_keys}

    return cls(
        policy="MultiInputPolicy",
        env=env,
        tensorboard_log=log_dir,
        verbose=1,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description="Train PPO/SAC on AirSim")
    parser.add_argument(
        "--algo", type=str, default="ppo", choices=["ppo", "sac"],
        help="RL algorithm: 'ppo' (default) or 'sac'",
    )
    parser.add_argument(
        "--config", type=str, default="configs/train_ppo.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=None,
        help="Override total_timesteps from config",
    )
    parser.add_argument(
        "--reward_config", type=str, default=None,
        help="Override reward weights from a separate YAML (e.g. configs/rewards/aggressive.yaml)",
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Custom run name (default: ppo_TIMESTAMP)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint .zip to resume training from",
    )
    parser.add_argument(
        "--overrides", type=str, default=None,
        help="JSON string of config overrides, e.g. '{\"reward\":{\"w_dist\":0.0}}'",
    )
    parser.add_argument(
        "--num_envs", type=int, default=1,
        help="Number of parallel AirSim environments (each on its own port)",
    )
    parser.add_argument(
        "--base_port", type=int, default=41451,
        help="Base API port for AirSim instances (env i uses base_port + i)",
    )
    parser.add_argument(
        "--airsim_path", type=str, default=_DEFAULT_AIRSIM_PATH,
        help="Path to AirSimNH shortcut or .exe — auto-launched if AirSim is not running.",
    )
    parser.add_argument(
        "--no_eval", action="store_true",
        help=(
            "Disable periodic evaluation (useful when only one AirSim instance is "
            "running and sharing it between training and eval would reset episodes)."
        ),
    )
    args = parser.parse_args()

    # --- Auto-launch AirSim if not running ---
    print(f"[train:{args.algo}] Starting...", flush=True)
    launch_airsim_if_needed(port=args.base_port, shortcut_path=args.airsim_path)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply JSON overrides (highest priority — applied before reward_config)
    if args.overrides:
        import json
        _deep_merge(cfg, json.loads(args.overrides))

    # Override reward weights if separate reward config provided
    if args.reward_config:
        with open(args.reward_config, "r") as f:
            reward_override = yaml.safe_load(f)
        cfg.setdefault("reward", {}).update(reward_override)

    algo = args.algo
    # Fall back to "ppo" section when a PPO config is used with --algo ppo
    algo_cfg = cfg.get(algo, cfg.get("ppo", {}))
    out_cfg = cfg["output"]
    frame_stack = cfg.get("frame_stack", 4)

    total_timesteps = args.total_timesteps or algo_cfg["total_timesteps"]

    # Timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{algo}_{timestamp}"
    run_dir = os.path.join(out_cfg["log_dir"], run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Environments ---
    num_envs = args.num_envs
    base_port = args.base_port

    print(f"[train:{algo}] Creating training environment (port {base_port})...", flush=True)
    train_env = make_vec_env(cfg, num_envs=num_envs, base_port=base_port)
    train_env = VecFrameStack(train_env, n_stack=frame_stack, channels_order="last")
    print(f"[train:{algo}] Training environment ready.", flush=True)

    # Eval env: always use a dedicated port (base_port + num_envs) so the eval
    # drone never resets the training drone mid-episode.  When --no_eval is set
    # (or the dedicated port is unreachable) evaluation is skipped entirely.
    eval_port = base_port + num_envs
    use_eval = not args.no_eval
    if use_eval and not _is_port_open(eval_port):
        print(
            f"[train:{algo}] WARNING: eval port {eval_port} unreachable. "
            "Disabling periodic evaluation. Pass --no_eval to suppress this warning.",
            flush=True,
        )
        use_eval = False

    eval_env = None
    if use_eval:
        print(f"[train:{algo}] Creating eval environment (port {eval_port})...", flush=True)
        eval_env = make_vec_env(cfg, num_envs=1, base_port=eval_port)
        eval_env = VecFrameStack(eval_env, n_stack=frame_stack, channels_order="last")
        print(f"[train:{algo}] Eval environment ready.", flush=True)

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=out_cfg.get("checkpoint_freq", 10000),
        save_path=ckpt_dir,
        name_prefix=algo,
    )

    reward_cb = RewardLoggingCallback()
    callbacks = [checkpoint_cb, reward_cb]

    if use_eval:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_dir, "best_model"),
            log_path=os.path.join(run_dir, "eval_logs"),
            eval_freq=out_cfg.get("eval_freq", 5000),
            n_eval_episodes=out_cfg.get("eval_episodes", 5),
            deterministic=True,
        )
        callbacks.append(eval_cb)
    else:
        print(f"[train:{algo}] Periodic evaluation disabled.", flush=True)

    # --- Model ---
    if args.resume:
        print(f"[train] Resuming {algo.upper()} from checkpoint: {args.resume}")
    model = build_model(algo, train_env, algo_cfg, log_dir=run_dir, resume=args.resume)

    # After loading, warn the user how many steps actually remain.  If the
    # checkpoint already reaches or exceeds the ceiling the run would complete
    # instantly with zero new steps — hard-exit instead of silently doing nothing.
    if args.resume:
        _ckpt_steps = model.num_timesteps
        _remaining = total_timesteps - _ckpt_steps
        print(
            f"[train:{algo}] Checkpoint at step {_ckpt_steps}. "
            f"Ceiling: {total_timesteps}. Remaining: {_remaining} steps.",
            flush=True,
        )
        if _remaining <= 0:
            raise SystemExit(
                f"[train:{algo}] ERROR: checkpoint is already at or past "
                f"total_timesteps={total_timesteps}. "
                "Increase total_timesteps in the YAML config before resuming."
            )

    eval_info = f"eval on {eval_port}" if use_eval else "eval disabled"
    print(f"[train:{algo}] Run directory: {run_dir}")
    print(f"[train:{algo}] Total timesteps: {total_timesteps}")
    print(f"[train:{algo}] Parallel envs: {num_envs} (ports {base_port}–{base_port + num_envs - 1},"
          f" {eval_info})")

    # Multi-environment rotation
    multi_env_cfg = cfg.get("multi_env", {})
    if multi_env_cfg.get("enabled", False):
        env_config_paths = multi_env_cfg.get("configs", [])
        rotate_every = multi_env_cfg.get("rotate_every_episodes", 50)
        if env_config_paths:
            env_scheduler = EnvironmentScheduler.from_config_paths(
                env_config_paths, rotate_every_episodes=rotate_every
            )
            callbacks.append(env_scheduler)
            print(f"[train:{algo}] Multi-env rotation ENABLED ({len(env_config_paths)} configs, "
                  f"rotate every {rotate_every} episodes)")

    # --- Training loop with automatic AirSim crash recovery ---
    _MAX_RESTARTS = 5
    _restart_count = 0
    _reset_num_ts = not bool(args.resume)

    try:
        while True:
            try:
                model.learn(
                    total_timesteps=total_timesteps,
                    callback=callbacks,
                    reset_num_timesteps=_reset_num_ts,
                )
                break  # completed normally

            except KeyboardInterrupt:
                print(f"\n[train:{algo}] Training interrupted by user.")
                break

            except Exception as _exc:
                # Only recover from AirSim connection failures.
                if not _is_airsim_error(_exc) or _restart_count >= _MAX_RESTARTS:
                    raise

                _restart_count += 1
                _steps_at_crash = model.num_timesteps
                print(
                    f"\n[train:{algo}] AirSim connection lost at step {_steps_at_crash}: {_exc}",
                    flush=True,
                )
                print(
                    f"[train:{algo}] Auto-recovery {_restart_count}/{_MAX_RESTARTS} — "
                    "saving checkpoint...",
                    flush=True,
                )

                # Save current weights to a recovery checkpoint (no AirSim needed).
                _rec_path = os.path.join(ckpt_dir, f"recovery_{_restart_count}")
                model.save(_rec_path)
                print(f"[train:{algo}] Recovery checkpoint: {_rec_path}.zip", flush=True)

                # Close the broken environment gracefully (already ignores errors).
                try:
                    train_env.close()
                except Exception:
                    pass

                # Wait for AirSim to restart.
                print(f"[train:{algo}] Waiting for AirSim on port {base_port}...", flush=True)
                try:
                    launch_airsim_if_needed(base_port, args.airsim_path)
                except (TimeoutError, RuntimeError) as _launch_err:
                    print(
                        f"[train:{algo}] AirSim did not restart: {_launch_err}\n"
                        f"[train:{algo}] Resume manually with: "
                        f"--resume {_rec_path}.zip",
                        flush=True,
                    )
                    break  # Exit loop; finally saves final model.

                # Rebuild environment with fresh AirSim connection.
                train_env = make_vec_env(cfg, num_envs=num_envs, base_port=base_port)
                train_env = VecFrameStack(train_env, n_stack=frame_stack, channels_order="last")

                # Reload model from recovery checkpoint to rebind new env.
                model = build_model(
                    algo, train_env, algo_cfg,
                    log_dir=run_dir,
                    resume=f"{_rec_path}.zip",
                )
                _reset_num_ts = False  # Step counter already correct in checkpoint.
                print(
                    f"[train:{algo}] Recovery complete. "
                    f"Resuming from step {model.num_timesteps}...",
                    flush=True,
                )

    finally:
        final_path = os.path.join(run_dir, "final_model")
        try:
            model.save(final_path)
            print(f"[train:{algo}] Final model saved to {final_path}.zip")
        except Exception as _save_err:
            print(f"[train:{algo}] WARNING: could not save final model: {_save_err}")
        try:
            train_env.close()
        except Exception:
            pass
        if eval_env is not None:
            try:
                eval_env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
