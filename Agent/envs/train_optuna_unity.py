from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import torch
from optuna.exceptions import TrialPruned
from optuna.trial import Trial
from torch.utils.tensorboard import SummaryWriter

from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


sys.path.append(str(Path(__file__).parent.parent))
from PPO.ppo import PPOClip, PPOConfig  


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Búsqueda de hiperparámetros para PPO usando Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Ruta al ejecutable de Unity. Usa 'editor' para conectarte al Unity Editor.",
    )
    parser.add_argument(
        "--no-graphics",
        action="store_true",
        help="Ejecuta Unity en modo headless (solo aplica cuando se usa un build).",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=20.0,
        help="Factor de aceleración del tiempo dentro de Unity.",
    )
    parser.add_argument("--base-port", type=int, default=5004, help="Puerto base para ML-Agents.")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID para ML-Agents.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla base para reproducibilidad.")

    parser.add_argument("--trials", type=int, default=10, help="Número de trials Optuna (<=0 = infinito).")
    parser.add_argument(
        "--trial-steps",
        type=int,
        default=1_000_000,
        help="Cantidad máxima de pasos de entorno por trial.",
    )
    parser.add_argument(
        "--metric-window",
        type=int,
        default=20,
        help="Número de episodios usados para promediar la recompensa reportada a Optuna.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=50_000,
        help="Cada cuántos pasos reportar el promedio a Optuna para permitir pruning.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="ppo-car-agent",
        help="Nombre de la study de Optuna.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="URL/Path para almacenar la study (ej: sqlite:///optuna.db).",
    )
    parser.add_argument(
        "--sampler",
        choices=["tpe", "random"],
        default="tpe",
        help="Sampler de Optuna.",
    )
    parser.add_argument(
        "--pruner",
        choices=["median", "none"],
        default="median",
        help="Pruner para detener trials sin progreso.",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=1,
        help="Cantidad de mejores trials cuyos modelos se copiarán a resultados/best_models.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/optuna_ppo",
        help="Directorio raíz para guardar modelos y resultados.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="Directorio raíz para logs de TensorBoard (uno por trial).",
    )
    return parser.parse_args()


def find_macos_binary(app_path: Path) -> Path:
    """Devuelve el ejecutable dentro de un .app."""
    macos_dir = app_path / "Contents" / "MacOS"
    if not macos_dir.exists():
        raise FileNotFoundError(f"No se encontró Contents/MacOS dentro de {app_path}")
    for candidate in sorted(macos_dir.iterdir()):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No se encontró binario dentro de {macos_dir}")


def detect_default_build(script_dir: Path) -> Optional[Path]:
    """Intenta detectar un build acorde al sistema operativo."""
    build_dir = script_dir.parent / "Build"
    if not build_dir.exists():
        return None

    candidates = []
    platform = sys.platform

    if platform.startswith("linux"):
        candidates.extend(sorted(build_dir.glob("*.x86_64")))
    elif platform == "darwin":
        candidates.extend(sorted(build_dir.glob("*.app")))
    elif os.name == "nt":
        candidates.extend(sorted(build_dir.glob("*.exe")))

    # fallback generales
    fallback = [
        build_dir / "RunCar.x86_64",
        build_dir / "Applying EANNs.exe",
        build_dir / "Build2.app",
    ]
    for item in fallback:
        if item.exists():
            candidates.append(item)

    if not candidates:
        for path in sorted(build_dir.iterdir()):
            if path.is_file() and os.access(path, os.X_OK):
                candidates.append(path)

    return candidates[0] if candidates else None


def resolve_unity_path(args: argparse.Namespace) -> Tuple[Optional[str], Optional[Path]]:
    """Determina qué ejecutable de Unity usar (o None para Unity Editor)."""
    script_dir = Path(__file__).parent.parent
    unity_env_path: Optional[str] = None
    original_build_path: Optional[Path] = None

    if args.env is None:
        detected = detect_default_build(script_dir)
        if detected:
            unity_env_path = str(detected.resolve())
            original_build_path = detected.resolve()
            print(f"Build detectado automáticamente: {unity_env_path}")
        else:
            print("⚠ No se detectó un build. Se usará Unity Editor (modo Play).")
    else:
        if args.env.lower() == "editor":
            print("Usando Unity Editor. Recuerda presionar Play cuando se indique.")
            unity_env_path = None
        else:
            env_path = Path(args.env).expanduser()
            if not env_path.is_absolute():
                env_path = (script_dir / env_path).resolve()
            env_path = env_path.resolve()

            if sys.platform == "darwin" and env_path.suffix == ".exe":
                raise RuntimeError("El build .exe no puede ejecutarse nativamente en macOS.")

            if not env_path.exists():
                raise FileNotFoundError(
                    f"No se encontró el ejecutable de Unity en: {env_path}"
                )

            if os.name != "nt":
                if not os.access(env_path, os.X_OK):
                    print(f"Agregando permisos de ejecución a {env_path}")
                    os.chmod(env_path, 0o755)

            unity_env_path = str(env_path)
            original_build_path = env_path

    return unity_env_path, original_build_path


def maybe_enable_headless(unity_env_path: Optional[str], args: argparse.Namespace) -> Optional[str]:
    """Crea un wrapper para ejecutar Unity sin ventana si es necesario."""
    if not args.no_graphics:
        return unity_env_path

    if unity_env_path is None:
        print("⚠ --no-graphics se ignora cuando se usa Unity Editor.")
        return None

    app_path = Path(unity_env_path)
    if sys.platform == "darwin" and app_path.suffix == ".app":
        binary_path = find_macos_binary(app_path)
    else:
        binary_path = app_path

    script_dir = Path(__file__).parent.parent
    wrapper_dir = script_dir / ".tmp"
    wrapper_dir.mkdir(exist_ok=True)

    wrapper_script = wrapper_dir / "run_headless_optuna.sh"
    wrapper_content = f"""#!/bin/bash
exec "{binary_path}" -batchmode -nographics "$@"
"""
    wrapper_script.write_text(wrapper_content)
    wrapper_script.chmod(0o755)
    print(f"Modo headless activado: {wrapper_script}")
    return str(wrapper_script)


def configure_unity_environment(
    unity_env_path: Optional[str],
    args: argparse.Namespace,
) -> Tuple[UnityEnvironment, str, int, int]:
    """Inicia UnityEnvironment y retorna la metadata requerida para entrenar."""
    engine_channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name=unity_env_path,
        seed=args.seed,
        side_channels=[engine_channel],
        worker_id=args.worker_id,
        base_port=args.base_port,
    )

    if args.no_graphics and unity_env_path:
        width, height = 84, 84
    else:
        width, height = 800, 600

    engine_channel.set_configuration_parameters(
        width=width,
        height=height,
        time_scale=args.time_scale,
    )

    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]
    obs_dim = spec.observation_specs[0].shape[0]
    act_dim = spec.action_spec.continuous_size

    print(f"✓ Conectado a Unity (Behavior: {behavior_name}, obs={obs_dim}, act={act_dim})")
    return env, behavior_name, obs_dim, act_dim


def config_to_dict(cfg: PPOConfig) -> Dict[str, float]:
    """Convierte PPOConfig a un dict serializable."""
    return {
        "learning_rate": cfg.learning_rate,
        "gamma": cfg.gamma,
        "gae_lambda": cfg.gae_lambda,
        "clip_range": cfg.clip_range,
        "batch_size": cfg.batch_size,
        "n_steps": cfg.n_steps,
        "n_epochs": cfg.n_epochs,
        "hidden_sizes": list(cfg.hidden_sizes),
        "ent_coef": cfg.ent_coef,
        "vf_coef": cfg.vf_coef,
        "max_grad_norm": cfg.max_grad_norm,
    }

def sample_config(trial: Trial) -> PPOConfig:
    """
    Define el espacio de búsqueda para Optuna centrado en estabilidad y exploración.
    """
    
    hidden_options = [
        (128, 128),
        (256, 256),
        (256, 256, 128),
        (512, 256),
    ]
    return PPOConfig(
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True),
        gamma=trial.suggest_float("gamma", 0.98, 0.999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.92, 0.98),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.3),
        batch_size=trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        n_steps=trial.suggest_categorical("n_steps", [2048, 4096, 8192]),
        n_epochs=trial.suggest_int("n_epochs", 3, 10),
        hidden_sizes=trial.suggest_categorical("hidden_sizes", hidden_options),
        ent_coef=trial.suggest_float("ent_coef", 0.01, 0.1),
        vf_coef=trial.suggest_float("vf_coef", 0.4, 0.8),
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.3, 0.8),
    )

    
def run_single_trial(
    trial: Trial,
    env: UnityEnvironment,
    behavior_name: str,
    obs_dim: int,
    act_dim: int,
    device: torch.device,
    args: argparse.Namespace,
    log_dir: Path,
) -> Tuple[float, PPOConfig, Path]:
    """Ejecuta un trial de entrenamiento completo y retorna su score."""
    config = sample_config(trial)
    agent = PPOClip(obs_dim, act_dim, config, device, discrete=False)

    with torch.no_grad():
        agent.policy.log_std.fill_(-0.5)

    writer = SummaryWriter(log_dir=str(log_dir))
    env.reset()

    pending_transitions: Dict[int, Dict[str, np.ndarray]] = {}
    episode_scores: Dict[int, float] = {}
    episode_rewards = []

    total_steps = 0
    episode_count = 0
    best_window_avg = -np.inf
    metric_window = max(1, args.metric_window)
    report_every = max(1, args.report_every)
    last_report_step = 0

    try:
        while total_steps < args.trial_steps:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # procesar episodios que terminaron
            for agent_id in terminal_steps.agent_id:
                term_step = terminal_steps[agent_id]
                final_reward = term_step.reward

                if agent_id in pending_transitions:
                    trans = pending_transitions.pop(agent_id)
                    agent.buffer.add(
                        obs=trans["obs"],
                        act=trans["action"],
                        logp=trans["logp"],
                        rew=final_reward,
                        done=True,
                        val=trans["val"],
                    )
                    total_steps += 1

                total_episode_reward = episode_scores.pop(agent_id, 0.0) + final_reward
                episode_rewards.append(total_episode_reward)
                episode_count += 1
                writer.add_scalar("reward/episode", total_episode_reward, episode_count)

                if len(episode_rewards) >= metric_window:
                    window_avg = float(np.mean(episode_rewards[-metric_window:]))
                    best_window_avg = max(best_window_avg, window_avg)
                    writer.add_scalar("reward/avg_window", window_avg, episode_count)
                    if total_steps - last_report_step >= report_every:
                        trial.report(window_avg, total_steps)
                        last_report_step = total_steps
                        if trial.should_prune():
                            raise TrialPruned()

            # obtener acciones para agentes activos
            if len(decision_steps) > 0:
                actions = []
                for agent_id in decision_steps.agent_id:
                    obs = decision_steps[agent_id].obs[0]
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    action, logp, val = agent.select_action(obs_tensor)
                    pending_transitions[agent_id] = {
                        "obs": obs,
                        "action": action.numpy(),
                        "logp": logp,
                        "val": val,
                    }
                    actions.append(action.numpy())

                if actions:
                    action_tuple = ActionTuple(continuous=np.vstack(actions))
                    env.set_actions(behavior_name, action_tuple)

            # avanza simulacion
            env.step()

            decision_steps_next, _ = env.get_steps(behavior_name)

            # completar transiciones con reward (done=False)
            for agent_id in decision_steps_next.agent_id:
                if agent_id in pending_transitions:
                    step_reward = decision_steps_next[agent_id].reward
                    episode_scores[agent_id] = episode_scores.get(agent_id, 0.0) + step_reward
                    trans = pending_transitions.pop(agent_id)
                    agent.buffer.add(
                        obs=trans["obs"],
                        act=trans["action"],
                        logp=trans["logp"],
                        rew=step_reward,
                        done=False,
                        val=trans["val"],
                    )
                    total_steps += 1

            # actualizar politica cuando el buffer este lleno
            if agent.buffer.is_full():
                if len(decision_steps_next) > 0:
                    last_agent = decision_steps_next.agent_id[0]
                    last_obs = decision_steps_next[last_agent].obs[0]
                    last_value = (
                        agent.value(
                            torch.tensor(last_obs, dtype=torch.float32)
                            .to(device)
                            .unsqueeze(0)
                        )
                        .squeeze(-1)
                        .item()
                    )
                else:
                    last_value = 0.0

                batch = agent.buffer.get(
                    last_value=last_value,
                    gamma=config.gamma,
                    lam=config.gae_lambda,
                )
                metrics = agent.update(batch)
                total_loss = metrics.get("loss_total", metrics["loss_policy"] + metrics["loss_value"])
                writer.add_scalar("loss/policy", metrics["loss_policy"], total_steps)
                writer.add_scalar("loss/value", metrics["loss_value"], total_steps)
                writer.add_scalar("loss/total", total_loss, total_steps)
                writer.add_scalar("metrics/entropy", metrics["entropy"], total_steps)

                display_avg = (
                    best_window_avg if best_window_avg > -np.inf else (episode_rewards[-1] if episode_rewards else 0.0)
                )
                print(
                    f"[Trial {trial.number}] step={total_steps} "
                    f"avg_reward={display_avg:.2f} loss={total_loss:.4f}"
                )

    finally:
        writer.flush()
        writer.close()

    if not episode_rewards:
        final_avg = -1.0
    else:
        final_avg = (
            float(np.mean(episode_rewards[-metric_window:]))
            if len(episode_rewards) >= metric_window
            else float(np.mean(episode_rewards))
        )

    trials_dir = Path(args.save_dir) / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    model_path = trials_dir / f"trial_{trial.number:03d}.pt"
    agent.save(str(model_path))

    return final_avg, config, model_path


def build_study(args: argparse.Namespace) -> optuna.Study:
    sampler: optuna.samplers.BaseSampler
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=args.seed)

    if args.pruner == "median":
        pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    else:
        pruner = optuna.pruners.NopPruner()

    return optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=args.storage is not None,
    )


def save_all_trial_configs(study: optuna.Study, args: argparse.Namespace) -> None:
    """Guarda las configuraciones de todos los trials completados."""
    completed_trials = [
        t for t in study.trials 
        if t.state == optuna.trial.TrialState.COMPLETE and "config" in t.user_attrs
    ]
    if not completed_trials:
        print("No hay trials completados para guardar configuraciones.")
        return

    # ordenar por reward (mejor primero)
    completed_trials.sort(key=lambda t: t.value if t.value is not None else -np.inf, reverse=True)
    
    all_configs = []
    for trial in completed_trials:
        trial_data = {
            "trial_number": trial.number,
            "reward": float(trial.value) if trial.value is not None else None,
            "state": trial.state.name,
            "config": trial.user_attrs.get("config", {}),
            "params": {k: float(v) if isinstance(v, (int, float)) else v for k, v in trial.params.items()},
            "model_path": trial.user_attrs.get("model_path", ""),
            "log_dir": trial.user_attrs.get("log_dir", ""),
        }
        all_configs.append(trial_data)
    
    all_configs_path = Path(args.save_dir) / "all_trials_configs.json"
    with all_configs_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "total_trials": len(all_configs),
                "trials": all_configs,
            },
            fh,
            indent=2,
        )
    print(f"✓ Configuraciones de {len(all_configs)} trials guardadas en: {all_configs_path}")


def save_best_models(study: optuna.Study, args: argparse.Namespace) -> None:
    """Copia los mejores modelos a un subdirectorio dedicado."""
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and "model_path" in t.user_attrs
    ]
    if not completed_trials:
        print("No hay trials completados para guardar modelos.")
        return

    completed_trials.sort(key=lambda t: t.value, reverse=True)
    k = min(args.save_top_k, len(completed_trials))
    best_dir = Path(args.save_dir) / "best_models"
    best_dir.mkdir(parents=True, exist_ok=True)

    for rank, trial in enumerate(completed_trials[:k], start=1):
        src = Path(trial.user_attrs["model_path"])
        if not src.exists():
            continue
        dst = best_dir / f"best_{rank:02d}_trial_{trial.number}_reward_{trial.value:.2f}.pt"
        shutil.copy2(src, dst)
        print(f"✓ Modelo copiado: {dst}")

    best_trial = completed_trials[0]
    best_config_path = Path(args.save_dir) / "best_trial_config.json"
    with best_config_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "trial_number": best_trial.number,
                "reward": best_trial.value,
                "config": best_trial.user_attrs.get("config", {}),
            },
            fh,
            indent=2,
        )
    print(f"Configuración del mejor trial guardada en: {best_config_path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_root = Path(args.tensorboard_dir) if args.tensorboard_dir else (save_dir / "tensorboard")
    tensorboard_root.mkdir(parents=True, exist_ok=True)

    unity_env_path, _ = resolve_unity_path(args)
    unity_env_path = maybe_enable_headless(unity_env_path, args)

    if unity_env_path:
        print(f"Ejecutable seleccionado: {unity_env_path}")
    else:
        print("Unity Editor será la fuente del entorno.")

    env = None
    try:
        env, behavior_name, obs_dim, act_dim = configure_unity_environment(unity_env_path, args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {device}")

        study = build_study(args)

        def objective(trial: Trial) -> float:
            # directorio unico para tensorboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trial_log_dir = tensorboard_root / f"trial_{trial.number:03d}_{timestamp}"
            trial_log_dir.mkdir(parents=True, exist_ok=True)

            env.reset()
            trial_value, cfg, model_path = run_single_trial(
                trial,
                env,
                behavior_name,
                obs_dim,
                act_dim,
                device,
                args,
                trial_log_dir,
            )
            trial.set_user_attr("config", config_to_dict(cfg))
            trial.set_user_attr("model_path", str(model_path))
            trial.set_user_attr("log_dir", str(trial_log_dir))
            print(f"Trial {trial.number} finalizó con reward medio {trial_value:.2f}")
            return trial_value

        n_trials = args.trials if args.trials > 0 else None
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # guardar configuraciones de todos los trials
        save_all_trial_configs(study, args)

        if study.best_trial is not None:
            print("\n===== Resultados Optuna =====")
            print(f"Trial ganador: #{study.best_trial.number} con reward {study.best_trial.value:.2f}")
            for k, v in study.best_trial.params.items():
                print(f"  {k}: {v}")
            save_best_models(study, args)
            print("\nPara visualizar TensorBoard:")
            print(f"  tensorboard --logdir {tensorboard_root}")
        else:
            print("Optuna terminó sin trials completados.")

    except KeyboardInterrupt:
        print("\nBúsqueda interrumpida por el usuario.")
    except Exception as exc:
        print(f"\n✗ Error durante la búsqueda con Optuna: {exc}")
        raise
    finally:
        if env is not None:
            env.close()
            print("Entorno de Unity cerrado.")


if __name__ == "__main__":
    main()

