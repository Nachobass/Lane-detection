#!/usr/bin/env python3
"""
Script para evaluar un modelo PPO entrenado en Unity sin entrenar.

Este script carga un modelo guardado y lo ejecuta en Unity para ver su comportamiento.

Uso:
    python eval_car_ppo.py --weights results/custom_ppo/ppo_final.pt [--episodes N] [--env PATH]
"""

import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# Ajustar el path para importar tu PPO
sys.path.append(str(Path(__file__).parent.parent))
from PPO.ppo import PPOClip


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluar modelo PPO entrenado en Unity",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Ruta al archivo de pesos del modelo (.pt)"
    )
    
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Ruta al ejecutable de Unity (opcional). Por defecto usa el build de Linux en ../Build/Run50CarsTrack1/Run50CarsTrack1.x86_64. Usa 'editor' para usar Unity Editor."
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Número de episodios a evaluar (default: 10)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed para reproducibilidad (default: 1)"
    )
    
    parser.add_argument(
        "--base-port",
        type=int,
        default=5004,
        help="Puerto base para comunicación con Unity (default: 5004)"
    )
    
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="ID del worker (default: 0)"
    )
    
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Time scale de Unity (default: 1.0, usa 1.0 para ver la simulación a velocidad normal)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("Evaluación de Modelo PPO en Unity")
    print("=" * 70)
    print()
    
    # Verificar que el archivo de pesos existe
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        script_dir = Path(__file__).parent.parent
        weights_path = script_dir / args.weights
    
    if not weights_path.exists():
        print(f"✗ Error: Archivo de pesos no encontrado: {weights_path}")
        print(f"  Ruta proporcionada: {args.weights}")
        return
    
    print(f"Modelo: {weights_path}")
    print(f"Episodios: {args.episodes}")
    print()
    
    # Procesar ruta del ejecutable de Unity
    unity_env_path = None
    
    if args.env is None:
        # Buscar el build de Linux por defecto
        script_dir = Path(__file__).parent.parent
        default_build = script_dir.parent / "Build" / "Run50CarsTrack1" / "Run50CarsTrack1.x86_64"
        if default_build.exists():
            unity_env_path = str(default_build.resolve())
            print(f"Usando build de Linux por defecto: {unity_env_path}")
        else:
            print("⚠ Build de Linux no encontrado. Usando Unity Editor.")
            print("  Para usar Unity Editor, presiona Play cuando se indique.")
    elif args.env.lower() == "editor":
        unity_env_path = None
        print("Usando Unity Editor. Asegúrate de presionar Play cuando se indique.")
    elif args.env:
        env_path = Path(args.env).expanduser()
        if not env_path.is_absolute():
            script_dir = Path(__file__).parent.parent
            env_path = (script_dir / env_path).resolve()
        else:
            env_path = env_path.resolve()
        
        if not env_path.exists():
            print(f"✗ Error: El ejecutable no existe en: {env_path}")
            return
        
        unity_env_path = str(env_path)
    
    print()
    
    # Conectar con Unity
    print("Conectando con Unity...")
    engine_channel = EngineConfigurationChannel()
    
    try:
        env = UnityEnvironment(
            file_name=unity_env_path,
            seed=args.seed,
            side_channels=[engine_channel],
            worker_id=args.worker_id,
            base_port=args.base_port
        )
        
        # Configurar Unity
        engine_channel.set_configuration_parameters(
            width=800,
            height=600,
            time_scale=args.time_scale
        )
        
        # Esperar a que el entorno se inicialice
        env.reset()
        
        # Obtener información del comportamiento
        behavior_name = list(env.behavior_specs)[0]
        spec = env.behavior_specs[behavior_name]
        
        obs_dim = spec.observation_specs[0].shape[0]
        act_dim = spec.action_spec.continuous_size
        
        print(f"✓ Conectado a Unity")
        print(f"  Behavior Name: {behavior_name}")
        print(f"  Observation Dimension: {obs_dim}")
        print(f"  Action Dimension: {act_dim}")
        print()
        
        # Cargar el modelo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cargando modelo desde: {weights_path}")
        print(f"Usando dispositivo: {device}")
        
        agent = PPOClip.load(
            str(weights_path),
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
            discrete=False
        )
        
        print("✓ Modelo cargado")
        print()
        
        # Poner el modelo en modo evaluacion (no entrenamiento)
        agent.eval()
        
        # Estadisticas
        returns = []
        episode_lengths = []
        episode_rewards = {}  # {agent_id: accumulated_reward}
        episode_steps = {}    # {agent_id: step_count}
        
        print("=" * 70)
        print("Iniciando evaluación...")
        print("=" * 70)
        print()
        if unity_env_path is None:
            print("NOTA: Asegúrate de que Unity Editor esté en Play mode")
        else:
            print("NOTA: El build de Unity se ejecutará automáticamente")
            print("      Puedes ver la simulación en la ventana de Unity")
        print()
        print(f"{'Episodio':<10} {'Return':<15} {'Steps':<10}")
        print("-" * 40)
        
        episode_count = 0
        
        while episode_count < args.episodes:
            # Obtener datos de todos los agentes
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            # Procesar agentes que terminaron
            for agent_id in terminal_steps.agent_id:
                term_step = terminal_steps[agent_id]
                final_reward = term_step.reward
                
                # Acumular reward final
                if agent_id in episode_rewards:
                    total_reward = episode_rewards[agent_id] + final_reward
                    total_steps = episode_steps[agent_id] + 1
                else:
                    total_reward = final_reward
                    total_steps = 1
                
                returns.append(total_reward)
                episode_lengths.append(total_steps)
                episode_count += 1
                
                print(f"{episode_count:<10} {total_reward:<15.2f} {total_steps:<10}")
                
                # Limpiar tracking de este agente
                if agent_id in episode_rewards:
                    del episode_rewards[agent_id]
                if agent_id in episode_steps:
                    del episode_steps[agent_id]
                
                if episode_count >= args.episodes:
                    break
            
            # Procesar agentes activos que necesitan actuar
            if len(decision_steps) > 0:
                actions_list = []
                
                for agent_id in decision_steps.agent_id:
                    decision_step = decision_steps[agent_id]
                    obs = decision_step.obs[0]
                    step_reward = decision_step.reward
                    
                    # Acumular reward
                    if agent_id not in episode_rewards:
                        episode_rewards[agent_id] = 0.0
                        episode_steps[agent_id] = 0
                    episode_rewards[agent_id] += step_reward
                    episode_steps[agent_id] += 1
                    
                    # Convertir a tensor
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                    
                    # Seleccionar accion (modo evaluacion, sin entrenar)
                    with torch.no_grad():
                        # En lugar de select_action, llamamos directo a la politica
                        # forward devuelve (mu, std). 'mu' es la accion media (sin ruido).
                        mu, _ = agent.policy(obs_tensor) 
                        action = mu
                    
                    actions_list.append(action.numpy())
                
                # Enviar acciones a unity
                if actions_list:
                    actions_np = np.vstack(actions_list)
                    action_tuple = ActionTuple(continuous=actions_np)
                    env.set_actions(behavior_name, action_tuple)
            
            # Avanzar simulacion
            env.step()
        
        # Cerrar entorno
        env.close()
        
        # Mostrar estadisticas finales
        print()
        print("=" * 70)
        print("Resultados de Evaluación")
        print("=" * 70)
        if returns:
            print(f"Episodios evaluados: {len(returns)}")
            print(f"Return promedio: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
            print(f"Return mínimo: {np.min(returns):.2f}")
            print(f"Return máximo: {np.max(returns):.2f}")
        else:
            print("No se completaron episodios durante la evaluación.")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nEvaluación interrumpida por el usuario.")
        if 'env' in locals():
            env.close()
    except Exception as e:
        print(f"\n✗ Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    main()

