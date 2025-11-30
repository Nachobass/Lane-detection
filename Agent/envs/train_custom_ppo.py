#!/usr/bin/env python3
"""
Script para entrenar el agente CarAgent usando TU implementación de PPO con Unity.

Este script usa la Python Low-Level API de ML-Agents para conectar tu algoritmo PPO
con el entorno de Unity, permitiendo que TU código sea el "cerebro" del agente.

Uso:
    python train_custom_ppo.py [--env PATH] [--max-steps N] [--time-scale F]
"""

import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# Ajustar el path para importar tu PPO
sys.path.append(str(Path(__file__).parent.parent))
from PPO.ppo import PPOClip, PPOConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Entrenar CarAgent con tu implementación de PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Ruta al ejecutable de Unity (opcional). Por defecto usa el build de Linux en ../Build/Run50CarsTrack1/Run50CarsTrack1.x86_64. Usa 'editor' para usar Unity Editor."
    )
    
    parser.add_argument(
        "--no-graphics",
        action="store_true",
        help="Ejecutar Unity en modo headless (sin ventana gráfica). Más rápido pero no puedes ver la simulación."
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000000,
        help="Número máximo de pasos de entrenamiento (default: 5000000)"
    )
    
    parser.add_argument(
        "--time-scale",
        type=float,
        default=20.0,
        help="Time scale de Unity para acelerar la simulación (default: 20.0)"
    )
    
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/custom_ppo",
        help="Directorio para guardar modelos (default: results/custom_ppo)"
    )
    
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100000,
        help="Frecuencia de guardado en pasos (default: 100000)"
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
        help="ID del worker (default: 0). Usa un número diferente si el puerto está ocupado"
    )
    
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="Directorio para logs de TensorBoard (default: {save_dir}/tensorboard)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("Entrenamiento con PPO Personalizado")
    print("=" * 70)
    print()
    # Crear directorios necesarios
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar TensorBoard - usar directamente el directorio sin subdirectorios
    if args.tensorboard_dir is None:
        tensorboard_dir = save_dir / "tensorboard"
    else:
        tensorboard_dir = Path(args.tensorboard_dir)
    
    # Crear el directorio si no existe
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    # Usar directamente el directorio de tensorboard (una sola run)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    
    # Procesar ruta del ejecutable de Unity
    unity_env_path = None
    original_build_path = None
    
    # Si no se especifica --env, usar el build de Linux por defecto
    if args.env is None:
        # Buscar el build de Linux por defecto
        script_dir = Path(__file__).parent.parent
        default_build = script_dir.parent / "Build" / "Run50CarsTrack1" / "Run50CarsTrack1.x86_64"
        if default_build.exists():
            original_build_path = default_build.resolve()
            unity_env_path = str(original_build_path)
            print(f"Usando build de Linux por defecto: {unity_env_path}")
        else:
            print("⚠ Build de Linux no encontrado. Usando Unity Editor.")
            print(f"  Buscado en: {default_build}")
            print("  Para usar Unity Editor, presiona Play cuando se indique.")
    elif args.env.lower() == "editor":
        # Usuario quiere usar Unity Editor explícitamente
        unity_env_path = None
        print("Usando Unity Editor. Asegúrate de presionar Play cuando se indique.")
    elif args.env:
        # Convertir ruta relativa a absoluta y expandir ~
        env_path = Path(args.env).expanduser()
        if not env_path.is_absolute():
            # Si es relativa, resolverla desde el directorio del script
            script_dir = Path(__file__).parent.parent
            env_path = (script_dir / env_path).resolve()
        else:
            env_path = env_path.resolve()
        
        # Verificar que el archivo existe
        if not env_path.exists():
            print(f"\n✗ Error: El ejecutable no existe en: {env_path}")
            print(f"  Ruta proporcionada: {args.env}")
            print(f"  Ruta resuelta: {env_path}")
            print("\nVerifica que:")
            print("  1. La ruta al ejecutable sea correcta")
            print("  2. El ejecutable exista en esa ubicación")
            print("  3. Si usas rutas relativas, asegúrate de estar en el directorio correcto")
            return
        
        # Verificar permisos de ejecución (solo en Linux/Mac)
        if hasattr(os, 'access') and os.name != 'nt':
            if not os.access(env_path, os.X_OK):
                print(f"\n⚠ Advertencia: El archivo no tiene permisos de ejecución")
                print(f"  Intentando agregar permisos...")
                try:
                    os.chmod(env_path, 0o755)
                    print(f"  ✓ Permisos agregados")
                except Exception as e:
                    print(f"  ✗ No se pudieron agregar permisos: {e}")
            
            # Verificar si es un ejecutable de Windows en Linux
            try:
                import subprocess
                result = subprocess.run(['file', str(env_path)], 
                                      capture_output=True, text=True, timeout=2)
                if 'PE32' in result.stdout or 'Windows' in result.stdout:
                    print("\n" + "=" * 70)
                    print("⚠ ADVERTENCIA: Ejecutable de Windows detectado")
                    print("=" * 70)
                    print()
                    print("El archivo es un ejecutable de Windows (.exe) y estás en Linux.")
                    print("ML-Agents no puede ejecutar ejecutables de Windows directamente en Linux.")
                    print()
                    print("Opciones:")
                    print("  1. Usa Unity Editor (recomendado):")
                    print("     ./train_custom_ppo.sh")
                    print("     (Luego presiona Play en Unity Editor)")
                    print()
                    print("  2. Crea un build de Linux del juego en Unity:")
                    print("     File > Build Settings > Linux > Build")
                    print()
                    print("  3. Usa Wine (no recomendado, puede tener problemas):")
                    print("     wine \"{}\"".format(env_path))
                    print()
                    print("Continuando de todas formas, pero probablemente fallará...")
                    print()
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                # Si 'file' no está disponible o falla, continuar
                pass
        
        original_build_path = env_path
        unity_env_path = str(env_path)
    
    # Si --no-graphics está activado y tenemos un ejecutable, crear un wrapper script
    if args.no_graphics and unity_env_path:
        # Crear un script wrapper temporal que ejecute Unity con -batchmode -nographics
        script_dir = Path(__file__).parent.parent
        wrapper_dir = script_dir / ".tmp"
        wrapper_dir.mkdir(exist_ok=True)
        
        wrapper_script = wrapper_dir / "run_headless.sh"
        build_abs_path = Path(unity_env_path).resolve()
        
        # Crear el script wrapper
        wrapper_content = f"""#!/bin/bash
# Wrapper script para ejecutar Unity en modo headless
exec "{build_abs_path}" -batchmode -nographics "$@"
"""
        wrapper_script.write_text(wrapper_content)
        wrapper_script.chmod(0o755)
        
        unity_env_path = str(wrapper_script.resolve())
        print(f"Modo headless: usando wrapper script para ejecutar sin ventana")
    
    print("Configuración:")
    print(f"  Env: {unity_env_path if unity_env_path else 'Unity Editor'}")
    print(f"  Max Steps: {args.max_steps}")
    print(f"  Time Scale: {args.time_scale}")
    print(f"  Save Dir: {args.save_dir}")
    print(f"  TensorBoard Dir: {tensorboard_dir}")
    print(f"  Base Port: {args.base_port}")
    print(f"  Worker ID: {args.worker_id}")
    print()
    
    # 1. Configuración de conexión con Unity
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
        
        # Configurar Unity (resolución y time scale)
        # Si --no-graphics está activado, usar resolución mínima
        if args.no_graphics:
            width, height = 84, 84  # Resolución mínima para headless
            print("Modo headless activado (sin ventana gráfica)")
        else:
            width, height = 800, 600
        
        engine_channel.set_configuration_parameters(
            width=width,
            height=height,
            time_scale=args.time_scale
        )
        
        # Esperar a que el entorno se inicialice
        env.reset()
        
        # 2. Obtener información del comportamiento
        behavior_name = list(env.behavior_specs)[0]
        spec = env.behavior_specs[behavior_name]
        
        # Obtener dimensiones automáticamente desde Unity
        # Asumiendo que la primera observación es un Vector Sensor
        obs_dim = spec.observation_specs[0].shape[0]
        act_dim = spec.action_spec.continuous_size
        
        print(f"✓ Conectado a Unity")
        print(f"  Behavior Name: {behavior_name}")
        print(f"  Observation Dimension: {obs_dim}")
        print(f"  Action Dimension: {act_dim}")
        print()
        
        # 3. Inicializar TU agente PPO
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {device}")
        print()
        
        # ... dentro de main() en train_custom_ppo.py ...

        # config = PPOConfig(
        #     learning_rate=3e-4,      # Estándar y robusto
        #     gamma=0.99,
        #     gae_lambda=0.95,
        #     clip_range=0.2,
        #     batch_size=512,          # AUMENTADO: De 128 a 512 para estabilizar el gradiente
        #     n_steps=4096,            # AUMENTADO: Recolectar más experiencia antes de actualizar
        #     n_epochs=5,              # Mantener en 5 para no sobre-ajustar
        #     hidden_sizes=(256, 256),
        #     ent_coef=0.05,           # CRÍTICO: Subido de 0.0 a 0.05 para forzar exploración
        #     vf_coef=0.5
        # )
        # config = PPOConfig(
        #     learning_rate=0.00040912205744437856,      
        #     gamma=0.9826503833523887,
        #     gae_lambda=0.9375286789121131,
        #     clip_range=0.17327236865873835,
        #     batch_size=256,          
        #     n_steps=4096,            
        #     n_epochs=3,              
        #     hidden_sizes=(512, 256),
        #     ent_coef=0.018790490260574548,           
        #     vf_coef=0.6736932106048628,
        #     max_grad_norm=0.5200762468698006
        # )
        config = PPOConfig(
            learning_rate=0.00010750512925563073,      
            gamma=0.9920917978140118,
            gae_lambda=0.9388613588645797,
            clip_range=0.20171413823294054,
            batch_size=128,          
            n_steps=4096,            
            n_epochs=10,              
            hidden_sizes=(512, 512),
            ent_coef=0.026791305299743228,           
            vf_coef=0.7570235993959912,
            max_grad_norm=0.5696711209578253
        )
        
        agent = PPOClip(obs_dim, act_dim, config, device, discrete=False)
        
        # IMPORTANTE: Reinicia la desviación estándar para que empiece explorando mucho
        # Sobrescribimos el log_std inicial de la política
        with torch.no_grad():
            agent.policy.log_std.fill_(-0.5) # Valores más altos = más exploración inicial
            
        # 4. Bucle de Entrenamiento
        total_steps = 0
        episode_count = 0
        episode_rewards = []
        
        # Diccionario para guardar transiciones pendientes (obs, action, logp, val)
        # Se guardan aquí antes de obtener el reward del siguiente paso
        pending_transitions = {}  # {agent_id: {'obs': obs, 'action': action, 'logp': logp, 'val': val}}
        
        # Diccionario para sumar rewards paso a paso: {agent_id: current_episode_score}
        episode_scores = {}
        
        print("=" * 70)
        print("Iniciando entrenamiento...")
        print("=" * 70)
        print()
        if unity_env_path is None:
            print("NOTA: Asegúrate de que Unity Editor esté en Play mode")
        else:
            print(f"NOTA: El build de Unity se ejecutará automáticamente")
            if not args.no_graphics:
                print("      Puedes ver la simulación en la ventana de Unity")
        print()
        
        while total_steps < args.max_steps:
            # 1. Obtener datos de TODOS los agentes
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            # 2. Procesar agentes que terminaron (Terminal Steps)
            # Aquí recibimos el reward final y guardamos la transición con done=True
            for agent_id in terminal_steps.agent_id:
                term_step = terminal_steps[agent_id]
                final_step_reward = term_step.reward
                
                # Si tenemos una transición pendiente, guardarla con el reward final
                if agent_id in pending_transitions:
                    trans = pending_transitions[agent_id]
                    agent.buffer.add(
                        obs=trans['obs'],
                        act=trans['action'],
                        logp=trans['logp'],
                        rew=final_step_reward,
                        done=True,
                        val=trans['val']
                    )
                    
                    # Recuperar lo acumulado y sumar el último reward
                    total_episode_reward = episode_scores.get(agent_id, 0.0) + final_step_reward
                    
                    # Limpiar el contador para este agente (ya que renacerá con el mismo ID o uno nuevo)
                    if agent_id in episode_scores:
                        del episode_scores[agent_id]
                    
                    # AHORA SÍ guardas el total real
                    episode_rewards.append(total_episode_reward)
                    episode_count += 1
                    total_steps += 1
                    
                    # Limpiar transición pendiente
                    del pending_transitions[agent_id]
                    
                    if episode_count % 10 == 0:
                        avg_reward = np.mean(episode_rewards[-10:])
                        print(f"Episodio {episode_count}, Recompensa promedio: {avg_reward:.2f}, Pasos: {total_steps}")
                    
                    # Registrar reward en TensorBoard (solo Episode_Reward)
                    writer.add_scalar("Reward", total_episode_reward, episode_count)
            
            # 3. Procesar agentes activos que necesitan actuar (Decision Steps)
            if len(decision_steps) > 0:
                # Preparamos lista para juntar las acciones de todos los agentes
                actions_list = []
                
                # Iteramos por cada agente que pide acción
                for agent_id in decision_steps.agent_id:
                    # Obtener la obs de ESTE agente específico
                    decision_step = decision_steps[agent_id]
                    obs = decision_step.obs[0]
                    
                    # Convertir a tensor
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    
                    # Tu PPO selecciona acción (devuelve acción para 1 solo agente)
                    action, logp, val = agent.select_action(obs_tensor)
                    
                    # Guardar transición pendiente (obs, action, logp, val)
                    # El reward lo obtendremos en el siguiente paso
                    pending_transitions[agent_id] = {
                        'obs': obs,
                        'action': action.numpy(),
                        'logp': logp,
                        'val': val
                    }
                    
                    # Agregar acción a la lista
                    actions_list.append(action.numpy())
                
                # Enviar TODAS las acciones a Unity de una vez
                if actions_list:
                    actions_np = np.vstack(actions_list)  # Shape (NumAgents, act_dim)
                    action_tuple = ActionTuple(continuous=actions_np)
                    
                    # ML-Agents empareja las acciones con los agent_id de decision_steps en orden
                    env.set_actions(behavior_name, action_tuple)
            
            # 4. Avanzar simulación
            env.step()
            
            # 5. Obtener rewards del paso siguiente y guardar transiciones completas
            decision_steps_next, terminal_steps_next = env.get_steps(behavior_name)
            
            # Procesar agentes que siguen activos (obtener sus rewards)
            for agent_id in decision_steps_next.agent_id:
                if agent_id in pending_transitions:
                    decision_step_next = decision_steps_next[agent_id]
                    step_reward = decision_step_next.reward
                    
                    # ACUMULAR REWARD
                    if agent_id not in episode_scores:
                        episode_scores[agent_id] = 0.0
                    episode_scores[agent_id] += step_reward
                    
                    # Ahora tenemos todo: obs, action, logp, reward, val, done=False
                    trans = pending_transitions[agent_id]
                    agent.buffer.add(
                        obs=trans['obs'],
                        act=trans['action'],
                        logp=trans['logp'],
                        rew=step_reward,
                        done=False,
                        val=trans['val']
                    )
                    
                    total_steps += 1
                    
                    # Limpiar transición pendiente (ya guardada)
                    del pending_transitions[agent_id]
            
            # 6. Actualizar cuando el buffer esté lleno
            if agent.buffer.is_full():
                # Obtener el último valor para GAE
                # Usar el valor de los agentes activos actuales
                if len(decision_steps_next) > 0:
                    # Usar el primer agente activo para obtener el último valor
                    last_agent_id = decision_steps_next.agent_id[0]
                    last_obs = decision_steps_next[last_agent_id].obs[0]
                    last_obs_tensor = torch.tensor(last_obs, dtype=torch.float32).to(device).unsqueeze(0)
                    last_value = agent.value(last_obs_tensor).squeeze(-1).item()
                else:
                    last_value = 0.0
                
                # Calcular GAE y obtener batch
                batch = agent.buffer.get(
                    last_value=last_value,
                    gamma=config.gamma,
                    lam=config.gae_lambda
                )
                
                # Actualizar el agente
                metrics = agent.update(batch)
                
                # Registrar métricas en TensorBoard (solo Loss y Entropy)
                total_loss = metrics.get('loss_total', metrics['loss_policy'] + metrics['loss_value'])
                writer.add_scalar("Loss", total_loss, total_steps)
                writer.add_scalar("Entropy", metrics['entropy'], total_steps)
                
                print(f"Update en paso {total_steps}:")
                print(f"  Policy Loss: {metrics['loss_policy']:.4f}")
                print(f"  Value Loss: {metrics['loss_value']:.4f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")
                print()
                
                # Guardar modelo periódicamente
                if total_steps % args.save_freq == 0:
                    save_path = save_dir / f"ppo_step_{total_steps}.pt"
                    agent.save(str(save_path))
                    print(f"Modelo guardado en: {save_path}")
                    print()
        
        # Guardar modelo final
        save_path = save_dir / "ppo_final.pt"
        agent.save(str(save_path))
        print(f"Modelo final guardado en: {save_path}")
        
        # Cerrar TensorBoard writer
        writer.close()
        print(f"Logs de TensorBoard guardados en: {tensorboard_dir}")
        print(f"\nPara visualizar los resultados, ejecuta:")
        print(f"  tensorboard --logdir {tensorboard_dir}")
        
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario.")
        if 'agent' in locals():
            save_path = save_dir / "ppo_interrupted.pt"
            agent.save(str(save_path))
            print(f"Modelo guardado en: {save_path}")
        if 'writer' in locals():
            writer.close()
            print(f"Logs de TensorBoard guardados en: {tensorboard_dir}")
    except Exception as e:
        error_msg = str(e).lower()
        
        # Detectar error de ejecutable no encontrado
        if "couldn't launch" in error_msg or "does not match any environments" in error_msg or "provided filename" in error_msg:
            print("\n" + "=" * 70)
            print("✗ Error: No se pudo lanzar el ejecutable de Unity")
            print("=" * 70)
            print()
            if unity_env_path:
                print(f"Ruta proporcionada: {args.env}")
                print(f"Ruta resuelta: {unity_env_path}")
                print()
                print("Posibles causas:")
                print("  1. El ejecutable no existe en la ruta especificada")
                print("  2. El ejecutable no tiene permisos de ejecución")
                print("  3. La ruta contiene caracteres especiales o espacios mal manejados")
                print("  4. El ejecutable está corrupto o incompleto")
                print()
                print("Soluciones:")
                print("  1. Verifica que el archivo existe:")
                print(f"     ls -la \"{unity_env_path}\"")
                print("  2. Si el archivo existe, verifica permisos:")
                print(f"     chmod +x \"{unity_env_path}\"")
                print("  3. Usa una ruta absoluta:")
                print(f"     ./train_custom_ppo.sh --env \"{unity_env_path}\"")
                print("  4. O usa Unity Editor en lugar del build:")
                print("     ./train_custom_ppo.sh")
            else:
                print("No se proporcionó ruta al ejecutable.")
                print("Asegúrate de que Unity Editor esté en Play mode.")
            print()
        # Detectar error de puerto ocupado
        elif "worker number" in error_msg and "still in use" in error_msg:
            print("\n" + "=" * 70)
            print("✗ Error: Puerto ocupado")
            print("=" * 70)
            print()
            print("El puerto está siendo usado por otra instancia.")
            print()
            print("Soluciones:")
            print("  1. Cierra cualquier instancia anterior del script")
            print("  2. Cierra Unity Editor y vuelve a abrirlo")
            print("  3. Usa un puerto diferente:")
            print(f"     ./train_custom_ppo.sh --base-port 5005")
            print("  4. O usa un worker_id diferente:")
            print(f"     ./train_custom_ppo.sh --worker-id 1")
            print()
        else:
            print(f"\n✗ Error durante el entrenamiento: {e}")
            import traceback
            traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
            print("Entorno cerrado.")
        if 'writer' in locals():
            writer.close()


if __name__ == "__main__":
    main()

