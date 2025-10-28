// /// Author: Assistant
// /// Date: Oct 2025

// #region Includes
// using System;
// using System.Net;
// using System.Net.Sockets;
// using System.Text;
// using System.Threading;
// using System.Collections.Concurrent;
// using UnityEngine;
// #endregion

// /// <summary>
// /// TCP bridge to control a single car from an external Python PPO process.
// /// Protocol: JSON per line over TCP. Messages:
// ///  - {"cmd":"reset"}
// ///  - {"cmd":"step","action":[turn, throttle]}
// /// Response to both includes: {"obs":[...],"reward":r,"done":b}
// /// </summary>
// public class RLBridgeServer : MonoBehaviour
// {
//     #region Members
//     [SerializeField]
//     private int port = 5555;

//     private Thread listenerThread;
//     private TcpListener listener;
//     private volatile bool running = false;

//     private AutoResetEvent stepRequested = new AutoResetEvent(false);
//     private AutoResetEvent stepCompleted = new AutoResetEvent(false);

//     private double[] pendingAction = null; // [turn, throttle]
//     private string stepResponseJson = null;

//     private CarController controlledCar;
//     private Sensor[] sensors;

//     private float prevCompletion = 0f;
//     private float episodeReward = 0f;

//     private volatile bool resetRequested = false;       // NEWWW
//     private string resetResponseJson = null;            // NEWWW

//     #endregion

//     #region Constructors
//     void Awake()
//     {
//         // Create or find a single car to control
//         if (TrackManager.Instance == null)
//         {
//             Debug.LogError("RLBridgeServer requires TrackManager in scene.");
//             return;
//         }

//         // Ensure exactly one controllable car
//         TrackManager.Instance.SetCarAmount(1);
//         var cars = TrackManager.Instance.GetCarEnumerator();
//         if (cars.MoveNext())
//         {
//             controlledCar = cars.Current;
//             controlledCar.UseUserInput = false;
//             controlledCar.UseExternalControl = true;
//         }

//         sensors = controlledCar.GetComponentsInChildren<Sensor>();
//     }

//     void Start()
//     {
//         StartServer();
//     }

//     void OnDestroy()
//     {
//         StopServer();
//     }
//     #endregion

//     #region Methods
//     private void StartServer()
//     {
//         if (running) return;
//         running = true;
//         listener = new TcpListener(IPAddress.Loopback, port);
//         listener.Start();
//         listenerThread = new Thread(ListenLoop);
//         listenerThread.IsBackground = true;
//         listenerThread.Start();
//         Debug.Log($"RLBridgeServer listening on 127.0.0.1:{port}");
//     }

//     private void StopServer()
//     {
//         running = false;
//         try { listener?.Stop(); } catch { }
//         try { listenerThread?.Join(200); } catch { }
//     }

//     private void ListenLoop()
//     {
//         while (running)
//         {
//             try
//             {
//                 using (TcpClient client = listener.AcceptTcpClient())
//                 using (var stream = client.GetStream())
//                 {
//                     stream.ReadTimeout = 60000;
//                     stream.WriteTimeout = 60000;
//                     var reader = new System.IO.StreamReader(stream, Encoding.UTF8);
//                     var writer = new System.IO.StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };

//                     // Session loop
//                     while (running && client.Connected)
//                     {
//                         string line = reader.ReadLine();
//                         if (line == null) break;
//                         var cmd = ParseCommand(line);
//                         if (cmd == null)
//                         {
//                             writer.WriteLine("{\"error\":\"bad_json\"}");
//                             continue;
//                         }

//                         // if (cmd.Name == "reset")
//                         // {
//                         //     string resp = UnityThread_Reset();
//                         //     writer.WriteLine(resp);
//                         // }
//                         if (cmd.Name == "reset")                                                                  // NEWWW                
//                         {
//                             resetRequested = true;
//                             // Espera a que FixedUpdate procese el reset
//                             int waitMs = 5000;
//                             int waited = 0;
//                             while (resetResponseJson == null && waited < waitMs)
//                             {
//                                 Thread.Sleep(50);
//                                 waited += 50;
//                             }
//                             string resp = resetResponseJson ?? "{\"error\":\"timeout\"}";
//                             resetResponseJson = null;
//                             writer.WriteLine(resp);
//                         }

//                         else if (cmd.Name == "step")
//                         {
//                             pendingAction = cmd.Action;
//                             // signal step and wait for completion
//                             stepRequested.Set();
//                             stepCompleted.WaitOne(5000);
//                             string resp = stepResponseJson ?? "{\"error\":\"timeout\"}";
//                             stepResponseJson = null;
//                             writer.WriteLine(resp);
//                         }
//                         else if (cmd.Name == "close")
//                         {
//                             writer.WriteLine("{\"ok\":true}");
//                             break;
//                         }
//                         else
//                         {
//                             writer.WriteLine("{\"error\":\"unknown_cmd\"}");
//                         }
//                     }
//                 }
//             }
//             catch (SocketException)
//             {
//                 if (!running) break;
//             }
//             catch (Exception ex)
//             {
//                 Debug.LogError($"RLBridgeServer exception: {ex}");
//             }
//         }
//     }

//     // Called from Unity thread once per physics step
//     void FixedUpdate()
//     {
//         if (controlledCar == null) return;

//         // If a step was requested, apply action for this physics tick and capture results
//         if (stepRequested.WaitOne(0))
//         {
//             // Apply action
//             double turn = 0, throttle = 0;
//             if (pendingAction != null && pendingAction.Length >= 2)
//             {
//                 turn = Mathf.Clamp((float)pendingAction[0], -1f, 1f);
//                 throttle = Mathf.Clamp((float)pendingAction[1], -1f, 1f);
//             }
//             controlledCar.Movement.SetInputs(new double[] { turn, throttle });

//             // Compute observation
//             float[] obs = BuildObservation();

//             // Reward: delta completion
//             float completion = controlledCar.CurrentCompletionReward;
//             float reward = completion - prevCompletion;
//             prevCompletion = completion;
//             bool done = !controlledCar.enabled;

//             // Package response
//             stepResponseJson = BuildResponseJson(obs, reward, done);
//             episodeReward += reward;

//             if (done)
//             {
//                 // keep car stopped until reset
//             }

//             stepCompleted.Set();
//         }
//         // if (resetRequested)                                                             // NEWWW    
//         // {
//         //     resetRequested = false;

//         //     // Esto ahora corre en el Main Thread
//         //     TrackManager.Instance.Restart();
//         //     prevCompletion = 0f;
//         //     episodeReward = 0f;

//         //     var cars = TrackManager.Instance.GetCarEnumerator();
//         //     if (cars.MoveNext())
//         //     {
//         //         controlledCar = cars.Current;
//         //         controlledCar.UseUserInput = false;
//         //         controlledCar.UseExternalControl = true;
//         //         sensors = controlledCar.GetComponentsInChildren<Sensor>();
//         //     }

//         //     float[] obs = BuildObservation();
//         //     resetResponseJson = BuildResponseJson(obs, 0f, false);
//         // }

//     }

//     private string UnityThread_Reset()
//     {
//         // Reset environment
//         TrackManager.Instance.Restart();
//         prevCompletion = 0f;
//         episodeReward = 0f;
//         // Refresh reference (car recreated on restart)
//         var cars = TrackManager.Instance.GetCarEnumerator();
//         if (cars.MoveNext())
//         {
//             controlledCar = cars.Current;
//             controlledCar.UseUserInput = false;
//             controlledCar.UseExternalControl = true;
//             sensors = controlledCar.GetComponentsInChildren<Sensor>();
//         }
//         float[] obs = BuildObservation();
//         return BuildResponseJson(obs, 0f, false);
//     }

//     private float[] BuildObservation()
//     {
//         if (sensors == null) sensors = controlledCar.GetComponentsInChildren<Sensor>();
//         float[] obs = new float[sensors.Length + 1]; // sensors + velocity
//         for (int i = 0; i < sensors.Length; i++)
//             obs[i] = sensors[i].Output;
//         obs[sensors.Length] = controlledCar.Movement.Velocity;
//         return obs;
//     }

//     private string BuildResponseJson(float[] obs, float reward, bool done)
//     {
//         // Simple manual JSON builder to avoid allocations
//         StringBuilder sb = new StringBuilder();
//         sb.Append("{\"obs\":[");
//         for (int i = 0; i < obs.Length; i++)
//         {
//             if (i > 0) sb.Append(',');
//             sb.Append(obs[i].ToString(System.Globalization.CultureInfo.InvariantCulture));
//         }
//         sb.Append("],\"reward\":");
//         sb.Append(reward.ToString(System.Globalization.CultureInfo.InvariantCulture));
//         sb.Append(",\"done\":");
//         sb.Append(done ? "true" : "false");
//         sb.Append('}');
//         return sb.ToString();
//     }

//     private class Command
//     {
//         public string Name;
//         public double[] Action;
//     }

//     private Command ParseCommand(string json)
//     {
//         try
//         {
//             // minimal parse without JSON lib
//             if (json.Contains("\"reset\"")) return new Command { Name = "reset" };
//             if (json.Contains("\"close\"")) return new Command { Name = "close" };
//             if (json.Contains("\"step\""))
//             {
//                 // Expect action array: "action":[a,b]
//                 int idx = json.IndexOf("[", StringComparison.Ordinal);
//                 int jdx = json.IndexOf("]", idx + 1, StringComparison.Ordinal);
//                 if (idx >= 0 && jdx > idx)
//                 {
//                     string arr = json.Substring(idx + 1, jdx - idx - 1);
//                     string[] parts = arr.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
//                     double[] act = new double[Math.Min(2, parts.Length)];
//                     for (int i = 0; i < act.Length; i++)
//                         act[i] = double.Parse(parts[i], System.Globalization.CultureInfo.InvariantCulture);
//                     return new Command { Name = "step", Action = act };
//                 }
//                 return new Command { Name = "step", Action = new double[] { 0, 0 } };
//             }
//         }
//         catch { }
//         return null;
//     }
//     #endregion
// }






























// #region Includes
// using System;
// using System.Net;
// using System.Net.Sockets;
// using System.Text;
// using System.Threading;
// using UnityEngine;
// #endregion

// /// <summary>
// /// TCP bridge to control a single car from an external Python PPO process.
// /// Protocol: JSON per line over TCP. Messages:
// ///  - {"cmd":"reset"}
// ///  - {"cmd":"step","action":[turn, throttle]}
// /// Response to both includes: {"obs":[...],"reward":r,"done":b}
// /// </summary>
// public class RLBridgeServer : MonoBehaviour
// {
//     #region Members
//     [SerializeField]
//     private int port = 5555;

//     [SerializeField]
//     private CarController controlledCar; // asignar desde Inspector o buscar en escena

//     [SerializeField]
//     private CarController carPrefab;

//     private Sensor[] sensors;

//     private Thread listenerThread;
//     private TcpListener listener;
//     private volatile bool running = false;

//     private AutoResetEvent stepRequested = new AutoResetEvent(false);
//     private AutoResetEvent stepCompleted = new AutoResetEvent(false);

//     private double[] pendingAction = null; // [turn, throttle]
//     private string stepResponseJson = null;

//     private float prevCompletion = 0f;
//     private float episodeReward = 0f;

//     #endregion

//     #region Unity Methods
//     // void Awake()
//     // {
//     //     // Si no se asignó el auto desde Inspector, buscar uno en la escena
//     //     if (controlledCar == null)
//     //     {
//     //         controlledCar = FindObjectOfType<CarController>();
//     //         if (controlledCar == null)
//     //         {
//     //             Debug.LogError("RLBridgeServer: No CarController found in scene!");
//     //             return;
//     //         }
//     //     }

//     //     controlledCar.UseUserInput = false;
//     //     controlledCar.UseExternalControl = true;

//     //     sensors = controlledCar.GetComponentsInChildren<Sensor>();
//     // }
//     // void Awake()
//     // {
//     //     if (controlledCar == null)
//     //     {
//     //         if (carPrefab != null)
//     //         {
//     //             controlledCar = Instantiate(carPrefab);
//     //         }
//     //         else
//     //         {
//     //             Debug.LogError("RLBridgeServer: No CarController prefab assigned!");
//     //             return;
//     //         }
//     //     }

//     //     controlledCar.UseUserInput = false;
//     //     controlledCar.UseExternalControl = true;
//     //     sensors = controlledCar.GetComponentsInChildren<Sensor>();
//     // }
//     void Awake()
//     {
//         // 1. Usar el auto asignado en el Inspector
//         if (controlledCar == null)
//         {
//             // 2. Instanciar prefab si está asignado
//             if (carPrefab != null)
//             {
//                 controlledCar = Instantiate(carPrefab);
//                 controlledCar.name = carPrefab.name; // opcional, limpia el nombre
//             }
//             // 3. Buscar cualquier CarController en la escena
//             else
//             {
//                 controlledCar = FindObjectOfType<CarController>();
//                 if (controlledCar == null)
//                 {
//                     Debug.LogError("RLBridgeServer: No CarController found in scene or prefab assigned!");
//                     return;
//                 }
//                 else
//                 {
//                     Debug.LogWarning("RLBridgeServer: Using existing CarController in scene as fallback.");
//                 }
//             }
//         }

//         // Configurar control externo
//         controlledCar.UseUserInput = false;
//         controlledCar.UseExternalControl = true;

//         // Obtener sensores
//         sensors = controlledCar.GetComponentsInChildren<Sensor>();
//     }


//     void Start()
//     {
//         StartServer();
//     }

//     void OnDestroy()
//     {
//         StopServer();
//     }

//     void FixedUpdate()
//     {
//         if (controlledCar == null) return;

//         // Procesar acción si fue solicitada
//         if (stepRequested.WaitOne(0))
//         {
//             double turn = 0, throttle = 0;
//             if (pendingAction != null && pendingAction.Length >= 2)
//             {
//                 turn = Mathf.Clamp((float)pendingAction[0], -1f, 1f);
//                 throttle = Mathf.Clamp((float)pendingAction[1], -1f, 1f);
//             }

//             controlledCar.Movement.SetInputs(new double[] { turn, throttle });

//             float[] obs = BuildObservation();

//             float completion = controlledCar.CurrentCompletionReward;
//             float reward = completion - prevCompletion;
//             prevCompletion = completion;
//             bool done = !controlledCar.enabled;

//             stepResponseJson = BuildResponseJson(obs, reward, done);
//             episodeReward += reward;

//             stepCompleted.Set();
//         }
//     }
//     #endregion

//     #region Server Methods
//     private void StartServer()
//     {
//         if (running) return;
//         running = true;
//         listener = new TcpListener(IPAddress.Loopback, port);
//         listener.Start();
//         listenerThread = new Thread(ListenLoop);
//         listenerThread.IsBackground = true;
//         listenerThread.Start();
//         Debug.Log($"RLBridgeServer listening on 127.0.0.1:{port}");
//     }

//     private void StopServer()
//     {
//         running = false;
//         try { listener?.Stop(); } catch { }
//         try { listenerThread?.Join(200); } catch { }
//     }

//     private void ListenLoop()
//     {
//         while (running)
//         {
//             try
//             {
//                 using (TcpClient client = listener.AcceptTcpClient())
//                 using (var stream = client.GetStream())
//                 {
//                     stream.ReadTimeout = 60000;
//                     stream.WriteTimeout = 60000;
//                     var reader = new System.IO.StreamReader(stream, Encoding.UTF8);
//                     var writer = new System.IO.StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };

//                     while (running && client.Connected)
//                     {
//                         string line = reader.ReadLine();
//                         if (line == null) break;

//                         var cmd = ParseCommand(line);
//                         if (cmd == null)
//                         {
//                             writer.WriteLine("{\"error\":\"bad_json\"}");
//                             continue;
//                         }

//                         if (cmd.Name == "reset")
//                         {
//                             string resp = UnityThread_Reset();
//                             writer.WriteLine(resp);
//                         }
//                         else if (cmd.Name == "step")
//                         {
//                             pendingAction = cmd.Action;
//                             stepRequested.Set();
//                             stepCompleted.WaitOne(5000);
//                             string resp = stepResponseJson ?? "{\"error\":\"timeout\"}";
//                             stepResponseJson = null;
//                             writer.WriteLine(resp);
//                         }
//                         else if (cmd.Name == "close")
//                         {
//                             writer.WriteLine("{\"ok\":true}");
//                             break;
//                         }
//                         else
//                         {
//                             writer.WriteLine("{\"error\":\"unknown_cmd\"}");
//                         }
//                     }
//                 }
//             }
//             catch (SocketException)
//             {
//                 if (!running) break;
//             }
//             catch (Exception ex)
//             {
//                 Debug.LogError($"RLBridgeServer exception: {ex}");
//             }
//         }
//     }
//     #endregion

//     #region Helper Methods
//     private string UnityThread_Reset()
//     {
//         if (controlledCar != null)
//         {
//             controlledCar.transform.position = Vector3.zero; // posición inicial
//             controlledCar.transform.rotation = Quaternion.identity;
//             controlledCar.Restart();
//         }

//         prevCompletion = 0f;
//         episodeReward = 0f;

//         float[] obs = BuildObservation();
//         return BuildResponseJson(obs, 0f, false);
//     }

//     private float[] BuildObservation()
//     {
//         if (sensors == null) sensors = controlledCar.GetComponentsInChildren<Sensor>();
//         float[] obs = new float[sensors.Length + 1]; // sensores + velocidad
//         for (int i = 0; i < sensors.Length; i++)
//             obs[i] = sensors[i].Output;
//         obs[sensors.Length] = controlledCar.Movement.Velocity;
//         return obs;
//     }

//     private string BuildResponseJson(float[] obs, float reward, bool done)
//     {
//         StringBuilder sb = new StringBuilder();
//         sb.Append("{\"obs\":[");
//         for (int i = 0; i < obs.Length; i++)
//         {
//             if (i > 0) sb.Append(',');
//             sb.Append(obs[i].ToString(System.Globalization.CultureInfo.InvariantCulture));
//         }
//         sb.Append("],\"reward\":");
//         sb.Append(reward.ToString(System.Globalization.CultureInfo.InvariantCulture));
//         sb.Append(",\"done\":");
//         sb.Append(done ? "true" : "false");
//         sb.Append('}');
//         return sb.ToString();
//     }

//     private class Command
//     {
//         public string Name;
//         public double[] Action;
//     }

//     private Command ParseCommand(string json)
//     {
//         try
//         {
//             if (json.Contains("\"reset\"")) return new Command { Name = "reset" };
//             if (json.Contains("\"close\"")) return new Command { Name = "close" };
//             if (json.Contains("\"step\""))
//             {
//                 int idx = json.IndexOf("[", StringComparison.Ordinal);
//                 int jdx = json.IndexOf("]", idx + 1, StringComparison.Ordinal);
//                 if (idx >= 0 && jdx > idx)
//                 {
//                     string arr = json.Substring(idx + 1, jdx - idx - 1);
//                     string[] parts = arr.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
//                     double[] act = new double[Math.Min(2, parts.Length)];
//                     for (int i = 0; i < act.Length; i++)
//                         act[i] = double.Parse(parts[i], System.Globalization.CultureInfo.InvariantCulture);
//                     return new Command { Name = "step", Action = act };
//                 }
//                 return new Command { Name = "step", Action = new double[] { 0, 0 } };
//             }
//         }
//         catch { }
//         return null;
//     }
//     #endregion
// }













// MAS O MENOS



// #region Includes
// using System;
// using System.Net;
// using System.Net.Sockets;
// using System.Text;
// using System.Threading;
// using UnityEngine;
// #endregion

// /// <summary>
// /// TCP bridge to control a single car from an external Python PPO process.
// /// Protocol: JSON per line over TCP. Messages:
// ///  - {"cmd":"reset"}
// ///  - {"cmd":"step","action":[turn, throttle]}
// /// Response to both includes: {"obs":[...],"reward":r,"done":b}
// /// </summary>
// public class RLBridgeServer : MonoBehaviour
// {
//     #region Members
//     [SerializeField]
//     private int port = 5555;

//     [SerializeField]
//     private CarController controlledCar; // asignar desde Inspector o buscar en escena

//     [SerializeField]
//     private CarController carPrefab;

//     private Sensor[] sensors;

//     private Thread listenerThread;
//     private TcpListener listener;
//     private volatile bool running = false;

//     private AutoResetEvent stepRequested = new AutoResetEvent(false);
//     private AutoResetEvent stepCompleted = new AutoResetEvent(false);

//     private double[] pendingAction = null; // [turn, throttle]
//     private string stepResponseJson = null;

//     private float prevCompletion = 0f;
//     private float episodeReward = 0f;

//     // Flags para reset seguro en main thread
//     private volatile bool resetRequested = false;
//     private string resetResponseJson = null;

//     private AutoResetEvent resetRequestedEvent = new AutoResetEvent(false);


//     #endregion

//     #region Unity Methods

//     void Awake()
//     {
//         // 1. Usar el auto asignado en el Inspector
//         if (controlledCar == null)
//         {
//             // 2. Instanciar prefab si está asignado
//             if (carPrefab != null)
//             {
//                 controlledCar = Instantiate(carPrefab);
//                 controlledCar.name = carPrefab.name; // opcional
//             }
//             // 3. Buscar cualquier CarController en la escena
//             else
//             {
//                 controlledCar = FindObjectOfType<CarController>();
//                 if (controlledCar == null)
//                 {
//                     Debug.LogError("RLBridgeServer: No CarController found in scene or prefab assigned!");
//                     return;
//                 }
//                 else
//                 {
//                     Debug.LogWarning("RLBridgeServer: Using existing CarController in scene as fallback.");
//                 }
//             }
//         }

//         // Configurar control externo
//         controlledCar.UseUserInput = false;
//         controlledCar.UseExternalControl = true;

//         // Obtener sensores
//         sensors = controlledCar.GetComponentsInChildren<Sensor>();
//     }


//     void Start()
//     {
//         StartServer();
//     }

//     void OnDestroy()
//     {
//         StopServer();
//     }

//     void FixedUpdate()
//     {
//         if (controlledCar == null) return;

//         // --- Procesar acción si fue solicitada ---
//         if (stepRequested.WaitOne(0))
//         {
//             double turn = 0, throttle = 0;
//             if (pendingAction != null && pendingAction.Length >= 2)
//             {
//                 turn = Mathf.Clamp((float)pendingAction[0], -1f, 1f);
//                 throttle = Mathf.Clamp((float)pendingAction[1], -1f, 1f);
//             }

//             controlledCar.Movement.SetInputs(new double[] { turn, throttle });

//             float[] obs = BuildObservation();

//             float completion = controlledCar.CurrentCompletionReward;
//             float reward = completion - prevCompletion;
//             prevCompletion = completion;
//             bool done = !controlledCar.enabled;

//             stepResponseJson = BuildResponseJson(obs, reward, done);
//             episodeReward += reward;

//             stepCompleted.Set();
//         }

//         // --- Procesar reset si fue solicitado ---
//         if (resetRequested)
//         {
//             resetRequested = false;

//             if (controlledCar != null)
//             {
//                 controlledCar.transform.position = Vector3.zero;
//                 controlledCar.transform.rotation = Quaternion.identity;
//                 controlledCar.Restart();
//             }

//             prevCompletion = 0f;
//             episodeReward = 0f;

//             float[] obs = BuildObservation();
//             resetResponseJson = BuildResponseJson(obs, 0f, false);

//             // Señalizamos al hilo del servidor que terminó el reset
//             stepCompleted.Set();
//         }
//         // Procesar reset si fue solicitado
//         if (resetRequestedEvent.WaitOne(0))
//         {
//             if (controlledCar != null)
//             {
//                 controlledCar.transform.position = Vector3.zero;
//                 controlledCar.transform.rotation = Quaternion.identity;
//                 controlledCar.Restart();
//             }

//             prevCompletion = 0f;
//             episodeReward = 0f;

//             float[] obs = BuildObservation();
//             resetResponseJson = BuildResponseJson(obs, 0f, false);

//             // Señalizamos al hilo del servidor que terminó el reset
//             stepCompleted.Set();
//         }

//     }
//     #endregion

//     #region Server Methods
//     private void StartServer()
//     {
//         if (running) return;
//         running = true;
//         listener = new TcpListener(IPAddress.Loopback, port);
//         listener.Start();
//         listenerThread = new Thread(ListenLoop);
//         listenerThread.IsBackground = true;
//         listenerThread.Start();
//         Debug.Log($"RLBridgeServer listening on 127.0.0.1:{port}");
//     }

//     private void StopServer()
//     {
//         running = false;
//         try { listener?.Stop(); } catch { }
//         try { listenerThread?.Join(200); } catch { }
//     }

//     private void ListenLoop()
//     {
//         while (running)
//         {
//             try
//             {
//                 using (TcpClient client = listener.AcceptTcpClient())
//                 using (var stream = client.GetStream())
//                 {
//                     stream.ReadTimeout = 60000;
//                     stream.WriteTimeout = 60000;
//                     var reader = new System.IO.StreamReader(stream, Encoding.UTF8);
//                     var writer = new System.IO.StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };

//                     while (running && client.Connected)
//                     {
//                         string line = reader.ReadLine();
//                         if (line == null) break;

//                         var cmd = ParseCommand(line);
//                         if (cmd == null)
//                         {
//                             writer.WriteLine("{\"error\":\"bad_json\"}");
//                             continue;
//                         }

//                         // if (cmd.Name == "reset")
//                         // {
//                         //     // Solo pedimos que se haga reset, no ejecutamos transform desde el thread
//                         //     resetRequested = true;
//                         //     int waited = 0;
//                         //     int waitMs = 5000;
//                         //     while (resetResponseJson == null && waited < waitMs)
//                         //     {
//                         //         Thread.Sleep(50);
//                         //         waited += 50;
//                         //     }
//                         //     string resp = resetResponseJson ?? "{\"error\":\"timeout\"}";
//                         //     resetResponseJson = null;
//                         //     writer.WriteLine(resp);
//                         // }
//                         if (cmd.Name == "reset")
//                         {
//                             resetResponseJson = null;
//                             resetRequestedEvent.Set();             // señal para FixedUpdate
//                             stepCompleted.WaitOne(5000);           // espera que termine
//                             string resp = resetResponseJson ?? "{\"error\":\"timeout\"}";
//                             resetResponseJson = null;
//                             writer.WriteLine(resp);
//                         }
//                         else if (cmd.Name == "step")
//                         {
//                             pendingAction = cmd.Action;
//                             stepRequested.Set();
//                             stepCompleted.WaitOne(5000);
//                             string resp = stepResponseJson ?? "{\"error\":\"timeout\"}";
//                             stepResponseJson = null;
//                             writer.WriteLine(resp);
//                         }
//                         else if (cmd.Name == "close")
//                         {
//                             writer.WriteLine("{\"ok\":true}");
//                             break;
//                         }
//                         else
//                         {
//                             writer.WriteLine("{\"error\":\"unknown_cmd\"}");
//                         }
//                     }
//                 }
//             }
//             catch (SocketException)
//             {
//                 if (!running) break;
//             }
//             catch (Exception ex)
//             {
//                 Debug.LogError($"RLBridgeServer exception: {ex}");
//             }
//         }
//     }
//     #endregion

//     #region Helper Methods
//     private float[] BuildObservation()
//     {
//         if (sensors == null) sensors = controlledCar.GetComponentsInChildren<Sensor>();
//         float[] obs = new float[sensors.Length + 1]; // sensores + velocidad
//         for (int i = 0; i < sensors.Length; i++)
//             obs[i] = sensors[i].Output;
//         obs[sensors.Length] = controlledCar.Movement.Velocity;
//         return obs;
//     }

//     private string BuildResponseJson(float[] obs, float reward, bool done)
//     {
//         StringBuilder sb = new StringBuilder();
//         sb.Append("{\"obs\":[");
//         for (int i = 0; i < obs.Length; i++)
//         {
//             if (i > 0) sb.Append(',');
//             sb.Append(obs[i].ToString(System.Globalization.CultureInfo.InvariantCulture));
//         }
//         sb.Append("],\"reward\":");
//         sb.Append(reward.ToString(System.Globalization.CultureInfo.InvariantCulture));
//         sb.Append(",\"done\":");
//         sb.Append(done ? "true" : "false");
//         sb.Append('}');
//         return sb.ToString();
//     }

//     private class Command
//     {
//         public string Name;
//         public double[] Action;
//     }

//     private Command ParseCommand(string json)
//     {
//         try
//         {
//             if (json.Contains("\"reset\"")) return new Command { Name = "reset" };
//             if (json.Contains("\"close\"")) return new Command { Name = "close" };
//             if (json.Contains("\"step\""))
//             {
//                 int idx = json.IndexOf("[", StringComparison.Ordinal);
//                 int jdx = json.IndexOf("]", idx + 1, StringComparison.Ordinal);
//                 if (idx >= 0 && jdx > idx)
//                 {
//                     string arr = json.Substring(idx + 1, jdx - idx - 1);
//                     string[] parts = arr.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
//                     double[] act = new double[Math.Min(2, parts.Length)];
//                     for (int i = 0; i < act.Length; i++)
//                         act[i] = double.Parse(parts[i], System.Globalization.CultureInfo.InvariantCulture);
//                     return new Command { Name = "step", Action = act };
//                 }
//                 return new Command { Name = "step", Action = new double[] { 0, 0 } };
//             }
//         }
//         catch { }
//         return null;
//     }
//     #endregion
// }























#region Includes
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
#endregion

/// <summary>
/// TCP bridge to control a single car from an external Python PPO process.
/// Protocol: JSON per line over TCP. Messages:
///  - {"cmd":"reset"}
///  - {"cmd":"step","action":[turn, throttle]}
/// Response to both includes: {"obs":[...],"reward":r,"done":b}
/// </summary>
public class RLBridgeServer : MonoBehaviour
{
    #region Members
    [SerializeField]
    private int port = 5555;

    [SerializeField]
    private CarController controlledCar; // asignar desde Inspector o buscar en escena

    [SerializeField]
    private CarController carPrefab;

    private Sensor[] sensors;

    private Thread listenerThread;
    private TcpListener listener;
    private volatile bool running = false;

    private AutoResetEvent stepRequested = new AutoResetEvent(false);
    private AutoResetEvent stepCompleted = new AutoResetEvent(false);

    private AutoResetEvent resetRequestedEvent = new AutoResetEvent(false);
    private AutoResetEvent resetCompleted = new AutoResetEvent(false);
    private string resetResponseJson = null;

    private double[] pendingAction = null; // [turn, throttle]
    private string stepResponseJson = null;

    private float prevCompletion = 0f;
    private float episodeReward = 0f;

    // Flags para reset seguro en main thread
    private volatile bool resetRequested = false;

    #endregion

    #region Unity Methods

    void Awake()
    {
        // 1. Usar el auto asignado en el Inspector
        if (controlledCar == null)
        {
            // 2. Instanciar prefab si está asignado
            if (carPrefab != null)
            {
                controlledCar = Instantiate(carPrefab);
                controlledCar.name = carPrefab.name; // opcional
            }
            // 3. Buscar cualquier CarController en la escena
            else
            {
                controlledCar = FindObjectOfType<CarController>();
                if (controlledCar == null)
                {
                    Debug.LogError("RLBridgeServer: No CarController found in scene or prefab assigned!");
                    return;
                }
                else
                {
                    Debug.LogWarning("RLBridgeServer: Using existing CarController in scene as fallback.");
                }
            }
        }

        // Configurar control externo
        controlledCar.UseUserInput = false;
        controlledCar.UseExternalControl = true;

        // Obtener sensores
        sensors = controlledCar.GetComponentsInChildren<Sensor>();
    }


    void Start()
    {
        StartServer();
    }

    void OnDestroy()
    {
        StopServer();
    }

    void FixedUpdate()
    {
        if (controlledCar == null) return;

        // --- Procesar acción si fue solicitada ---
        if (stepRequested.WaitOne(0))
        {
            double turn = 0, throttle = 0;
            if (pendingAction != null && pendingAction.Length >= 2)
            {
                turn = Mathf.Clamp((float)pendingAction[0], -1f, 1f);
                throttle = Mathf.Clamp((float)pendingAction[1], -1f, 1f);
            }

            controlledCar.Movement.SetInputs(new double[] { turn, throttle });

            float[] obs = BuildObservation();

            float completion = controlledCar.CurrentCompletionReward;
            float reward = completion - prevCompletion;
            prevCompletion = completion;
            bool done = !controlledCar.enabled;

            stepResponseJson = BuildResponseJson(obs, reward, done);
            episodeReward += reward;

            stepCompleted.Set();
        }

        // --- Procesar reset si fue solicitado ---
        if (resetRequested)
        {
            resetRequested = false;

            if (controlledCar != null)
            {
                controlledCar.transform.position = Vector3.zero;
                controlledCar.transform.rotation = Quaternion.identity;
                controlledCar.Restart();
            }

            prevCompletion = 0f;
            episodeReward = 0f;

            float[] obs = BuildObservation();
            resetResponseJson = BuildResponseJson(obs, 0f, false);

            // Señalizamos al hilo del servidor que terminó el reset
            stepCompleted.Set();
        }
        // Procesar reset si fue solicitado
        // if (resetRequestedEvent.WaitOne(0))
        // {
        //     if (controlledCar != null)
        //     {
        //         controlledCar.transform.position = Vector3.zero;
        //         controlledCar.transform.rotation = Quaternion.identity;
        //         controlledCar.Restart();
        //     }

        //     prevCompletion = 0f;
        //     episodeReward = 0f;

        //     float[] obs = BuildObservation();
        //     resetResponseJson = BuildResponseJson(obs, 0f, false);

        //     // Señalizamos al hilo del servidor que terminó el reset
        //     stepCompleted.Set();
        // }
        if (resetRequestedEvent.WaitOne(0))
        {
            if (controlledCar != null)
            {
                controlledCar.transform.position = Vector3.zero;
                controlledCar.transform.rotation = Quaternion.identity;
                controlledCar.Restart();
            }

            prevCompletion = 0f;
            episodeReward = 0f;

            float[] obs = BuildObservation();
            resetResponseJson = BuildResponseJson(obs, 0f, false);

            // notificar que el reset terminó
            resetCompleted.Set();
        }

    }
    #endregion

    #region Server Methods
    private void StartServer()
    {
        if (running) return;
        running = true;
        listener = new TcpListener(IPAddress.Loopback, port);
        listener.Start();
        listenerThread = new Thread(ListenLoop);
        listenerThread.IsBackground = true;
        listenerThread.Start();
        Debug.Log($"RLBridgeServer listening on 127.0.0.1:{port}");
    }

    private void StopServer()
    {
        running = false;
        try { listener?.Stop(); } catch { }
        try { listenerThread?.Join(200); } catch { }
    }

    private void ListenLoop()
    {
        while (running)
        {
            try
            {
                using (TcpClient client = listener.AcceptTcpClient())
                using (var stream = client.GetStream())
                {
                    stream.ReadTimeout = 60000;
                    stream.WriteTimeout = 60000;
                    var reader = new System.IO.StreamReader(stream, Encoding.UTF8);
                    var writer = new System.IO.StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };

                    while (running && client.Connected)
                    {
                        string line = reader.ReadLine();
                        if (line == null) break;

                        var cmd = ParseCommand(line);
                        if (cmd == null)
                        {
                            writer.WriteLine("{\"error\":\"bad_json\"}");
                            continue;
                        }

                        // if (cmd.Name == "reset")
                        // {
                        //     // Solo pedimos que se haga reset, no ejecutamos transform desde el thread
                        //     resetRequested = true;
                        //     int waited = 0;
                        //     int waitMs = 5000;
                        //     while (resetResponseJson == null && waited < waitMs)
                        //     {
                        //         Thread.Sleep(50);
                        //         waited += 50;
                        //     }
                        //     string resp = resetResponseJson ?? "{\"error\":\"timeout\"}";
                        //     resetResponseJson = null;
                        //     writer.WriteLine(resp);
                        // }
                        // if (cmd.Name == "reset")
                        // {
                        //     resetResponseJson = null;
                        //     resetRequestedEvent.Set();             // señal para FixedUpdate
                        //     stepCompleted.WaitOne(5000);           // espera que termine
                        //     string resp = resetResponseJson ?? "{\"error\":\"timeout\"}";
                        //     resetResponseJson = null;
                        //     writer.WriteLine(resp);
                        // }
                        if (cmd.Name == "reset")
                        {
                            resetResponseJson = null;
                            resetRequestedEvent.Set();            // disparar reset
                            resetCompleted.WaitOne(5000);         // esperar que termine
                            string resp = resetResponseJson ?? "{\"error\":\"timeout\"}";
                            resetResponseJson = null;
                            writer.WriteLine(resp);
                        }

                        else if (cmd.Name == "step")
                        {
                            pendingAction = cmd.Action;
                            stepRequested.Set();
                            stepCompleted.WaitOne(5000);
                            string resp = stepResponseJson ?? "{\"error\":\"timeout\"}";
                            stepResponseJson = null;
                            writer.WriteLine(resp);
                        }
                        else if (cmd.Name == "close")
                        {
                            writer.WriteLine("{\"ok\":true}");
                            break;
                        }
                        else
                        {
                            writer.WriteLine("{\"error\":\"unknown_cmd\"}");
                        }
                    }
                }
            }
            catch (SocketException)
            {
                if (!running) break;
            }
            catch (Exception ex)
            {
                Debug.LogError($"RLBridgeServer exception: {ex}");
            }
        }
    }
    #endregion

    #region Helper Methods
    private float[] BuildObservation()
    {
        if (sensors == null) sensors = controlledCar.GetComponentsInChildren<Sensor>();
        float[] obs = new float[sensors.Length + 1]; // sensores + velocidad
        for (int i = 0; i < sensors.Length; i++)
            obs[i] = sensors[i].Output;
        obs[sensors.Length] = controlledCar.Movement.Velocity;
        return obs;
    }

    private string BuildResponseJson(float[] obs, float reward, bool done)
    {
        StringBuilder sb = new StringBuilder();
        sb.Append("{\"obs\":[");
        for (int i = 0; i < obs.Length; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append(obs[i].ToString(System.Globalization.CultureInfo.InvariantCulture));
        }
        sb.Append("],\"reward\":");
        sb.Append(reward.ToString(System.Globalization.CultureInfo.InvariantCulture));
        sb.Append(",\"done\":");
        sb.Append(done ? "true" : "false");
        sb.Append('}');
        return sb.ToString();
    }

    private class Command
    {
        public string Name;
        public double[] Action;
    }

    private Command ParseCommand(string json)
    {
        try
        {
            if (json.Contains("\"reset\"")) return new Command { Name = "reset" };
            if (json.Contains("\"close\"")) return new Command { Name = "close" };
            if (json.Contains("\"step\""))
            {
                int idx = json.IndexOf("[", StringComparison.Ordinal);
                int jdx = json.IndexOf("]", idx + 1, StringComparison.Ordinal);
                if (idx >= 0 && jdx > idx)
                {
                    string arr = json.Substring(idx + 1, jdx - idx - 1);
                    string[] parts = arr.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    double[] act = new double[Math.Min(2, parts.Length)];
                    for (int i = 0; i < act.Length; i++)
                        act[i] = double.Parse(parts[i], System.Globalization.CultureInfo.InvariantCulture);
                    return new Command { Name = "step", Action = act };
                }
                return new Command { Name = "step", Action = new double[] { 0, 0 } };
            }
        }
        catch { }
        return null;
    }
    #endregion
}
