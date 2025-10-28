
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
    private CarController controlledCar; // assign from Inspector or find in scene

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

    #endregion

    #region Unity Methods

    void Awake()
    {
        // 1. Use the car assigned in the Inspector
        if (controlledCar == null)
        {
            // 2. Instantiate prefab if assigned
            if (carPrefab != null)
            {
                controlledCar = Instantiate(carPrefab);
                controlledCar.name = carPrefab.name;
            }
            // 3. Find any CarController in the scene
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

        // Configure external control
        controlledCar.UseUserInput = false;
        controlledCar.UseExternalControl = true;

        // Get sensors
        sensors = controlledCar.GetComponentsInChildren<Sensor>();
        Debug.Log($"RLBridgeServer: Found {sensors.Length} sensors");
    }

    void Start()
    {
        StartServer();
    }

    void OnDestroy()
    {
        StopServer();
    }

    // void FixedUpdate()
    // {
    //     // Debug: Log every 60 frames to see if FixedUpdate is running
    //     if (Time.frameCount % 60 == 0)
    //     {
    //         // Debug.Log($"RLBridgeServer: FixedUpdate running (frame {Time.frameCount})");
    //     }

    //     if (controlledCar == null) 
    //     {
    //         // Debug.LogWarning("RLBridgeServer: FixedUpdate - controlledCar is null");
    //         return;
    //     }

    //     // Debug: Check if reset is requested
    //     if (resetRequestedEvent.WaitOne(0))
    //     {
    //         // Debug.Log("RLBridgeServer: FixedUpdate processing reset");
    //     }

    //     // Process step action if requested
    //     if (stepRequested.WaitOne(0))
    //     {
    //         double turn = 0, throttle = 0;
    //         if (pendingAction != null && pendingAction.Length >= 2)
    //         {
    //             turn = Mathf.Clamp((float)pendingAction[0], -1f, 1f);
    //             throttle = Mathf.Clamp((float)pendingAction[1], -1f, 1f);
    //         }

    //         controlledCar.Movement.SetInputs(new double[] { turn, throttle });

    //         float[] obs = BuildObservation();
    //         float completion = controlledCar.CurrentCompletionReward;
    //         float reward = completion - prevCompletion;
    //         prevCompletion = completion;
    //         bool done = !controlledCar.enabled;

    //         stepResponseJson = BuildResponseJson(obs, reward, done);
    //         episodeReward += reward;

    //         stepCompleted.Set();
    //     }

    //     // Process reset if requested
    //     if (resetRequestedEvent.WaitOne(0))
    //     {
    //         Debug.Log("RLBridgeServer: FixedUpdate processing reset");

    //         if (controlledCar != null)
    //         {
    //             Debug.Log("RLBridgeServer: Resetting car position and rotation");
    //             controlledCar.transform.position = Vector3.zero;
    //             controlledCar.transform.rotation = Quaternion.identity;
                
    //             // Try to restart the car, but don't fail if the method doesn't exist
    //             try
    //             {
    //             controlledCar.Restart();
    //                 Debug.Log("RLBridgeServer: Car.Restart() called successfully");
    //             }
    //             catch (Exception ex)
    //             {
    //                 Debug.LogWarning($"RLBridgeServer: Car.Restart() failed: {ex.Message}");
    //             }
    //         }
    //         else
    //         {
    //             Debug.LogError("RLBridgeServer: controlledCar is null during reset!");
    //         }

    //         prevCompletion = 0f;
    //         episodeReward = 0f;

    //         Debug.Log("RLBridgeServer: Building observation after reset");
    //         float[] obs = BuildObservation();
    //         resetResponseJson = BuildResponseJson(obs, 0f, false);

    //         Debug.Log("RLBridgeServer: Setting reset completed event");
    //         // Notify that reset is complete
    //         resetCompleted.Set();
    //         Debug.Log("RLBridgeServer: Reset completed successfully");
    //     }
    // }

    void FixedUpdate()
    {
        if (controlledCar == null) return;

        // Procesar reset si se solicitó
        if (resetRequestedEvent.WaitOne(0))
        {
            Debug.Log("RLBridgeServer: FixedUpdate processing reset");

            if (controlledCar != null)
            {
                controlledCar.transform.position = Vector3.zero;
                controlledCar.transform.rotation = Quaternion.identity;

                try
                {
                    controlledCar.Restart();
                    Debug.Log("RLBridgeServer: Car.Restart() called successfully");
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"RLBridgeServer: Car.Restart() failed: {ex.Message}");
                }
            }

            prevCompletion = 0f;
            episodeReward = 0f;

            float[] obs = BuildObservation();
            resetResponseJson = BuildResponseJson(obs, 0f, false);
            resetCompleted.Set();

            Debug.Log("RLBridgeServer: Reset completed successfully");
        }

        // Procesar paso si se solicitó
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

                    Debug.Log("RLBridgeServer: Client connected");

                    while (running && client.Connected)
                    {
                        string line = reader.ReadLine();
                        if (line == null) break;

                        Debug.Log($"RLBridgeServer: Received command: {line}");

                        var cmd = ParseCommand(line);
                        if (cmd == null)
                        {
                            Debug.LogError($"RLBridgeServer: Failed to parse command: {line}");
                            writer.WriteLine("{\"error\":\"bad_json\"}");
                            continue;
                        }

                        if (cmd.Name == "reset")
                        {
                            Debug.Log("RLBridgeServer: Processing reset command");
                            resetResponseJson = null;
                            Debug.Log("RLBridgeServer: Setting resetRequestedEvent");
                            resetRequestedEvent.Set();            // trigger reset
                            Debug.Log("RLBridgeServer: Waiting for reset completion...");
                            bool completed = resetCompleted.WaitOne(5000);         // wait for completion
                            Debug.Log($"RLBridgeServer: Reset completed: {completed}");
                            string resp = resetResponseJson ?? "{\"error\":\"timeout\"}";
                            resetResponseJson = null;
                            writer.WriteLine(resp);
                            Debug.Log($"RLBridgeServer: Reset response: {resp}");
                        }
                        else if (cmd.Name == "step")
                        {
                            Debug.Log($"RLBridgeServer: Processing step command: [{cmd.Action[0]}, {cmd.Action[1]}]");
                            pendingAction = cmd.Action;
                            stepRequested.Set();
                            stepCompleted.WaitOne(5000);
                            string resp = stepResponseJson ?? "{\"error\":\"timeout\"}";
                            stepResponseJson = null;
                            writer.WriteLine(resp);
                            Debug.Log($"RLBridgeServer: Step response: {resp}");
                        }
                        else if (cmd.Name == "close")
                        {
                            Debug.Log("RLBridgeServer: Client requested close");
                            writer.WriteLine("{\"ok\":true}");
                            break;
                        }
                        else
                        {
                            Debug.LogError($"RLBridgeServer: Unknown command: {cmd.Name}");
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
        float[] obs = new float[sensors.Length + 1]; // sensors + velocity
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