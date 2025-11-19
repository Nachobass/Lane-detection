#region Includes
using System;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
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
            // 3. Find any CarController in the scene (prefer one with sensors)
            else
            {
                CarController[] allCars = FindObjectsOfType<CarController>();
                if (allCars == null || allCars.Length == 0)
                {
                    Debug.LogError("RLBridgeServer: No CarController found in scene or prefab assigned!");
                    return;
                }
                
                // Prefer a car that has sensors
                CarController carWithSensors = null;
                foreach (CarController car in allCars)
                {
                    Sensor[] carSensors = car.GetComponentsInChildren<Sensor>();
                    if (carSensors != null && carSensors.Length > 0)
                    {
                        carWithSensors = car;
                        break; // Found one with sensors, use it
                    }
                }
                
                if (carWithSensors != null)
                {
                    controlledCar = carWithSensors;
                    Debug.Log($"RLBridgeServer: Using CarController '{controlledCar.name}' with sensors (found {allCars.Length} total cars)");
                }
                else
                {
                    // No car has sensors, use the first one found
                    controlledCar = allCars[0];
                    Debug.LogWarning($"RLBridgeServer: Using CarController '{controlledCar.name}' as fallback (no car with sensors found out of {allCars.Length} total cars)");
                }
            }
        }
        
        // Check if assigned car has sensors, if not, try to find one that does
        if (controlledCar != null)
        {
            Sensor[] assignedCarSensors = controlledCar.GetComponentsInChildren<Sensor>();
            if (assignedCarSensors == null || assignedCarSensors.Length == 0)
            {
                // Assigned car has no sensors, look for one that does
                CarController[] allCars = FindObjectsOfType<CarController>();
                CarController carWithSensors = null;
                
                foreach (CarController car in allCars)
                {
                    if (car == controlledCar) continue; // Skip the one we already checked
                    Sensor[] carSensors = car.GetComponentsInChildren<Sensor>();
                    if (carSensors != null && carSensors.Length > 0)
                    {
                        carWithSensors = car;
                        break;
                    }
                }
                
                if (carWithSensors != null)
                {
                    Debug.LogWarning($"RLBridgeServer: Assigned car '{controlledCar.name}' has no sensors. Switching to '{carWithSensors.name}' which has sensors.");
                    controlledCar = carWithSensors;
                }
            }
        }

        // Configure external control
        controlledCar.UseUserInput = false;
        controlledCar.UseExternalControl = true;

        // Get sensors
        sensors = controlledCar.GetComponentsInChildren<Sensor>();
        if (sensors == null || sensors.Length == 0)
        {
            Debug.LogWarning($"RLBridgeServer: Car '{controlledCar.name}' has no sensors! Creating 5 sensors automatically.");
            CreateSensorsForCar(controlledCar);
            sensors = controlledCar.GetComponentsInChildren<Sensor>();
        }
        
        if (sensors == null || sensors.Length == 0)
        {
            Debug.LogWarning($"RLBridgeServer: Car '{controlledCar.name}' has no sensors! Observation will only include velocity.");
            sensors = new Sensor[0]; // Ensure it's not null
        }
        else
        {
            Debug.Log($"RLBridgeServer: Found {sensors.Length} sensors on car '{controlledCar.name}'");
        }
        
        // Verify Movement component exists
        if (controlledCar.Movement == null)
        {
            Debug.LogError($"RLBridgeServer: Car '{controlledCar.name}' does not have a Movement component!");
        }
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

        // Procesar reset si se solicitó
        if (resetRequestedEvent.WaitOne(0))
        {
            Debug.Log("RLBridgeServer: FixedUpdate processing reset");

            if (controlledCar != null)
            {
                // Reset car position and rotation
                controlledCar.transform.position = Vector3.zero;
                controlledCar.transform.rotation = Quaternion.identity;

                // Reset car: use Restart() if Agent exists, otherwise reset manually
                try
                {
                    if (controlledCar.Agent != null)
                    {
                        // Agent exists - use Restart() which will handle everything including Agent.Reset()
                        controlledCar.Restart();
                        Debug.Log("RLBridgeServer: Car.Restart() called successfully (Agent exists)");
                    }
                    else
                    {
                        // Agent is null (external control) - manually reset car without calling Agent.Reset()
                        // Enable movement
                        if (controlledCar.Movement != null)
                        {
                            controlledCar.Movement.enabled = true;
                        }
                        else
                        {
                            Debug.LogWarning("RLBridgeServer: controlledCar.Movement is null during reset");
                        }

                        // Show sensors
                        if (sensors == null)
                        {
                            sensors = controlledCar.GetComponentsInChildren<Sensor>();
                        }
                        if (sensors != null && sensors.Length > 0)
                        {
                            foreach (Sensor s in sensors)
                            {
                                if (s != null)
                                {
                                    s.Show();
                                }
                            }
                        }

                        // Reset checkpoint timer (CheckpointCaptured sets timeSinceLastCheckpoint = 0)
                        controlledCar.CheckpointCaptured();

                        // Enable the car
                        controlledCar.enabled = true;

                        Debug.Log("RLBridgeServer: Car reset manually (no Agent - external control mode)");
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogError($"RLBridgeServer: Error during car reset: {ex.Message}\n{ex.StackTrace}");
                    // Try to at least enable the car and movement even if reset partially failed
                    try
                    {
                        if (controlledCar.Movement != null)
                        {
                            controlledCar.Movement.enabled = true;
                        }
                        controlledCar.enabled = true;
                    }
                    catch { }
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
            try
            {
                if (controlledCar == null)
                {
                    Debug.LogError("RLBridgeServer: controlledCar is null in FixedUpdate step processing");
                    stepResponseJson = BuildResponseJson(new float[] { 0f }, 0f, true);
                    stepCompleted.Set();
                    return;
                }

                double turn = 0, throttle = 0;
                if (pendingAction != null && pendingAction.Length >= 2)
                {
                    turn = Mathf.Clamp((float)pendingAction[0], -1f, 1f);
                    throttle = Mathf.Clamp((float)pendingAction[1], -1f, 1f);
                }

                if (controlledCar.Movement != null)
                {
                    controlledCar.Movement.SetInputs(new double[] { turn, throttle });
                }
                else
                {
                    Debug.LogWarning("RLBridgeServer: controlledCar.Movement is null");
                }

                float[] obs = BuildObservation();
                
                // Safely get completion reward
                // The reward is typically updated by TrackManager, but Agent may be null when using external control
                float completion = 0f;
                try
                {
                    // Try to get reward from CarController's CurrentCompletionReward property
                    if (controlledCar != null)
                    {
                        // Check if Agent exists before accessing its properties
                        var agent = controlledCar.Agent;
                        if (agent != null)
                        {
                            // Check if Genotype exists before accessing CurrentCompletionReward
                            if (agent.Genotype != null)
                            {
                                completion = controlledCar.CurrentCompletionReward;
                            }
                            else
                            {
                                // Agent exists but Genotype is null
                                completion = 0f;
                            }
                        }
                        else
                        {
                            // Agent is null - this is expected when using external control without Agent setup
                            completion = 0f;
                        }
                    }
                }
                catch (System.NullReferenceException)
                {
                    // Agent or Genotype is null - this is expected when using external control without Agent setup
                    completion = 0f;
                }
                catch (System.Exception ex)
                {
                    // Other exception - log but continue with 0 reward
                    Debug.LogWarning($"RLBridgeServer: Error accessing completion reward: {ex.Message}");
                    completion = 0f;
                }
                
                float reward = completion - prevCompletion;
                prevCompletion = completion;
                bool done = controlledCar == null || !controlledCar.enabled;

                stepResponseJson = BuildResponseJson(obs, reward, done);
                episodeReward += reward;
                stepCompleted.Set();
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"RLBridgeServer: Exception in FixedUpdate step processing: {ex.Message}\n{ex.StackTrace}");
                // Return safe default observation
                stepResponseJson = BuildResponseJson(new float[] { 0f }, 0f, false);
                stepCompleted.Set();
            }
        }
        // if (stepRequested.WaitOne(0))
        // {
        //     try
        //     {
        //         double turn = 0, throttle = 0;
        //         if (pendingAction != null && pendingAction.Length >= 2)
        //         {
        //             turn = Mathf.Clamp((float)pendingAction[0], -1f, 1f);
        //             throttle = Mathf.Clamp((float)pendingAction[1], -1f, 1f);
        //         }

        //         if (controlledCar != null && controlledCar.Movement != null)
        //         {
        //             controlledCar.Movement.SetInputs(new double[] { turn, throttle });
        //         }

        //         float[] obs = BuildObservation();
        //         float completion = controlledCar != null ? controlledCar.CurrentCompletionReward : 0f;
        //         float reward = completion - prevCompletion;
        //         prevCompletion = completion;
        //         bool done = controlledCar == null || !controlledCar.enabled;

        //         stepResponseJson = BuildResponseJson(obs, reward, done);
        //         episodeReward += reward;
        //         stepCompleted.Set();
        //         Debug.Log($"RLBridgeServer: Step processed successfully. Reward: {reward}, Done: {done}");
        //     }
        //     catch (System.Exception ex)
        //     {
        //         Debug.LogError($"RLBridgeServer: Error processing step in FixedUpdate: {ex.Message}\n{ex.StackTrace}");
        //         // Set error response
        //         stepResponseJson = $"{{\"error\":\"step_processing_failed\",\"message\":\"{ex.Message}\"}}";
        //         stepCompleted.Set();
        //     }
        // }

        // if (stepRequested.WaitOne(0))
        // {
        //     Debug.Log("RLBridgeServer: Step triggered in FixedUpdate");

        //     // ignorar el auto por ahora
        //     float[] obs = new float[] { 0f };
        //     float reward = 0f;
        //     bool done = false;

        //     stepResponseJson = BuildResponseJson(obs, reward, done);
        //     stepCompleted.Set();

        //     Debug.Log("RLBridgeServer: Step completed with dummy response");
        // }

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

    // private void ListenLoop()
    // {
    //     while (running)
    //     {
    //         try
    //         {
    //             using (TcpClient client = listener.AcceptTcpClient())
    //             using (var stream = client.GetStream())
    //             {
    //                 stream.ReadTimeout = 60000;
    //                 stream.WriteTimeout = 60000;
    //                 var reader = new System.IO.StreamReader(stream, Encoding.UTF8);
    //                 var writer = new System.IO.StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };

    //                 Debug.Log("RLBridgeServer: Client connected");

    //                 while (running && client.Connected)
    //                 {
    //                     string line = reader.ReadLine();
    //                     if (line == null) break;

    //                     Debug.Log($"RLBridgeServer: Received command: {line}");

    //                     var cmd = ParseCommand(line);
    //                     if (cmd == null)
    //                     {
    //                         Debug.LogError($"RLBridgeServer: Failed to parse command: {line}");
    //                         writer.WriteLine("{\"error\":\"bad_json\"}");
    //                         continue;
    //                     }

    //                     if (cmd.Name == "reset")
    //                     {
    //                         Debug.Log("RLBridgeServer: Processing reset command");
    //                         resetResponseJson = null;
    //                         Debug.Log("RLBridgeServer: Setting resetRequestedEvent");
    //                         resetRequestedEvent.Set();            // trigger reset
    //                         Debug.Log("RLBridgeServer: Waiting for reset completion...");
    //                         // bool completed = resetCompleted.WaitOne(1000);         // wait for completion
    //                         bool completed = resetCompleted.WaitOne();         // wait for completion
    //                         Debug.Log($"RLBridgeServer: Reset completed: {completed}"); 
    //                         if (!completed) throw new Exception("Reset failed");
    //                         string resp = resetResponseJson ?? "{\"error\":\"timeout\"}";
    //                         resetResponseJson = null;
    //                         writer.WriteLine(resp);
    //                         Debug.Log($"RLBridgeServer: Reset response: {resp}");
    //                     }
    //                     else if (cmd.Name == "step")
    //                     {
    //                         Debug.Log($"RLBridgeServer: Processing step command: [{cmd.Action[0]}, {cmd.Action[1]}]");
    //                         pendingAction = cmd.Action;
    //                         stepRequested.Set();
    //                         stepCompleted.WaitOne(5000);
    //                         string resp = stepResponseJson ?? "{\"error\":\"timeout\"}";
    //                         stepResponseJson = null;
    //                         writer.WriteLine(resp);
    //                         Debug.Log($"RLBridgeServer: Step response: {resp}");
    //                     }
    //                     else if (cmd.Name == "close")
    //                     {
    //                         Debug.Log("RLBridgeServer: Client requested close");
    //                         writer.WriteLine("{\"ok\":true}");
    //                         break;
    //                     }
    //                     else
    //                     {
    //                         Debug.LogError($"RLBridgeServer: Unknown command: {cmd.Name}");
    //                         writer.WriteLine("{\"error\":\"unknown_cmd\"}");
    //                     }
    //                 }
    //             }
    //         }
    //         catch (SocketException)
    //         {
    //             if (!running) break;
    //         }
    //         catch (Exception ex)
    //         {
    //             Debug.LogError($"RLBridgeServer exception: {ex}");
    //         }
    //     }
    // }
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
                            resetRequestedEvent.Set(); // trigger reset

                            Debug.Log("RLBridgeServer: Waiting for reset completion...");
                            bool completed = false;
                            int waited = 0;
                            int timeoutMs = 10000; // 5 segundos
                            int intervalMs = 10;

                            while (!completed && waited < timeoutMs)
                            {
                                completed = resetCompleted.WaitOne(intervalMs);
                                waited += intervalMs;
                            }

                            if (!completed)
                            {
                                Debug.LogWarning("RLBridgeServer: Reset timeout");
                            }
                            else
                            {
                                Debug.Log("RLBridgeServer: Reset completed successfully");
                            }

                            string resp = resetResponseJson ?? "{\"error\":\"timeout\"}";
                            resetResponseJson = null;
                            writer.WriteLine(resp);
                            Debug.Log($"RLBridgeServer: Reset response: {resp}");
                        }
                        else if (cmd.Name == "step")
                        {
                            Debug.Log($"RLBridgeServer: Processing step command: [{cmd.Action[0]}, {cmd.Action[1]}]");
                            // Reset the completion event before requesting a new step
                            stepCompleted.Reset();
                            stepResponseJson = null;
                            pendingAction = cmd.Action;
                            stepRequested.Set();

                            bool stepDone = false;
                            int waitedStep = 0;
                            int stepTimeoutMs = 10000; // Increased timeout to 10 seconds
                            int stepIntervalMs = 10;

                            while (!stepDone && waitedStep < stepTimeoutMs)
                            {
                                stepDone = stepCompleted.WaitOne(stepIntervalMs);
                                waitedStep += stepIntervalMs;
                            }

                            if (!stepDone)
                            {
                                Debug.LogError($"RLBridgeServer: Step timeout after {waitedStep}ms. FixedUpdate may not be running or car is not responding.");
                            }

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
    /// <summary>
    /// Automatically creates 5 sensors for a car that doesn't have any.
    /// Sensors are arranged in a typical front-facing configuration.
    /// </summary>
    private void CreateSensorsForCar(CarController car)
    {
        if (car == null) return;

        try
        {
            // Try to load cross sprite from Resources (if available)
            Sprite crossSprite = Resources.Load<Sprite>("Sprites/cross");
            if (crossSprite == null)
            {
                // Try alternative path
                crossSprite = Resources.Load<Sprite>("cross");
            }
            
            // If still null, create a simple default sprite
            if (crossSprite == null)
            {
                // Create a simple 4x4 white texture for the cross
                Texture2D tex = new Texture2D(4, 4);
                Color[] pixels = new Color[16];
                for (int i = 0; i < 16; i++)
                    pixels[i] = Color.white;
                tex.SetPixels(pixels);
                tex.Apply();
                
                crossSprite = Sprite.Create(tex, new Rect(0, 0, 4, 4), new Vector2(0.5f, 0.5f), 40f);
            }

            // Define 5 sensor positions and directions (typical front-facing arrangement)
            // Format: (localPosition, crossOffset) where crossOffset determines direction
            Vector2[] sensorPositions = new Vector2[]
            {
                new Vector2(0f, 0f),        // Sensor 0: Center (at car origin)
                new Vector2(-0.3f, 0f),     // Sensor 1: Slight left
                new Vector2(0.3f, 0f),      // Sensor 2: Slight right
                new Vector2(-0.5f, 0f),     // Sensor 3: Left side
                new Vector2(0.5f, 0f),      // Sensor 4: Right side
            };

            Vector2[] crossOffsets = new Vector2[]
            {
                new Vector2(0f, 1.5f),      // Sensor 0: Forward
                new Vector2(-0.5f, 1.2f),   // Sensor 1: Forward-left
                new Vector2(0.5f, 1.2f),    // Sensor 2: Forward-right
                new Vector2(-1.5f, 0f),     // Sensor 3: Left
                new Vector2(1.5f, 0f),      // Sensor 4: Right
            };

            // Try to get layer mask from existing sensors (if any other car has sensors configured)
            int layerMask = -1; // All layers by default
            Sensor[] existingSensors = FindObjectsOfType<Sensor>();
            if (existingSensors != null && existingSensors.Length > 0)
            {
                // Use reflection to get LayerToSense from an existing sensor
                var field = typeof(Sensor).GetField("LayerToSense", BindingFlags.NonPublic | BindingFlags.Instance);
                if (field != null)
                {
                    LayerMask existingMask = (LayerMask)field.GetValue(existingSensors[0]);
                    layerMask = existingMask.value;
                }
            }
            // If no existing sensors or failed to get mask, use all layers (-1) to detect walls
            if (layerMask == -1)
            {
                layerMask = -1; // All layers (will detect walls on any layer)
            }

            // Create 5 sensors
            for (int i = 0; i < 5; i++)
            {
                // Create sensor GameObject
                GameObject sensorObj = new GameObject($"Sensor{i}");
                sensorObj.transform.SetParent(car.transform, false);
                sensorObj.transform.localPosition = sensorPositions[i];

                // Add Sensor component
                Sensor sensor = sensorObj.AddComponent<Sensor>();

                // Create Cross child GameObject
                GameObject crossObj = new GameObject("Cross");
                crossObj.transform.SetParent(sensorObj.transform, false);
                crossObj.transform.localPosition = crossOffsets[i];

                // Add SpriteRenderer to Cross
                SpriteRenderer crossRenderer = crossObj.AddComponent<SpriteRenderer>();
                if (crossSprite != null)
                {
                    crossRenderer.sprite = crossSprite;
                }
                crossRenderer.color = new Color(1f, 0f, 0f, 0.5f); // Semi-transparent red for visibility

                // Set Cross reference and LayerToSense using reflection
                var crossField = typeof(Sensor).GetField("Cross", BindingFlags.NonPublic | BindingFlags.Instance);
                var layerField = typeof(Sensor).GetField("LayerToSense", BindingFlags.NonPublic | BindingFlags.Instance);

                if (crossField != null)
                {
                    crossField.SetValue(sensor, crossRenderer);
                }
                if (layerField != null)
                {
                    layerField.SetValue(sensor, (LayerMask)layerMask);
                }
            }

            Debug.Log($"RLBridgeServer: Created 5 sensors automatically for car '{car.name}'");
        }
        catch (Exception ex)
        {
            Debug.LogError($"RLBridgeServer: Failed to create sensors automatically: {ex.Message}\n{ex.StackTrace}");
        }
    }

    private float[] BuildObservation()
    {
        try
        {
            if (controlledCar == null)
            {
                Debug.LogError("RLBridgeServer: controlledCar is null in BuildObservation");
                return new float[] { 0f }; // Return minimal observation
            }

            // Get sensors if not already cached
            if (sensors == null)
            {
                sensors = controlledCar.GetComponentsInChildren<Sensor>();
                if (sensors == null)
                {
                    Debug.LogWarning("RLBridgeServer: GetComponentsInChildren<Sensor>() returned null, using empty array");
                    sensors = new Sensor[0];
                }
                Debug.Log($"RLBridgeServer: Found {sensors.Length} sensors on car");
            }

            // Create observation array: sensors + velocity
            float[] obs = new float[sensors.Length + 1];
            
            // Read sensor values (handle null sensors)
            for (int i = 0; i < sensors.Length; i++)
            {
                if (sensors[i] != null)
                {
                    obs[i] = sensors[i].Output;
                }
                else
                {
                    Debug.LogWarning($"RLBridgeServer: Sensor at index {i} is null");
                    obs[i] = 0f;
                }
            }

            // Add velocity
            if (controlledCar.Movement != null)
            {
                obs[sensors.Length] = controlledCar.Movement.Velocity;
            }
            else
            {
                Debug.LogWarning("RLBridgeServer: controlledCar.Movement is null, velocity set to 0");
                obs[sensors.Length] = 0f;
            }

            return obs;
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"RLBridgeServer: Exception in BuildObservation: {ex.Message}\n{ex.StackTrace}");
            // Return safe default observation
            return new float[] { 0f };
        }
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