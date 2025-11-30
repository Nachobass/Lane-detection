#region Includes
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
#endregion

public class CarAgent : Unity.MLAgents.Agent
{
    #region Members
    
    [Header("Car Components")]
    [SerializeField]
    private CarController carController;
    
    [SerializeField]
    private CarMovement carMovement;
    
    private Sensor[] sensors;
    
    [Header("Reward Settings")]
    [SerializeField]
    private float checkpointReward = 1.0f;
    
    [SerializeField]
    private float wallHitPenalty = -1.0f;
    
    [SerializeField]
    private float timeoutPenalty = -1.0f;

    [SerializeField] 
    private float existPenalty = -0.001f;  

    [SerializeField]
    private float checkpointDelayPenalty = 0.0f;
    
    [SerializeField]
    private float progressRewardMultiplier = 5.0f;


    [SerializeField]
    private float velocityRewardMultiplier = 0.0f;
    
    [Header("Track Settings")]
    [SerializeField]
    private TrackManager trackManager;
    
    private float previousCompletion = 0f;
    private float timeSinceLastCheckpoint = 0f;
    private const float CHECKPOINT_DELAY_PENALTY = 10f;
    private const float MAX_CHECKPOINT_DELAY = 50f;
    private uint currentCheckpointIndex = 1;
    
    void FixedUpdate()
    {
        if (trackManager != null && carController != null)
        {
            float currentCompletion = carController.CurrentCompletionReward;
            if (currentCompletion > previousCompletion + 0.0001f)
            {
                float progressReward = (currentCompletion - previousCompletion) * progressRewardMultiplier;
                AddReward(progressReward);
                // if (carMovement.Velocity > 2.0f)
                // {
                //     // 0.001 puntos por cada frame que vaya rapido
                //     AddReward(velocityRewardMultiplier * carMovement.Velocity); 
                // }
                previousCompletion = currentCompletion;
            }
        }
    }
    
    #endregion
    
    #region Unity Methods
    
    public override void Initialize()
    {
        base.Initialize();
        
        if (carController == null)
            carController = GetComponent<CarController>();
        
        if (carMovement == null)
            carMovement = GetComponent<CarMovement>();
        
        if (trackManager == null)
            trackManager = FindFirstObjectByType<TrackManager>();
        
        sensors = GetComponentsInChildren<Sensor>();
        
        if (carController != null)
        {
            carController.UseUserInput = false;
        }
    }
    
    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        
        if (carController != null)
        {
            if (trackManager != null && trackManager.PrototypeCar != null)
            {
                carController.transform.position = trackManager.PrototypeCar.transform.position;
                carController.transform.rotation = trackManager.PrototypeCar.transform.rotation;
            }
            else
            {
                carController.transform.position = Vector3.zero;
                carController.transform.rotation = Quaternion.identity;
            }
            
            if (carMovement != null)
            {
                carMovement.enabled = true;
                carMovement.Stop();
                carMovement.SetInputs(new double[] { 0, 0 });
            }
            
            if (sensors != null)
            {
                foreach (Sensor s in sensors)
                    s.Show();
            }
        }
        
        previousCompletion = 0f;
        timeSinceLastCheckpoint = 0f;
        currentCheckpointIndex = 1;
    }
    
    #endregion
    
    #region ML-Agents Methods
    
    public override void CollectObservations(VectorSensor sensor)
    {
        if (sensors != null)
        {
            foreach (Sensor s in sensors)
            {
                float normalizedOutput = Mathf.Clamp01(s.Output / 10f);
                sensor.AddObservation(normalizedOutput);
            }
        }
        else
        {
            for (int i = 0; i < 5; i++)
                sensor.AddObservation(0f);
        }
        
        if (carMovement != null)
        {
            float normalizedVelocity = Mathf.Clamp(carMovement.Velocity / 20f, -1f, 1f);
            sensor.AddObservation(normalizedVelocity);
        }
        else
        {
            sensor.AddObservation(0f);
        }
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        float turn = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float throttle = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        
        if (carMovement != null)
        {
            carMovement.SetInputs(new double[] { turn, throttle });
        }

        AddReward(existPenalty);
        
        timeSinceLastCheckpoint += Time.fixedDeltaTime;
        if (timeSinceLastCheckpoint > CHECKPOINT_DELAY_PENALTY)
        {
            AddReward(checkpointDelayPenalty);
        }
        if (timeSinceLastCheckpoint > MAX_CHECKPOINT_DELAY)
        {
            AddReward(timeoutPenalty);
            EndEpisode();
        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
    
    #endregion
    
    #region Reward Methods
    
    public void OnWallHit()
    {
        AddReward(wallHitPenalty);
        EndEpisode();
    }
    
    public void OnCheckpointCaptured()
    {
        AddReward(checkpointReward);
        timeSinceLastCheckpoint = 0f;
        currentCheckpointIndex++;
    }
    
    public void OnTrackCompleted()
    {
        AddReward(100f);
        EndEpisode();
    }
    
    #endregion
}

