#region Includes
using UnityEngine;
#endregion

/// <summary>
/// Class representing a controlling container for a 2D physical simulation
/// of a car with 5 front facing sensors, detecting the distance to obstacles.
/// </summary>
public class CarController : MonoBehaviour
{
    #region Members
    #region IDGenerator
    // Used for unique ID generation
    private static int idGenerator = 0;
    /// <summary>
    /// Returns the next unique id in the sequence.
    /// </summary>
    private static int NextID
    {
        get { return idGenerator++; }
    }
    #endregion

    // Maximum delay in seconds between the collection of two checkpoints until this car dies.
    private const float MAX_CHECKPOINT_DELAY = 7;

    /// <summary>
    /// Current completion reward (tracked by TrackManager for ML-Agents).
    /// </summary>
    public float CurrentCompletionReward
    {
        get;
        set;
    }


    /// <summary>
    /// Whether this car is controllable by user input (keyboard).
    /// </summary>
    public bool UseUserInput = false;

    /// <summary>
    /// The ML-Agents CarAgent component.
    /// </summary>
    private CarAgent carAgent;

    /// <summary>
    /// The movement component of this car.
    /// </summary>
    public CarMovement Movement
    {
        get;
        private set;
    }

    /// <summary>
    /// The current inputs for controlling the CarMovement component.
    /// </summary>
    public double[] CurrentControlInputs
    {
        get { return Movement.CurrentInputs; }
    }

    /// <summary>
    /// The cached SpriteRenderer of this car.
    /// </summary>
    public SpriteRenderer SpriteRenderer
    {
        get;
        private set;
    }

    private Sensor[] sensors;
    private float timeSinceLastCheckpoint;
    #endregion

    #region Constructors
    void Awake()
    {
        //Cache components
        Movement = GetComponent<CarMovement>();
        SpriteRenderer = GetComponent<SpriteRenderer>();
        sensors = GetComponentsInChildren<Sensor>();
        
        // Check if using ML-Agents
        carAgent = GetComponent<CarAgent>();
    }
    void Start()
    {
        Movement.HitWall += Die;

        //Set name to be unique
        this.name = "Car (" + NextID + ")";
    }
    #endregion

    #region Methods
    /// <summary>
    /// Restarts this car, making it movable again.
    /// </summary>
    public void Restart()
    {
        Movement.enabled = true;
        timeSinceLastCheckpoint = 0;

        foreach (Sensor s in sensors)
            s.Show();

        // ML-Agents handles episode reset in OnEpisodeBegin
        this.enabled = true;
    }

    // Unity method for normal update
    void Update()
    {
        timeSinceLastCheckpoint += Time.deltaTime;
    }

    // Unity method for physics update
    void FixedUpdate()
    {
        // ML-Agents handles control in CarAgent.OnActionReceived
        // Just check for timeout
        if (timeSinceLastCheckpoint > MAX_CHECKPOINT_DELAY)
        {
            Die();
        }
    }

    // Makes this car die (making it unmovable and stops the Agent from calculating the controls for the car).
    private void Die()
    {
        this.enabled = false;
        Movement.Stop();
        Movement.enabled = false;

        foreach (Sensor s in sensors)
            s.Hide();

        // ML-Agents will handle episode end
        if (carAgent != null)
        {
            carAgent.OnWallHit();
        }
    }

    public void CheckpointCaptured()
    {
        timeSinceLastCheckpoint = 0;
        
        // Notify ML-Agents agent
        if (carAgent != null)
        {
            carAgent.OnCheckpointCaptured();
        }
    }

    public void FinishLap()
    {
        if (carAgent != null)
        {
            carAgent.OnTrackCompleted();
        }
    }
    #endregion
}
