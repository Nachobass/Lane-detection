/// Script para crear automáticamente un auto para ML-Agents
/// Este script asegura que haya al menos un agente en la escena cuando ML-Agents se conecta

#region Includes
using UnityEngine;
#endregion

public class MLAgentsCarSpawner : MonoBehaviour
{
    #region Members
    
    [Header("ML-Agents Configuration")]
    [SerializeField]
    [Tooltip("Número de autos a crear para ML-Agents (normalmente 1 para entrenamiento)")]
    private int numberOfCars = 1;
    
    [SerializeField]
    [Tooltip("Si es true, crea los autos automáticamente al inicio")]
    private bool autoSpawnOnStart = true;
    
    #endregion
    
    #region Unity Methods
    
    void Start()
    {
        if (autoSpawnOnStart)
        {
            SpawnCars();
        }
    }
    
    #endregion
    
    #region Methods
    
    /// <summary>
    /// Crea los autos usando el TrackManager si está disponible.
    /// </summary>
    public void SpawnCars()
    {
        if (TrackManager.Instance != null)
        {
            // Usar el método del TrackManager para crear autos
            TrackManager.Instance.SetCarAmount(numberOfCars);
            Debug.Log($"[MLAgentsCarSpawner] Created {numberOfCars} car(s) for ML-Agents training.");
        }
        else
        {
            Debug.LogWarning("[MLAgentsCarSpawner] TrackManager.Instance is null. Make sure the track scene is loaded.");
        }
    }
    
    #endregion
}

