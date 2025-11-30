/// Author: Samuel Arzt
/// Date: March 2017


#region Includes
using UnityEngine.UI;
using UnityEngine;
using System;
#endregion

/// <summary>
/// Class for controlling the various ui elements of the simulation
/// </summary>
public class UISimulationController : MonoBehaviour
{
    #region Members
    private CarController target;
    /// <summary>
    /// The Car to fill the GUI data with.
    /// </summary>
    public CarController Target
    {
        get { return target; }
        set
        {
            if (target != value)
            {
                target = value;
            }
        }
    }

    // GUI element references to be set in Unity Editor.
    [SerializeField]
    private Text[] InputTexts;
    [SerializeField]
    private Text Evaluation;
    [SerializeField]
    private Text GenerationCount;
    #endregion

    #region Constructors
    #endregion

    #region Methods
    void Update()
    {
        if (Target != null)
        {
            //Display controls
            if (Target.CurrentControlInputs != null)
            {
                for (int i = 0; i < InputTexts.Length && i < Target.CurrentControlInputs.Length; i++)
                    InputTexts[i].text = Target.CurrentControlInputs[i].ToString();
            }

            var carAgent = Target.GetComponent<CarAgent>();
            if (carAgent != null)
            {
                Evaluation.text = Target.CurrentCompletionReward.ToString("F3");
                if (GenerationCount != null)
                    GenerationCount.text = "ML-Agents";
            }
            else
            {
                Evaluation.text = "N/A";
                if (GenerationCount != null)
                    GenerationCount.text = "N/A";
            }
        }
    }

    /// <summary>
    /// Starts to display the gui elements.
    /// </summary>
    public void Show()
    {
        gameObject.SetActive(true);
    }

    /// <summary>
    /// Stops displaying the gui elements.
    /// </summary>
    public void Hide()
    {
        gameObject.SetActive(false);
    }
    #endregion
}
