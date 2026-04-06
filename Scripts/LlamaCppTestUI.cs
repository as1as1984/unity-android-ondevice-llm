using System;
using System.Collections;
using System.IO;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// Test UI for llama.cpp + OpenCL GPU inference.
/// </summary>
public class LlamaCppTestUI : MonoBehaviour
{
    [Header("UI")]
    public Button generateButton;
    public TextMeshProUGUI statusText;
    public TextMeshProUGUI outputText;
    public TextMeshProUGUI timeText;

    private static readonly string Prompt =
        "<|system|>\nRPG dungeon JSON generator. Output ONLY valid JSON.<|end|>\n" +
        "<|user|>\nGenerate dungeon data for floors 1-5 in JSON. " +
        "Player: lazy baker's son (aggressive). " +
        "Floors 1-3: regular mobs, Floor 4: elite, Floor 5: boss. " +
        "Format: [{\"floor\":1,\"mob\":\"name\",\"hp\":number,\"atk\":number},...] " +
        "Output ONLY JSON.<|end|>\n<|assistant|>\n";

    private LlamaCppBridge _bridge;
    private bool _ready;
    private bool _generating;

    private void Start()
    {
        generateButton.interactable = false;
        generateButton.onClick.AddListener(OnGenerateClicked);
        SetStatus("Initializing llama.cpp...");
        SetOutput("");
        timeText.text = "";
        StartCoroutine(Initialize());
    }

    public void OnGenerateClicked()
    {
        Debug.Log($"[LlamaCppUI] Button clicked! _ready={_ready} _generating={_generating}");
        if (!_ready || _generating) return;
        StartCoroutine(Generate());
    }

    private IEnumerator Initialize()
    {
        string modelPath = "/data/local/tmp/model.gguf";
        if (!File.Exists(modelPath))
            modelPath = Path.Combine(Application.persistentDataPath, "Models", "llama-cpp", "model.gguf");

        Debug.Log($"[LlamaCppUI] modelPath: {modelPath}");
        Debug.Log($"[LlamaCppUI] exists: {File.Exists(modelPath)}");

        if (!File.Exists(modelPath))
        {
            SetStatus("Model not found.\nadb push model.gguf /data/local/tmp/ and restart");
            yield break;
        }

        SetStatus("Loading model (GPU)...");
        yield return null;

        _bridge = new LlamaCppBridge();
        Exception err = null;
        bool done = false;
        float loadStart = Time.realtimeSinceStartup;

        ThreadPool.QueueUserWorkItem(_ =>
        {
            try { _bridge.Initialize(modelPath, nGpuLayers: 99, nCtx: 1280); }
            catch (Exception e) { err = e; }
            finally { done = true; }
        });

        while (!done)
        {
            float elapsed = Time.realtimeSinceStartup - loadStart;
            SetStatus($"Loading model (GPU)... {elapsed:F0}s");
            yield return new WaitForSeconds(0.5f);
        }

        float loadTime = Time.realtimeSinceStartup - loadStart;

        if (err != null)
        {
            SetStatus($"Init failed: {err.Message}");
            Debug.LogError($"[LlamaCppUI] Init error: {err.Message}\n{err.StackTrace}");
            yield break;
        }

        _ready = true;
        generateButton.interactable = true;
        SetStatus($"Ready (loaded in {loadTime:F1}s). Press button to generate.");
    }

    private IEnumerator Generate()
    {
        _generating = true;
        generateButton.interactable = false;
        SetOutput("");
        timeText.text = "";
        SetStatus("Generating (GPU)...");

        string result = null;
        Exception err = null;
        bool done = false;
        float t0 = Time.realtimeSinceStartup;
        int tokenCount = 0;

        ThreadPool.QueueUserWorkItem(_ =>
        {
            try
            {
                result = _bridge.Generate(Prompt, 150, tok =>
                {
                    tokenCount++;
                });
            }
            catch (Exception e) { err = e; }
            finally { done = true; }
        });

        while (!done)
        {
            float elapsed = Time.realtimeSinceStartup - t0;
            timeText.text = $"Elapsed: {elapsed:F0}s | Tokens: {tokenCount}";
            yield return new WaitForSeconds(0.5f);
        }

        float total = Time.realtimeSinceStartup - t0;

        if (err != null)
        {
            SetStatus($"Error: {err.Message}");
            Debug.LogError($"[LlamaCppUI] Error: {err.Message}\n{err.StackTrace}");
        }
        else
        {
            string summary = $"Done: {total:F1}s | {tokenCount} tokens | {tokenCount / Mathf.Max(total, 0.1f):F1} tok/s";
            SetStatus(summary);
            SetOutput(result ?? "(null)");
            timeText.text = "";
            Debug.Log($"[LlamaCppUI] {summary}");
            Debug.Log($"[LlamaCppUI] Result: {result}");
        }

        generateButton.interactable = true;
        _generating = false;
    }

    private void SetStatus(string msg)
    {
        if (statusText != null) statusText.text = msg;
        Debug.Log($"[LlamaCppUI] {msg}");
    }

    private void SetOutput(string text)
    {
        if (outputText != null) outputText.text = text;
    }

    private void OnDestroy()
    {
        _bridge?.Dispose();
    }
}
