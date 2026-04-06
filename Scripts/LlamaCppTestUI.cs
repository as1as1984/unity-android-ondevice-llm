using System;
using System.Collections;
using System.IO;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// llama.cpp + OpenCL GPU 테스트 UI.
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
        "<|user|>\n던전 5층 데이터 JSON. 플레이어: 게으른 빵집 아들(공격형). " +
        "1-3층 일반몹, 4층 엘리트, 5층 보스. " +
        "형식: [{\"floor\":1,\"mob\":\"몬스터이름\",\"hp\":숫자,\"atk\":숫자},...] " +
        "JSON만 출력.<|end|>\n<|assistant|>\n";

    private LlamaCppBridge _bridge;
    private bool _ready;
    private bool _generating;

    private void Start()
    {
        generateButton.interactable = false;
        generateButton.onClick.AddListener(OnGenerateClicked);
        SetStatus("llama.cpp 초기화 중...");
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
            SetStatus("모델 없음.\nadb push model.gguf /data/local/tmp/ 후 재시작");
            yield break;
        }

        SetStatus("모델 로딩 중 (GPU)...");
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
            SetStatus($"모델 로딩 중 (GPU)... {elapsed:F0}s");
            yield return new WaitForSeconds(0.5f);
        }

        float loadTime = Time.realtimeSinceStartup - loadStart;

        if (err != null)
        {
            SetStatus($"초기화 실패: {err.Message}");
            Debug.LogError($"[LlamaCppUI] Init error: {err.Message}\n{err.StackTrace}");
            yield break;
        }

        _ready = true;
        generateButton.interactable = true;
        SetStatus($"준비 완료 (로딩 {loadTime:F1}s). 버튼을 눌러 생성하세요.");
    }

    private IEnumerator Generate()
    {
        _generating = true;
        generateButton.interactable = false;
        SetOutput("");
        timeText.text = "";
        SetStatus("생성 중 (GPU)...");

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
            timeText.text = $"경과: {elapsed:F0}s | 토큰: {tokenCount}";
            yield return new WaitForSeconds(0.5f);
        }

        float total = Time.realtimeSinceStartup - t0;

        if (err != null)
        {
            SetStatus($"오류: {err.Message}");
            Debug.LogError($"[LlamaCppUI] Error: {err.Message}\n{err.StackTrace}");
        }
        else
        {
            string summary = $"완료: {total:F1}s | {tokenCount}토큰 | {tokenCount / Mathf.Max(total, 0.1f):F1} tok/s";
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
        // outputText (ScrollView 내부)와 statusText 둘 다에 표시 시도
        if (outputText != null) outputText.text = text;
    }

    private void OnDestroy()
    {
        _bridge?.Dispose();
    }
}
