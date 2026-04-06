using System;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

/// <summary>
/// P/Invoke bridge for llama.cpp via simplified C wrapper (unity_bridge.c).
/// </summary>
public class LlamaCppBridge : IDisposable
{
    const string DLL = "unity_llama";

    [DllImport(DLL)] static extern void unity_llama_backend_init();
    [DllImport(DLL)] static extern void unity_llama_backend_free();
    [DllImport(DLL)] static extern IntPtr unity_llama_model_load(string path, int n_gpu_layers);
    [DllImport(DLL)] static extern void unity_llama_model_free(IntPtr model);
    [DllImport(DLL)] static extern IntPtr unity_llama_context_create(IntPtr model, int n_ctx, int n_batch);
    [DllImport(DLL)] static extern void unity_llama_context_free(IntPtr ctx);
    [DllImport(DLL)] static extern IntPtr unity_llama_get_vocab(IntPtr model);
    [DllImport(DLL)] static extern int unity_llama_tokenize(IntPtr vocab, string text, int[] tokens, int n_max);
    [DllImport(DLL)] static extern int unity_llama_decode_batch(IntPtr ctx, int[] tokens, int n_tokens);
    [DllImport(DLL)] static extern int unity_llama_sample_greedy(IntPtr ctx, IntPtr vocab);
    [DllImport(DLL)] static extern int unity_llama_token_to_text(IntPtr vocab, int token, byte[] buf, int buf_size);
    [DllImport(DLL)] static extern int unity_llama_vocab_eos(IntPtr vocab);

    private IntPtr _model;
    private IntPtr _ctx;
    private IntPtr _vocab;
    private bool _initialized;

    public void Initialize(string modelPath, int nGpuLayers = 99, int nCtx = 1280)
    {
        Debug.Log("[LlamaCpp] Backend init...");
        unity_llama_backend_init();

        Debug.Log($"[LlamaCpp] Loading model: {modelPath}");
        _model = unity_llama_model_load(modelPath, nGpuLayers);
        if (_model == IntPtr.Zero)
            throw new Exception("Failed to load model");

        Debug.Log("[LlamaCpp] Model loaded. Creating context...");
        _ctx = unity_llama_context_create(_model, nCtx, 512);
        if (_ctx == IntPtr.Zero)
            throw new Exception("Failed to create context");

        _vocab = unity_llama_get_vocab(_model);

        _initialized = true;
        Debug.Log("[LlamaCpp] Initialized successfully");
    }

    public string Generate(string prompt, int maxTokens = 150, Action<string> onToken = null)
    {
        if (!_initialized) throw new Exception("Not initialized");

        int[] tokens = new int[2048];
        int nTokens = unity_llama_tokenize(_vocab, prompt, tokens, tokens.Length);
        Debug.Log($"[LlamaCpp] Prompt tokens: {nTokens}");

        int[] promptTokens = new int[nTokens];
        Array.Copy(tokens, promptTokens, nTokens);
        int ret = unity_llama_decode_batch(_ctx, promptTokens, nTokens);
        if (ret != 0) throw new Exception($"Decode failed: {ret}");

        int eosToken = unity_llama_vocab_eos(_vocab);
        var sb = new StringBuilder();
        byte[] buf = new byte[128];

        for (int i = 0; i < maxTokens; i++)
        {
            int newToken = unity_llama_sample_greedy(_ctx, _vocab);

            if (newToken == eosToken)
            {
                Debug.Log($"[LlamaCpp] EOS at step {i}");
                break;
            }

            int len = unity_llama_token_to_text(_vocab, newToken, buf, buf.Length);
            if (len > 0)
            {
                string piece = Encoding.UTF8.GetString(buf, 0, len);
                sb.Append(piece);
                onToken?.Invoke(piece);
            }

            int[] next = new int[] { newToken };
            unity_llama_decode_batch(_ctx, next, 1);
        }

        return sb.ToString();
    }

    public void Dispose()
    {
        if (_ctx != IntPtr.Zero) { unity_llama_context_free(_ctx); _ctx = IntPtr.Zero; }
        if (_model != IntPtr.Zero) { unity_llama_model_free(_model); _model = IntPtr.Zero; }
        unity_llama_backend_free();
        _initialized = false;
    }
}
