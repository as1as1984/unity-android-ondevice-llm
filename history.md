# On-Device LLM RPG Development Log — Day 1

**Project:** 2D turn-based roguelike RPG with on-device LLM  
**Goal:** Run LLM inference on Android GPU without any server  
**Date:** 2026-04-03  
**Devices:** Mac (development), Samsung Galaxy S24 Ultra (testing)

---

## 0. Why This Tech Stack?

### Why On-Device LLM?
- **Primary goal: learning on-device AI.** The game is a vehicle for learning.
- Zero server cost — indie dev can ship globally without operational expenses
- Offline play — LLM generates content without network
- Auto-localization — LLM outputs in device language, no separate translation needed

### Why Phi-4-mini (3.8B)?
- **Official ONNX from Microsoft** — no conversion needed
- INT4 quantized version available — fits in mobile memory (4.9GB)
- 3.8B params — minimum viable size for structured JSON output
- Hypothesis: small models are sufficient for comic fantasy RPG dialogue/mob generation
- Easy to switch to Phi-3.5-mini (same architecture, more validated)

### Why ONNX Runtime?
- **Cross-platform** — Android, iOS, Windows, Mac
- Unity C# bindings exist (`Microsoft.ML.OnnxRuntime`)
- **asus4/onnxruntime-unity** package for easy Unity integration
- Hardware acceleration EP switching in one line (QNN/NNAPI/CoreML)
- IL2CPP compatible

### Why Unity?
- Ideal ecosystem for 2D roguelike RPG (tilemap, sprites, UI)
- Android/iOS cross-platform builds
- C# for both game logic and LLM inference (no Python bridge needed)

### Why Samsung Galaxy S24 Ultra (test device)?
- Snapdragon 8 Gen 3 — Hexagon NPU, QNN acceleration capable
- 12GB RAM — room for 4.9GB model
- Flagship performance ceiling — if it doesn't work here, it won't work anywhere

---

## 1. Environment Setup

### Unity Project
- Unity 6000.4.1f1, 2D URP template
- Android IL2CPP, Min SDK 31, Target SDK 35

### ONNX Runtime
- `com.github.asus4.onnxruntime` v0.4.4 (NPM scoped registry)
- `com.github.asus4.onnxruntime.unity` v0.4.4
- IL2CPP compatibility verified

### Model Download
- Phi-4-mini INT4 quantized ONNX (cpu_and_mobile variant)
- Total 4.9GB (`model.onnx` 52MB + `model.onnx.data` 4.9GB)
- Downloaded from Hugging Face

---

## 2. C# Tokenizer Implementation

Built a tiktoken-style BPE tokenizer for Phi-4-mini in C#.

### Implementation (`Phi4Tokenizer.cs`)
- Loaded vocab (200,029), merges (199,742), special tokens (12) from `tokenizer.json`
- GPT-2 byte-to-unicode conversion table
- BPE encoding/decoding with cache
- Special token splitting

### Failure 1: Merges Parsing Error
```
Newtonsoft.Json.Linq.JValue → JArray cast failure
```
- **Cause:** merges in tokenizer.json were `["tok1","tok2"]` arrays, not `"tok1 tok2"` strings
- **Fix:** Added `if (merges[i] is JArray pair)` branch to support both formats

---

## 3. LLM Inference Engine

### Implementation (`LlmGenerator.cs`)
- KV cache-based auto-regressive greedy decoding
- 32 layers, 8 KV heads, head_size 128
- Prefill (full prompt at once) → Decode (one token at a time)
- past_key_values / present tensor management

### Failure 2: DenseTensor Constructor Error
```
CS1503: DenseTensor<long>(seqLen, new[] {batch, seqLen})
```
- **Cause:** First argument should not be an int value
- **Fix:** `new DenseTensor<long>(new[] {batch, seqLen})`

### Failure 3: Model Path Error
- **Cause:** `../../..` (3 levels up) instead of `../..` (2 levels)
- **Fix:** Corrected `Path.Combine` levels

---

## 4. First Generation Tests (Unity Editor, Mac CPU)

### Test 1: Long Prompt + 512 Tokens
- Prompt tokens: 84
- **Result: Over 10 minutes → aborted**
- ~1 sec/token (Mac CPU), KV cache copy growing O(n)

### Test 2: Short Prompt + 150 Tokens
- ChatML format prompt (`<|system|>...<|user|>...<|assistant|>`)
- Minimized JSON structure (floor, mob, hp, atk only)
- **Result: 181 seconds (3 min), EOS at token 110, valid JSON!**

### Issues
1. **Speed:** 181 seconds — too slow even for loading screens
2. **Prompt misunderstanding:** Mob names were the player character name instead of actual mob names

---

## 5. Test UI Implementation

Built a UI scene for Android device testing.

### Implementation
- `LlmTestUI.cs`: Button click → model loading → generation → result display
- Real-time elapsed time + token count + tok/s display
- `LlmTestSceneBuilder.cs`: Editor script for automatic scene creation

### Failure 4: Korean Font Rendering
- **Cause:** TMP default font (LiberationSans) doesn't include Korean characters
- **Fix:** Created TMP Font Asset from NotoSansKR
- Custom Range: `32-126,44032-55203,12593-12686` (ASCII + Korean)
- **Additional failure:** Entered hex values (`AC00-D7A3`) first → TMP only accepts decimal → `FormatException`

### Failure 5: Input System Conflict
```
InvalidOperationException: You are trying to read Input using the UnityEngine.Input class,
but you have switched active Input handling to Input System package
```
- **Cause:** Needed `InputSystemUIInputModule` instead of `StandaloneInputModule`
- **Fix:** Auto-detect New Input System → switch module

### Failure 6: Button Click Not Responding
- **Cause:** `btn.onClick.AddListener()` is runtime-only, not serialized with scene
- **Fix:** Replaced with `UnityEventTools.AddPersistentListener()`

### Test 3: UI Generation (Unity Editor)
- **Result: 246 seconds (4 min), generation successful**

---

## 6. Android Build & Device Testing

### Failure 7: Gradle Build Failure — Model File Size
```
compressReleaseAssets FAILED
Required array size too large
```
- **Cause:** 5GB model in StreamingAssets → Java 2.1GB array limit exceeded
- **Attempt 1:** Renamed `Models` → `Models_SKIP` → Failed (Unity includes all StreamingAssets regardless of name)
- **Attempt 2:** Deleted Gradle cache (`Library/Bee/Android` 15GB) → Failed (model files persisted in cache)
- **Final fix:** Moved model folder completely outside Assets + deleted Gradle cache → Build succeeded (43s)

### Android Deployment
- APK: Built without model files (lightweight)
- Model: Transferred via `adb push` (4.9GB, 94s)

### Failure 8: Blank Screen on Launch
- **Cause:** SampleScene was first in Build Settings, LlmTestScene was missing
- **Fix:** Set LlmTestScene as scene index 0 → rebuilt

### Test 4: Samsung Galaxy S24 Ultra (CPU only)
- Model loading: ~6s
- Tokenizer loading: ~1s
- **Generation time: 523 seconds (8 min 43 sec) — 2.1x slower than Mac**

| Metric | Mac (Editor) | S24 Ultra (CPU only) |
|--------|-------------|---------------------|
| Model loading | ~6s | ~6s |
| Tokenizer | ~0.7s | ~1s |
| **Generation** | **246s** | **523s** |
| Speed comparison | Baseline | **2.1x slower** |

---

## 7. NPU Utilization Issue

### Current State
- Only NNAPI enabled → most operations fall back to CPU
- **QNN (Qualcomm Neural Network) EP not included** in asus4/onnxruntime package
- S24 Ultra's Snapdragon 8 Gen 3 Hexagon NPU completely unused

### Test 5: S24 Ultra + QNN HTP EP (Runtime Compilation)

**Setup:**
- Replaced `libonnxruntime.so` from `onnxruntime-android-qnn:1.24.3` AAR
- Added QNN runtime .so libraries (libQnnHtp, libQnnHtpV75, etc.)
- `opt.AppendExecutionProvider("QNN", { backend_type: "htp", htp_performance_mode: "burst" })`

**Build failures:**
1. `libonnxruntime.so` duplicate → conflict with asus4 AAR → resolved by replacing .so inside AAR
2. `16KB-aligned` warning (Android 15+) → ignored, build succeeded

**Result:**

| Metric | CPU only | QNN HTP (runtime) |
|--------|----------|------------------|
| Time | 523s | 490s |
| Tokens | 110 (EOS) | 150 (no EOS) |
| tok/s | 0.21 | 0.31 |
| Improvement | Baseline | **1.5x (negligible)** |

**Conclusion:** Runtime QNN compilation of ONNX models is ineffective. **Pre-converted QNN context binaries required.**

### Additional Analysis: INFO Logs Reveal Root Cause

```
Failed in loading stub: dlopen failed: library "libcdsprpc.so" not found
Failed to create transport for device, error: 4000
Failed to setup so cleaning up
```

**QNN HTP was not actually running.**
- `libcdsprpc.so` is a Qualcomm DSP RPC library (vendor partition)
- App sandbox cannot access vendor libraries → HTP init failed
- EP "registration" succeeded but backend setup failed → full CPU fallback

**Fix:** Declare `<uses-native-library android:name="libcdsprpc.so" android:required="false"/>` in AndroidManifest (Android 12+, no root required)

### Samsung SDK Investigation

| SDK | Status | Usable? |
|-----|--------|---------|
| Samsung Neural SDK | Discontinued for 3rd party | No |
| ENN SDK (Exynos) | Exynos only | No (S24 Ultra global = Snapdragon) |
| Galaxy AI | 1st party apps only | No |
| Samsung ONE | Exynos NPU only | No |

Samsung's own SDKs cannot access NPU on Snapdragon variant devices.

### Test 6: AndroidManifest libcdsprpc.so Access Attempt

**Serial failures:**
1. Declared inside `<application>` → Build succeeded, app invisible in launcher
2. Declared under `<manifest>` → AAPT error
3. Removed declaration → still invisible
4. Deleted custom AndroidManifest entirely → **app visible again**

**Conclusion:** Unity 6's custom AndroidManifest completely replaces the auto-generated one, breaking the launcher Activity.

### Test 7: Gradle Template Injection + HTP Confirmed

After multiple Gradle template approaches failed (manifestOutputDirectory removed from AGP, doLast not reflected in final APK, etc.), used apktool repackaging to confirm:

**HTP initialization succeeded:**
```
QnnDsp <W> Initializing HtpProvider ✅
QnnDsp <W> PrepareLibLoader Loading libQnnHtpPrepare.so ✅
```

**But the core problem:**
```
number of nodes in the graph: 363, number of nodes supported by QNN: 3
```
Only **3 out of 363 nodes** ran on QNN HTP. The rest fell back to CPU.
INT4 block quantization (MatMulNBits) operators are unsupported by QNN HTP.

**Final result:** Effectively identical to CPU-only. Runtime QNN compilation is useless for INT4 quantized models.

### Android Scoped Storage Issue

Files pushed via `adb push` to app's data directory were invisible to the app.
- **Cause:** After `adb uninstall` + reinstall, directory ownership changes (scoped storage)
- **Fix:** Launch app first (creates directory) → close → adb push → relaunch

---

## 8. LiteRT-LM Integration Attempt (Day 2)

### Model Preparation
- `litert-community/Phi-4-mini-instruct` from HuggingFace
- `ekv4096` (.litertlm bundle, 3.9GB) — KV cache 4096 tokens
- `ekv1280` (.task + .tflite, 3.9GB each) — KV cache 1280 tokens, legacy format

### Library Setup
- `litertlm-android-0.10.0.aar` (Google Maven, 18MB) — includes JNI engine
- GPU accelerator .so files from Git LFS
- Unity C# bridge using `AndroidJavaObject` to call Kotlin API

### Bazel Native Build → Failed
```
Target //c:engine is incompatible and cannot be built
```
Rust dependency (cxxbridge) doesn't support Android ARM64 cross-compilation. → Switched to Kotlin AAR approach.

### Issue: JNI Thread Restriction
- `ThreadPool` calling `AndroidJavaObject` → NullReferenceException
- **Cause:** JNI calls must be on the main thread
- **Fix:** Initialize from coroutine on main thread

### Test 8: ekv4096 (.litertlm) + GPU Backend
```
Requested allocation size - 18446744071872970752 bytes
Max allocation size for this GPU - 1073741824 bytes
```
**GPU memory 1GB limit exceeded.** int32 overflow. Unity occupying GPU leaves insufficient VRAM.

### Test 9: ekv1280 (.task) + GPU Backend
```
Failed to create engine: INTERNAL ERROR at llm_litert_compiled_model_executor.cc:1955
```
No GPU allocation error, but engine creation failed. Suspected `.task` format incompatibility with AAR v0.10.0.

### Day 2 Lessons
1. **Unity + GPU LLM conflict** — Unity rendering and LLM inference compete for the same GPU VRAM
2. **LiteRT-LM is hard to integrate with Unity** — Bazel build failure, JNI thread constraints, model format issues
3. **Android Scoped Storage is hostile to adb testing** — can't directly push to app data folder
4. **On-device AI reality** — even with GPU acceleration, coexistence with Unity games is challenging

---

## 9. llama.cpp + Adreno OpenCL Attempt (Day 2 continued)

### Why llama.cpp?
- Qualcomm officially contributed Adreno-optimized OpenCL kernels
- Expected 15-30 tok/s on S24 Ultra (Adreno 750) with Q4_0
- CMake build — much easier than Bazel
- C API → direct Unity P/Invoke integration

### Build Process

1. **OpenCL Headers** — Copied KhronosGroup/OpenCL-Headers to NDK sysroot ✅
2. **OpenCL ICD Loader** — Cross-compiled for Android ARM64 ✅
3. **First llama.cpp build** → Failed due to apostrophe in project path (`Demon Lord's Castle`)
   ```
   /bin/sh: -c: line 0: unexpected EOF while looking for matching `'
   ```
4. **Symlink workaround** (`/tmp/llama-cpp`) → **Build succeeded!** 377/377 targets

### Model
- `unsloth/Phi-4-mini-instruct-GGUF` Q8_0 (3.8GB)
- Q4_0 not in repo, Q4_K_M unsupported by OpenCL → chose Q8_0

### Test Result: Library Load Failure

```
dlopen failed: library "libomp.so" not found:
needed by libggml-cpu.so
```

**Cause:** `libggml-cpu.so` depends on OpenMP (`libomp.so`) which was not included in the APK.

**Fix:** Rebuild with `-DGGML_OPENMP=OFF`

---

## 10. llama.cpp OpenCL Success! (Day 3: 2026-04-05)

### libomp.so Fix
- Rebuilt with `-DGGML_OPENMP=OFF` → dependency removed

### P/Invoke Struct Crash Fix
- Direct `LlamaModelParams` struct marshaling → SIGSEGV (struct layout mismatch)
- **Fix:** Created C wrapper functions (`unity_bridge.c`) — handles complex structs internally, exposes simple interface
- 8 wrapper functions: `unity_llama_model_load(path, n_gpu_layers)`, etc.

### Button Event Fix
- Inspector-linked OnClick not reflected in build (persistent listener serialization issue)
- **Fix:** Runtime registration via `generateButton.onClick.AddListener(OnGenerateClicked)` in `Start()`

### Test 10: S24 Ultra + llama.cpp OpenCL (Adreno 750) 🎉

| Metric | Result |
|--------|--------|
| Model | Phi-4-mini Q8_0 (3.8GB GGUF) |
| Model loading | ~23s |
| Prompt tokens | 100 |
| **Generation time** | **16.8s** |
| **Generated tokens** | **150** |
| **tok/s** | **8.9** |
| GPU | Adreno 750 (OpenCL) |

### Full Benchmark Comparison

| Engine | tok/s | 150 tokens | vs ONNX |
|--------|-------|------------|---------|
| ONNX Runtime CPU (S24) | 0.21 | 523s | Baseline |
| ONNX Runtime QNN (S24) | 0.31 | 490s | 1.5x |
| ONNX Runtime CPU (Mac) | 0.45 | 246s | 2.1x |
| **llama.cpp OpenCL (S24)** | **8.9** | **16.8s** | **42x** |

**Game-viable on-device AI speed achieved!**

### Key Lessons
1. **ONNX Runtime + QNN is useless for INT4 models** — only 3/363 nodes on NPU
2. **LiteRT-LM conflicts with Unity GPU** — shared GPU VRAM causes memory exhaustion
3. **llama.cpp OpenCL is the answer** — Qualcomm official Adreno optimization, easy CMake build
4. **C wrapper is essential for P/Invoke** — never marshal complex C structs directly
5. **Q8_0 is optimal for Adreno OpenCL** — Q4_0 and Q6_K are actually slower

### Quantization Benchmarks on Adreno 750 (Day 4: 2026-04-06)

| Quantization | Model Size | Load Time | tok/s | Notes |
|-------------|-----------|-----------|-------|-------|
| **Q8_0** | 3.8GB | 24.5s | **9.0** | GPU compute optimized |
| Q6_K | 3.2GB | 24.5s | 4.2 | GPU dequantization overhead |
| Q4_0 | 2.3GB | 17.6s | 5.1 | Faster loading but slower compute |

**Conclusion:** Q8_0 is fastest on Adreno OpenCL. Q4_0/Q6_K introduce GPU dequantization overhead.

### Model Size Comparison: Qwen3-1.7B vs Phi-4-mini (Day 4)

After confirming Q8_0 is optimal, tried switching to a smaller model.

**Qwen3-1.7B Q8_0 (1.8GB)** test results:

| | Phi-4-mini (3.8B) | Qwen3-1.7B (1.7B) |
|---|---|---|
| Model size | 3.8GB | **1.8GB** |
| Load time | 24.5s | **14.4s** |
| Speed | 9.0 tok/s | **16.6 tok/s** |
| 150 tokens | 16.8s | **9.1s** |
| Output quality | "monster_name" (literal) | **"rabbit"** (actual name) ✅ |
| JSON structure | Valid | Valid ✅ |

**Qwen3-1.7B adopted as default model.** Superior in every metric:
- Speed **1.8x** faster (9.0 → 16.6 tok/s)
- Model size **53% smaller** (3.8GB → 1.8GB)
- Output quality **better** (generates actual mob names)
- Load time **41% shorter** (24.5s → 14.4s)

---

## Unity On-Device LLM Pipeline Architecture

```
[Unity C# (Game Logic)]
    │
    │ P/Invoke
    ▼
[unity_bridge.c (C Wrapper)]
    │
    │ llama.h C API
    ▼
[libllama.so + libggml-opencl.so]
    │
    │ OpenCL Kernels
    ▼
[Adreno GPU (Snapdragon)]
    │
    ▼
[GGUF Model File (/data/local/tmp/)]
```

**Build Pipeline:**
1. **llama.cpp** — NDK + CMake cross-compile for Android ARM64 (`-DGGML_OPENCL=ON -DGGML_OPENMP=OFF`)
2. **unity_bridge.c** — Wraps llama.h complex structs into simple C functions
3. **NDK clang** — Compiles unity_bridge.c into `libunity_llama.so`
4. **.so placement** — `libllama.so`, `libggml*.so`, `libunity_llama.so` → `Assets/Plugins/Android/libs/arm64-v8a/`
5. **C# P/Invoke** — `[DllImport("unity_llama")]` for direct Unity calls

**Core wrapper functions (unity_bridge.c):**
```c
unity_llama_backend_init()          // Initialize backend
unity_llama_model_load(path, gpu)   // Load model (GPU layer offload)
unity_llama_context_create(model)   // Create context
unity_llama_tokenize(vocab, text)   // Text → tokens
unity_llama_decode_batch(ctx, tok)  // Decode tokens
unity_llama_sample_greedy(ctx)      // Sample next token
unity_llama_token_to_text(vocab)    // Token → text
```

**Why this architecture:**
- Direct marshaling of llama.h structs from Unity C# causes memory layout mismatches → crash
- C wrapper handles structs internally, exposes only opaque pointers → stable
- OpenCL loads device system library (`libOpenCL.so`) at runtime → no bundling needed
- Model placed outside APK (4GB+ APK impossible) → `adb push` or in-app download

## Final Stack

| Component | Choice |
|-----------|--------|
| Inference engine | llama.cpp (OpenCL, Adreno optimized) |
| Model | Qwen3-1.7B Q8_0 (1.8GB) |
| GPU backend | OpenCL (Adreno 750) |
| Performance | 16.6 tok/s, 9.1s / 150 tokens |
| Unity integration | C wrapper (unity_bridge.c) + P/Invoke |
