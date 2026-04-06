# unity-android-ondevice-llm

A reference implementation for running on-device LLM inference with GPU acceleration on Unity Android.

Uses llama.cpp's Adreno OpenCL backend for Snapdragon GPU inference, bridged to Unity C# via a C wrapper + P/Invoke.

## Pipeline Architecture

```
[Unity C# (Game Logic)]
    │ P/Invoke (DllImport)
    ▼
[unity_bridge.c (C Wrapper)]
    │ llama.h C API
    ▼
[libllama.so + libggml-opencl.so (llama.cpp)]
    │ OpenCL Kernels
    ▼
[Adreno GPU (Snapdragon)]
    │
    ▼
[GGUF Model File]
```

## Benchmarks (Samsung Galaxy S24 Ultra, Snapdragon 8 Gen 3)

| Model | Size | Quant | tok/s | 150 tokens |
|-------|------|-------|-------|------------|
| **Qwen3-1.7B** | **1.8GB** | **Q8_0** | **16.6** | **9.1s** |
| Phi-4-mini (3.8B) | 3.8GB | Q8_0 | 9.0 | 16.8s |
| Phi-4-mini (3.8B) | 2.3GB | Q4_0 | 5.1 | 29.6s |
| Phi-4-mini (3.8B) | 3.2GB | Q6_K | 4.2 | 31.5s |

> On Adreno OpenCL, Q8_0 is the fastest. Q4_0/Q6_K are slower due to GPU dequantization overhead.

### Inference Engine Comparison (Phi-4-mini 3.8B)

| Engine | Backend | tok/s | Notes |
|--------|---------|-------|-------|
| ONNX Runtime | CPU | 0.21 | Baseline |
| ONNX Runtime + QNN | HTP (3/363 nodes) | 0.31 | INT4 ops unsupported |
| LiteRT-LM | GPU (OpenCL) | - | GPU memory exceeded |
| **llama.cpp** | **Adreno OpenCL** | **9.0** | **Qualcomm optimized** |

## Build Instructions

### 1. Prerequisites

- Android NDK (Unity's bundled NDK works)
- CMake, Ninja
- Python 3 (huggingface_hub)

### 2. OpenCL Setup

```bash
export ANDROID_NDK=/path/to/ndk
NDK_SYSROOT=$ANDROID_NDK/toolchains/llvm/prebuilt/*/sysroot

# OpenCL Headers
git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers
cp -r OpenCL-Headers/CL $NDK_SYSROOT/usr/include/

# OpenCL ICD Loader
git clone --depth 1 https://github.com/KhronosGroup/OpenCL-ICD-Loader
cd OpenCL-ICD-Loader && mkdir build && cd build
cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DOPENCL_ICD_LOADER_HEADERS_DIR=$NDK_SYSROOT/usr/include \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=28
ninja
cp libOpenCL.so $NDK_SYSROOT/usr/lib/aarch64-linux-android/
```

### 3. Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && mkdir build-android && cd build-android

cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_OPENCL=ON \
  -DGGML_OPENMP=OFF

ninja
```

> **Note:** If your project path contains apostrophes (`'`), the OpenCL kernel embedding script will fail. Use a symlink to work around this.

### 4. Build C Wrapper

```bash
CC=$ANDROID_NDK/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android28-clang

$CC -shared -o libunity_llama.so unity_bridge.c \
  -I llama.cpp/include -I llama.cpp/ggml/include \
  -L llama.cpp/build-android/bin \
  -lllama -lggml -lggml-base
```

### 5. Unity Integration

Copy to `Assets/Plugins/Android/libs/arm64-v8a/`:
```
libllama.so
libggml.so
libggml-base.so
libggml-cpu.so
libggml-opencl.so
libunity_llama.so
```

### 6. Model Deployment

```bash
# Recommended: Qwen3-1.7B Q8_0
pip install huggingface_hub
huggingface-cli download unsloth/Qwen3-1.7B-GGUF Qwen3-1.7B-Q8_0.gguf

# Push to device
adb push Qwen3-1.7B-Q8_0.gguf /data/local/tmp/model.gguf
```

## File Descriptions

| File | Description |
|------|-------------|
| `unity_bridge.c` | C wrapper that hides llama.h structs for safe P/Invoke |
| `Scripts/LlamaCppBridge.cs` | Unity C# P/Invoke bindings |
| `Scripts/LlamaCppTestUI.cs` | Test UI — button to generate, displays result |
| `GradleTemplates/launcherTemplate.gradle` | Injects libcdsprpc.so / libOpenCL.so into manifest |
| `history.md` | Full 4-day development log (including all failures) |

## Why a C Wrapper?

llama.cpp's C structs (`llama_model_params`, `llama_context_params`) contain pointers, enums, and bools that are difficult to marshal correctly with Unity's `[StructLayout]`. Incorrect layout causes SIGSEGV.

`unity_bridge.c` handles these structs internally and exposes only opaque pointers and primitive types:

```c
void* unity_llama_model_load(const char* path, int n_gpu_layers);
int   unity_llama_tokenize(void* vocab, const char* text, int* tokens, int n_max);
int   unity_llama_sample_greedy(void* ctx, void* vocab);
```

## Development Journey

See [history.md](history.md) for the full story — from ONNX Runtime (0.3 tok/s) through QNN HTP, LiteRT-LM, and finally llama.cpp OpenCL (16.6 tok/s). Each approach's failure and the lessons learned are documented.

## License

MIT
