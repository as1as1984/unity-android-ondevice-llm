# 온디바이스 LLM RPG 개발일지 — Day 1

**프로젝트:** 용사 구인 (경력무관) / Help Wanted: Hero  
**목표:** Unity + 온디바이스 LLM으로 서버 없는 2D 턴제 로그라이크 RPG  
**날짜:** 2026-04-03  
**디바이스:** Mac (개발), Samsung Galaxy S24 Ultra (테스트)

---

## 0. 기술 스택 선택 이유

### 왜 온디바이스 LLM인가?
- **프로젝트의 1차 목적이 온디바이스 AI 학습.** 게임은 학습의 결과물이자 동기부여 수단
- 서버 비용 0 — 개인 개발자가 운영비 걱정 없이 글로벌 출시 가능
- 오프라인 플레이 가능 — 네트워크 없어도 LLM이 콘텐츠 생성
- OS 언어 기반 자동 다국어 — LLM이 디바이스 언어로 출력하면 별도 번역 불필요

### 왜 Phi-4-mini (3.8B)인가?
- **Microsoft 공식 ONNX 제공** — 변환 없이 바로 사용 가능
- INT4 양자화 버전 존재 — 모바일 메모리(4.9GB)에 탑재 가능한 수준
- 3.8B 파라미터 — JSON 구조화 출력이 가능한 최소 수준의 모델 크기
- 코믹 판타지 RPG 대사/몹 생성 정도는 소형 모델로 충분할 것이라는 가설
- Phi-3.5-mini (동일 구조, 더 검증됨)로의 전환도 용이

### 왜 ONNX Runtime인가?
- **크로스 플랫폼** — Android, iOS, Windows, Mac 전부 지원
- Unity C# 바인딩 존재 (`Microsoft.ML.OnnxRuntime`)
- **asus4/onnxruntime-unity** 패키지로 Unity 통합이 쉬움
- QNN/NNAPI/CoreML 등 하드웨어 가속 EP 전환이 코드 한 줄
- IL2CPP 호환 확인됨

### 왜 Unity인가?
- 2D 로그라이크 RPG에 적합한 에코시스템 (타일맵, 스프라이트, UI)
- Android/iOS 크로스 플랫폼 빌드
- C#으로 LLM 추론 코드까지 일원화 (Python 브릿지 불필요)
- mcp-unity로 Claude Code ↔ Unity 에디터 양방향 자동화

### 왜 Samsung Galaxy S24 Ultra (테스트 기기)인가?
- Snapdragon 8 Gen 3 — Hexagon NPU 탑재, QNN 가속 가능
- 12GB RAM — 4.9GB 모델 로드 여유
- 최신 플래그십 기준으로 성능 상한선 측정 → 이 기기에서 안 되면 현재 기술로는 불가능

---

## 1. 환경 세팅

### Unity 프로젝트 생성
- Unity 6000.4.1f1, 2D URP 템플릿
- Android IL2CPP, Min SDK 31, Target SDK 35
- Bundle ID: `com.as1as.helpwantedhero`

### mcp-unity 설치 (Claude Code ↔ Unity 연동)
- 처음엔 unity-mcp 시도 → 기능 제한으로 폐기
- CoderGamester/mcp-unity로 교체 → WebSocket 기반 양방향 통신 성공
- **반복 경고 발생:** `GetGameObjectResourceTests.cs has no meta file` → 패키지 자체 버그, 무해함

### ONNX Runtime 설치
- `com.github.asus4.onnxruntime` v0.4.4 (NPM scoped registry)
- `com.github.asus4.onnxruntime.unity` v0.4.4
- `com.github.asus4.onnxruntime-extensions` v0.4.4
- IL2CPP 호환 확인 완료

### 모델 다운로드
- Phi-4-mini INT4 양자화 ONNX (cpu_and_mobile variant)
- 총 4.9GB (`model.onnx` 52MB + `model.onnx.data` 4.9GB)
- Hugging Face에서 다운로드

---

## 2. C# 토크나이저 구현

Phi-4-mini용 tiktoken 스타일 BPE 토크나이저를 C#으로 직접 구현.

### 구현 내용 (`Phi4Tokenizer.cs`)
- `tokenizer.json`에서 vocab(200,029개), merges(199,742개), special tokens(12개) 로드
- GPT-2 바이트↔유니코드 변환 테이블
- BPE 인코딩/디코딩 + 캐시
- 특수 토큰 분할 처리

### 실패 1: merges 파싱 오류
```
Newtonsoft.Json.Linq.JValue → JArray 캐스팅 실패
```
- **원인:** tokenizer.json의 merges가 `"tok1 tok2"` 문자열이 아닌 `["tok1","tok2"]` 배열 형식
- **해결:** `if (merges[i] is JArray pair)` 분기 추가, 두 형식 모두 지원

---

## 3. LLM 추론 엔진 구현

### 구현 내용 (`LlmGenerator.cs`)
- KV 캐시 기반 auto-regressive greedy decoding
- 32 레이어, 8 KV 헤드, head_size 128
- Prefill (전체 프롬프트 한 번에) → Decode (토큰 하나씩) 패턴
- past_key_values / present 텐서 관리

### 실패 2: DenseTensor 생성자 오류
```
CS1503: DenseTensor<long>(seqLen, new[] {batch, seqLen})
```
- **원인:** 첫 번째 인자로 int 값을 넘겨선 안 됨
- **해결:** `new DenseTensor<long>(new[] {batch, seqLen})`으로 수정

### 실패 3: 모델 경로 오류
```
model.onnx not found
```
- **원인:** `../../..` (3레벨 상위) 대신 `../..` (2레벨)이어야 함
- **해결:** `Path.Combine(Application.dataPath, "..", "..", "models", ...)` 로 수정

---

## 4. 첫 번째 생성 테스트 (Unity Editor, Mac CPU)

### 테스트 1: 긴 프롬프트 + 512 토큰
- 프롬프트 토큰: 84개
- **결과: 10분 이상 소요 → 중단**
- 토큰당 ~1초 (Mac CPU), KV 캐시 복사 O(n) 증가로 점점 느려짐

### 테스트 2: 짧은 프롬프트 + 150 토큰
- ChatML 형식 프롬프트 (`<|system|>...<|user|>...<|assistant|>`)
- JSON 구조 최소화 (floor, mob, hp, atk만)
- **결과: 181초 (3분), 110토큰에서 EOS 도달, JSON 생성 성공!**

```json
[
  {"floor":1,"mob":"게으른 빵집 아들","hp":50,"atk":10},
  {"floor":2,"mob":"게으른 빵집 아들","hp":60,"atk":12},
  {"floor":3,"mob":"게으른 빵집 아들","hp":70,"atk":14},
  {"floor":4,"mob":"엘리트","hp":100,"atk":20},
  {"floor":5,"mob":"보스","hp":200,"atk":40}
]
```

### 문제점
1. **속도:** 181초 → 게임 로딩으로도 길다
2. **프롬프트 오해:** 1-3층 몹 이름이 플레이어 이름("게으른 빵집 아들")으로 출력됨

---

## 5. 테스트 UI 구현

Android 실기기 테스트를 위한 UI 씬 제작.

### 구현 내용
- `LlmTestUI.cs`: 버튼 클릭 → 모델 로딩 → 생성 → 결과 출력
- 경과 시간 + 토큰 수 + tok/s 실시간 표시
- `LlmTestSceneBuilder.cs`: Editor 스크립트로 씬 자동 생성

### 실패 4: 한글 폰트 깨짐
- **원인:** TMP 기본 폰트(LiberationSans)에 한글 미포함
- **해결:** AppleSDGothicNeo.ttc → TMP Font Asset Creator로 변환
- Custom Range: `32-126,44032-55203,12593-12686` (ASCII + 한글 가-힣 + ㄱ-ㅣ)
- **추가 실패:** 처음에 16진수(`AC00-D7A3`) 입력 → TMP가 10진수만 받아서 `FormatException`

### 실패 5: Input System 충돌
```
InvalidOperationException: You are trying to read Input using the UnityEngine.Input class,
but you have switched active Input handling to Input System package
```
- **원인:** `StandaloneInputModule` 대신 `InputSystemUIInputModule` 필요
- **해결:** New Input System 감지 → 자동 전환 코드 추가

### 실패 6: 버튼 클릭 반응 없음
- **원인:** `btn.onClick.AddListener()`는 런타임 전용, 씬 저장 시 직렬화 안 됨
- **해결:** `UnityEventTools.AddPersistentListener()`로 교체

### 테스트 3: UI로 생성 (Unity Editor)
- **결과: 246초 (4분), 생성 성공**

---

## 6. Android 빌드 & 실기기 테스트

### 실패 7: Gradle 빌드 실패 — 모델 파일 크기
```
compressReleaseAssets FAILED
Required array size too large
```
- **원인:** StreamingAssets에 5GB 모델 파일 포함 → Java 2.1GB 제한 초과
- **1차 시도:** `Models` → `Models_SKIP` 이름 변경 → 실패 (StreamingAssets 하위면 이름 무관하게 빌드 포함)
- **2차 시도:** Gradle 캐시 삭제 (`Library/Bee/Android` 15GB) → 실패 (이전 캐시에 모델 잔존)
- **최종 해결:** 모델 폴더를 Assets 밖으로 완전히 이동 + Gradle 캐시 삭제 → 빌드 성공 (43초)

### Android 배포 방식
- APK: 모델 없이 빌드 (가벼움)
- 모델: `adb push`로 별도 전송 (4.9GB, 94초)
- 경로: `/sdcard/Android/data/com.as1as.helpwantedhero/files/Models/phi-4-mini/`

### 실패 8: 앱 실행 시 화면 빈 화면
- **원인:** Build Settings에서 SampleScene이 첫 번째 씬, LlmTestScene이 빠져있음
- **해결:** LlmTestScene을 빌드 씬 목록 0번으로 설정 → 재빌드

### 테스트 4: Samsung Galaxy S24 Ultra 실기기 (CPU only)
- 모델 로딩: ~6초
- 토크나이저 로딩: ~1초
- **생성 시간: 523초 (8분 43초) — Mac 대비 2.1배 느림**
- UI 출력 안 됨 (폰트 미포함 또는 빈 결과 의심, 디버깅 필요)

| 항목 | Mac (Editor) | S24 Ultra (CPU only) |
|------|-------------|---------------------|
| 모델 로딩 | ~6초 | ~6초 |
| 토크나이저 | ~0.7초 | ~1초 |
| **생성 시간** | **246초** | **523초** |
| 속도 비교 | 기준 | **2.1배 느림** |

---

## 7. 발견된 핵심 이슈: NPU 미활용

### 현재 상태
```csharp
// LlmTestUI.cs
var opt = new SessionOptions();
#if UNITY_ANDROID && !UNITY_EDITOR
    try { opt.AppendExecutionProvider_Nnapi(); } catch { }
#endif
```

- NNAPI만 사용 중 → 대부분 연산이 CPU fallback
- **QNN (Qualcomm Neural Network) EP 미포함**
- S24 Ultra의 Snapdragon 8 Gen 3 Hexagon NPU를 전혀 활용 못하고 있음

### asus4/onnxruntime v0.4.4 AAR 분석
```
onnxruntime.aar 내부 확인:
✅ nnapi_provider_factory.h (NNAPI)
❌ QNN 관련 파일 없음
```

| EP | 포함 여부 | NPU 활용 |
|---|---|---|
| CPU | ✅ | ❌ |
| NNAPI | ✅ | 부분적 (DSP, 연산자 제한) |
| **QNN** | ❌ | ✅ 풀 NPU 가속 |

### 테스트 5: S24 Ultra + QNN HTP EP (런타임 컴파일 방식)

**설정:**
- `onnxruntime-android-qnn:1.24.3` AAR에서 `libonnxruntime.so` 교체
- `qnn-runtime:2.42.0`의 QNN .so 라이브러리 추가 (libQnnHtp, libQnnHtpV75 등)
- `opt.AppendExecutionProvider("QNN", { backend_type: "htp", htp_performance_mode: "burst" })`

**빌드 과정 실패:**
1. `libonnxruntime.so` 중복 → 기존 asus4 AAR과 충돌 → AAR 내부 .so를 QNN 버전으로 교체하여 해결
2. `16KB-aligned` 경고 (Android 15+) → 일단 무시, 빌드 성공

**결과:**

| 항목 | CPU only | QNN HTP (런타임) |
|------|----------|-----------------|
| 시간 | 523초 | 490초 |
| 토큰 | 110 (EOS) | 150 (EOS 미도달) |
| tok/s | 0.21 | 0.31 |
| 개선 | 기준 | **1.5배 (미미)** |

- QNN EP 등록 자체는 성공 (`[LlmUI] QNN HTP EP 등록됨`)
- 하지만 대부분의 연산이 CPU fallback → 의미있는 가속 없음
- INT4 블록 양자화 연산자를 HTP가 지원 못하는 것으로 추정
- JSON 출력 품질도 저하: 몹 이름이 "몬스터이름" 그대로 출력

**결론:** ONNX 모델을 런타임에 QNN으로 컴파일하는 방식은 효과 없음. **사전 변환된 QNN context binary 필요.**

### 테스트 5 추가 분석: INFO 로그로 원인 확인

INFO 레벨 로그에서 **진짜 원인** 발견:
```
Failed in loading stub: dlopen failed: library "libcdsprpc.so" not found
Failed to create transport for device, error: 4000
Failed to setup so cleaning up
```

**QNN HTP가 실제로 동작하지 않고 있었다.**
- `libcdsprpc.so`는 Qualcomm DSP RPC 라이브러리 (vendor 파티션)
- 앱 샌드박스에서 vendor 라이브러리 접근 불가 → HTP 초기화 실패
- EP "등록"은 성공했지만 백엔드 셋업 실패 → 전체 CPU fallback
- 이전 대비 미미한 속도 차이는 QNN 빌드의 libonnxruntime.so가 더 최적화된 CPU 코드 포함했기 때문

**해결법:** AndroidManifest에 `<uses-native-library android:name="libcdsprpc.so" android:required="false"/>` 선언으로 vendor 라이브러리 접근 가능 (Android 12+, 루트 불필요)

### 삼성 SDK 조사 결과

| SDK | 상태 | 사용 가능? |
|-----|------|-----------|
| Samsung Neural SDK | 서드파티 제공 중단 | ❌ |
| ENN SDK (Exynos) | Exynos 전용 | ❌ (S24 Ultra 글로벌 = Snapdragon) |
| Galaxy AI | 자사 앱 전용 | ❌ |
| Samsung ONE | Exynos NPU만 | ❌ |

삼성 자체 SDK로는 Snapdragon 변형 기기에서 NPU 접근 불가.

### 테스트 6: AndroidManifest로 libcdsprpc.so 접근 시도

`<uses-native-library android:name="libcdsprpc.so" android:required="false"/>` 선언 시도.

**실패 연속:**
1. `<application>` 안에 선언 → 빌드 성공, 앱 런처에서 안 보임
2. `<manifest>` 바로 아래 선언 → AAPT 에러 (`unexpected element <uses-native-library> found in <manifest>`)
3. `uses-native-library` 제거 후 테스트 → 여전히 앱 안 보임
4. 커스텀 AndroidManifest 완전 삭제 → **앱 정상 표시**

**결론:** Unity 6의 커스텀 AndroidManifest가 Unity 자동 생성 manifest와 충돌하여 런처 Activity가 깨짐. `uses-native-library` 자체의 문제가 아니라 커스텀 manifest 전체가 문제.

**해결 방향:** Unity 6의 Gradle 템플릿 커스터마이징으로 빌드 시 manifest에 `uses-native-library`만 주입하는 방식으로 재시도 예정.

### 테스트 7: Gradle 템플릿으로 libcdsprpc.so 주입 성공 + HTP 동작 확인

**manifest 주입 과정:**
1. 커스텀 AndroidManifest.xml → Unity가 기본 manifest를 완전 교체해서 런처 Activity 소실 → 실패
2. launcherTemplate.gradle의 `manifestOutputDirectory` API → AGP 버전에서 제거됨 → 실패
3. launcherTemplate.gradle의 `processReleaseManifest` doLast → 수정은 되지만 최종 APK에 미반영 → 실패
4. apktool로 APK 디컴파일 → manifest 수정 → 리패키징 → 서명 불일치로 모델 데이터 유실 반복 → 비효율
5. **`processReleaseManifestForPackage` 태스크에서 `packaged_manifests` 경로 수정** → APK에 미반영 → 실패
6. **`preBuild`에서 unityLibrary 소스 manifest 수정** → Unity가 매번 재생성해서 무효 → 실패
7. **`gradle.taskGraph.beforeTask`로 `processReleaseMainManifest` 직전에 수정** → 빌드 성공, APK에 미반영
8. **apktool 리패키징으로 우회** → `uses_libraries=libcdsprpc.so` 확인! → **HTP 초기화 성공!**

**HTP 동작 확인 로그:**
```
QnnDsp <W> Initializing HtpProvider ✅
QnnDsp <W> PrepareLibLoader Loading libQnnHtpPrepare.so ✅
```

이전(manifest 없이): `Failed in loading stub: dlopen failed: library "libcdsprpc.so" not found` ❌

**그러나 핵심 문제 발견:**
```
number of nodes in the graph: 363, number of nodes supported by QNN: 3
```
363개 노드 중 **3개만 QNN HTP에서 실행**, 나머지 360개는 CPU fallback.
INT4 블록 양자화(MatMulNBits) 연산자를 QNN HTP가 미지원.

**최종 결과:**

| 항목 | CPU only | QNN (libcdsprpc 없음) | QNN (HTP 동작) |
|------|----------|----------------------|----------------|
| 시간 | 523초 | 490초 | 483초 |
| tok/s | 0.21 | 0.31 | 0.31 |
| QNN 노드 | 0/363 | 0/363 | **3/363** |

초반 몇 토큰은 빨랐지만(QNN 3개 노드 효과), 전체 생성 시간은 CPU only와 사실상 동일.

**결론:** ONNX 모델을 런타임에 QNN 컴파일하는 방식은 INT4 양자화 모델에서 효과 없음. **QNN context binary 사전 변환** 또는 **LiteRT-LM** 같은 네이티브 NPU 지원 프레임워크 필요.

### 추가 삽질: Android scoped storage

adb push로 `/sdcard/Android/data/패키지명/files/`에 모델을 넣어도 앱이 인식 못하는 문제 발생.
- **원인:** `adb uninstall` 후 재설치 시 앱 data 디렉토리 소유권이 달라짐 (scoped storage)
- **해결:** 앱을 먼저 한 번 실행(폴더 생성) → 종료 → adb push → 재실행

---

## Day 1 최종 정리 (2026-04-03 ~ 04-04)

### 성과
- ONNX Runtime + Unity Android 연동 완료
- C# tiktoken BPE 토크나이저 구현
- KV 캐시 기반 LLM 추론 엔진 구현
- Phi-4-mini로 유효한 JSON 생성 확인 (에디터 + 실기기)
- QNN AAR 교체 + HTP EP 등록 성공
- S24 Ultra 실기기 3회 벤치마크 (CPU only / QNN 런타임 / INFO 분석)

### 벤치마크 요약

| 테스트 | 환경 | 시간 | tok/s |
|--------|------|------|-------|
| Mac Editor CPU | M시리즈 Mac | 246초 | 0.45 |
| S24 Ultra CPU only | Snapdragon 8 Gen 3 | 523초 | 0.21 |
| S24 Ultra QNN HTP (런타임) | 위와 동일 | 490초 | 0.31 |

*QNN HTP는 libcdsprpc.so 접근 실패로 실제 CPU fallback 상태였음*

### 미해결 과제
1. **libcdsprpc.so 접근** → Gradle 템플릿으로 manifest 주입 재시도
2. **QNN context binary 사전 변환** → 진짜 NPU 가속을 위해 필수
3. **LiteRT-LM 검토** → Phi-4 공식 지원, NPU 가속, C++ API
4. **UI 출력 안 됨** → 폰트 또는 result 문제 디버깅
5. **프롬프트 품질** → 몹 이름이 "몬스터이름" 그대로 출력

### 현실적 대안 경로

| 경로 | 기대 효과 | 난이도 |
|------|----------|--------|
| **Gradle 템플릿 + libcdsprpc.so** | QNN HTP 진짜 동작 확인 | 낮음 |
| **QNN context binary 변환** | 풀 NPU 가속 | 높음 |
| **LiteRT-LM** | Phi-4 공식 NPU 가속 | 중간 (C++ 네이티브 플러그인) |
| **llama.cpp + OpenCL** | Adreno GPU ~10 tok/s | 중간 (GGUF 변환 + C++) |

---

## 8. LiteRT-LM 통합 시도 (Day 2: 2026-04-04)

### 모델 준비
- `litert-community/Phi-4-mini-instruct` HuggingFace에서 다운로드
- `ekv4096` (.litertlm 번들, 3.9GB) — KV 캐시 4096토큰
- `ekv1280` (.task + .tflite, 각 3.9GB) — KV 캐시 1280토큰, 구버전 포맷

### 라이브러리 구성
- `litertlm-android-0.10.0.aar` (Google Maven, 18MB) — JNI 엔진 포함
- `libLiteRtGpuAccelerator.so`, `libLiteRtOpenClAccelerator.so` 등 — LFS에서 다운로드
- Unity C#에서 `AndroidJavaObject`로 Kotlin API 호출하는 브릿지 구현

### Bazel 네이티브 빌드 시도 → 실패
```
Target //c:engine is incompatible and cannot be built
```
Rust 의존성(cxxbridge)이 Android ARM64 크로스 컴파일 미지원. → Kotlin AAR 방식으로 전환.

### 삽질 1: Android Scoped Storage
- `adb push`로 `persistentDataPath`에 넣어도 앱이 `File.Exists() == false`
- 원인: 앱 삭제 후 재설치 시 디렉토리 소유권 변경
- 해결 시도: 앱 먼저 실행(폴더 생성) → 종료 → push → 여전히 실패
- 최종 우회: `/data/local/tmp/`에 push → 앱에서 직접 읽기

### 삽질 2: JNI 스레드 제한
- `ThreadPool`에서 `AndroidJavaObject` 호출 → NullReferenceException
- 원인: JNI는 메인 스레드에서만 호출 가능
- 해결: 초기화를 코루틴 메인 스레드에서 직접 호출

### 테스트 8: ekv4096 (.litertlm) + GPU 백엔드
```
Failed to create litert::ml_drift::DelegateKernelLiteRt: UNAVAILABLE
Requested allocation size - 18446744071872970752 bytes
Max allocation size for this GPU - 1073741824 bytes
Shape - {bhwdc, {1, 1, 1, 1, -1836580864}}
```
**GPU 메모리 1GB 제한 초과.** int32 오버플로 발생. Unity가 GPU를 점유 중이라 가용 VRAM 부족.

### 테스트 9: ekv1280 (.task) + GPU 백엔드
```
Initializing OpenCL-based API from serialized data → OK
Failed to save serialized data: Invalid file descriptor → Warning
Failed to create engine: INTERNAL ERROR at llm_litert_compiled_model_executor.cc:1955
```
GPU 할당 에러는 안 나왔지만 엔진 생성 단계에서 실패. `.task` 포맷과 AAR v0.10.0 호환 문제 또는 GPU delegate 내부 에러 추정.

### 미해결 문제

| 문제 | 원인 | 상태 |
|------|------|------|
| GPU 메모리 초과 (ekv4096) | KV 캐시 텐서가 GPU 최대 할당 1GB 초과 | 해결 불가 (모델 문제) |
| 엔진 생성 실패 (ekv1280) | .task 구버전 포맷 또는 GPU delegate 내부 에러 | 추가 조사 필요 |
| Unity + GPU 충돌 | Unity 렌더링이 GPU 점유 → LLM GPU 가속용 VRAM 부족 | 구조적 문제 |
| 화면 밝기 낮음 | Unity 기본 설정 | 수정 필요 |

### Day 2 교훈

1. **Unity + GPU LLM은 충돌한다** — Unity 렌더링과 LLM GPU 추론이 같은 GPU를 공유하면 VRAM 경쟁
2. **LiteRT-LM은 아직 Unity 통합이 어렵다** — Bazel 빌드 실패, JNI 스레드 제약, 모델 포맷 호환 문제
3. **Android Scoped Storage는 adb 테스트의 적** — 앱 데이터 폴더에 직접 push 불가, 우회 필요
4. **온디바이스 AI의 현실** — GPU 가속이 있어도 Unity 게임과 공존이 어려움

### 다음 시도 방향

| 경로 | 기대 | 난이도 |
|------|------|--------|
| **llama.cpp + Adreno OpenCL** | Qualcomm 공식 최적화, ~10 tok/s | 중간 (C++ 네이티브 플러그인) |
| **LiteRT-LM ekv1280 디버깅** | .task 포맷 호환 확인, GPU delegate 설정 조정 | 낮음 |
| **LiteRT-LM CPU 백엔드** | 7.3 tok/s (벤치마크), ONNX 대비 23배 | 낮음 (코드 한 줄) |
| **더 작은 모델** | Qwen3-0.6B (~600MB), GPU 메모리 충분 | 중간 |

---

## 9. llama.cpp + Adreno OpenCL 시도 (Day 2 후반)

### 왜 llama.cpp인가?
- Qualcomm이 공식으로 Adreno 최적화 OpenCL 커널을 기여
- S24 Ultra (Adreno 750)에서 Q4_0 기준 15-30 tok/s 기대
- CMake 빌드 — Bazel보다 훨씬 쉬움
- C API → Unity P/Invoke로 직접 연동 가능

### 빌드 과정

1. **OpenCL Headers** — KhronosGroup/OpenCL-Headers를 NDK sysroot에 복사 ✅
2. **OpenCL ICD Loader** — KhronosGroup/OpenCL-ICD-Loader를 NDK 크로스 컴파일 ✅
3. **llama.cpp 첫 빌드 시도** → 프로젝트 경로에 작은따옴표(`Demon Lord's Castle`)가 있어서 쉘 이스케이프 에러
   ```
   /bin/sh: -c: line 0: unexpected EOF while looking for matching `'
   ```
4. **심볼릭 링크로 우회** (`/tmp/llama-cpp`) → **빌드 성공!** 377/377 타겟 완료

### 모델
- `unsloth/Phi-4-mini-instruct-GGUF` Q8_0 (3.8GB)
- Q4_0은 레포에 없음, Q4_K_M은 OpenCL 미지원 → Q8_0 선택

### Unity 통합
- `libllama.so` (34MB), `libggml.so`, `libggml-base.so`, `libggml-cpu.so`, `libggml-opencl.so` → `Assets/Plugins/Android/libs/arm64-v8a/`
- C# P/Invoke 브릿지 (`LlamaCppBridge.cs`) — llama.h C API 직접 호출
- `LlamaCppTestUI.cs` — 테스트 UI

### 테스트 결과: 라이브러리 로드 실패

```
dlopen failed: library "libomp.so" not found:
needed by libggml-cpu.so in namespace clns-9
```

**원인:** `libggml-cpu.so`가 OpenMP(`libomp.so`)에 의존하는데 APK에 포함되지 않음.

**해결 방법 (다음 세션):**
1. `-DGGML_OPENMP=OFF`로 재빌드 (OpenMP 비활성화, 성능 약간 하락)
2. 또는 NDK의 `libomp.so`를 `Assets/Plugins/Android/libs/arm64-v8a/`에 복사

### Day 2 최종 요약

| 시도 | 결과 | 핵심 원인 |
|------|------|----------|
| QNN HTP + libcdsprpc manifest | HTP 동작 확인, 363노드 중 3개만 NPU | INT4 연산자 HTP 미지원 |
| LiteRT-LM ekv4096 GPU | GPU 메모리 초과 | 텐서 할당 1GB 제한 |
| LiteRT-LM ekv1280 GPU | 엔진 생성 실패 | .task 포맷 호환 또는 delegate 에러 |
| llama.cpp OpenCL | 라이브러리 로드 실패 | libomp.so 누락 |

**다음 세션 첫 번째 할 일:** llama.cpp `-DGGML_OPENMP=OFF`로 재빌드 → 실기기 테스트. 이게 성공하면 Adreno GPU 가속으로 15-30 tok/s 달성 가능.

---

## 10. llama.cpp OpenCL 성공! (Day 3: 2026-04-05)

### libomp.so 문제 해결
- `-DGGML_OPENMP=OFF` 추가하여 재빌드 → 의존성 제거 확인

### P/Invoke 구조체 크래시 해결
- `LlamaModelParams` 구조체 직접 마샬링 시도 → SIGSEGV (구조체 레이아웃 불일치)
- **해결:** C 래퍼 함수 (`unity_bridge.c`) 작성 — 복잡한 구조체를 내부에서 처리하고 단순한 인터페이스만 노출
- `unity_llama_model_load(path, n_gpu_layers)` 등 8개 함수로 단순화

### 버튼 이벤트 미작동 해결
- Inspector에서 연결했지만 빌드에 반영 안 됨 (persistent listener 직렬화 문제)
- **해결:** `Start()`에서 `generateButton.onClick.AddListener(OnGenerateClicked)` 런타임 등록

### 테스트 10: S24 Ultra + llama.cpp OpenCL (Adreno 750) 🎉

| 항목 | 결과 |
|------|------|
| 모델 | Phi-4-mini Q8_0 (3.8GB GGUF) |
| 모델 로딩 | ~23초 |
| 프롬프트 토큰 | 100 |
| **생성 시간** | **16.8초** |
| **생성 토큰** | **150** |
| **tok/s** | **8.9** |
| GPU | Adreno 750 (OpenCL) |

### 전체 벤치마크 비교

| 방식 | tok/s | 150토큰 생성 | ONNX 대비 |
|------|-------|-------------|----------|
| ONNX Runtime CPU (S24) | 0.21 | 523초 | 기준 |
| ONNX Runtime QNN (S24) | 0.31 | 490초 | 1.5x |
| ONNX Runtime CPU (Mac) | 0.45 | 246초 | 2.1x |
| **llama.cpp OpenCL (S24)** | **8.9** | **16.8초** | **42x** |

**온디바이스 AI로 게임에 사용 가능한 속도 달성!**
16.8초면 던전 입장 로딩 화면에서 충분히 사용 가능.

### 핵심 교훈
1. **ONNX Runtime + QNN은 INT4 모델에서 사실상 무용** — 363개 중 3개 노드만 NPU
2. **LiteRT-LM은 Unity와 GPU 충돌** — 같은 GPU를 렌더링과 LLM이 공유하면 메모리 부족
3. **llama.cpp OpenCL이 정답** — Qualcomm 공식 Adreno 최적화, 빌드도 쉬움 (CMake)
4. **C 래퍼가 P/Invoke의 핵심** — 복잡한 C 구조체를 직접 마샬링하지 말고 래퍼로 단순화
5. **Q8_0이 Adreno OpenCL 최적** — Q4_0, Q6_K 모두 더 느림 (아래 비교 참조)

### 양자화별 Adreno 750 OpenCL 벤치마크 (Day 4: 2026-04-06)

| 양자화 | 모델 크기 | 로딩 시간 | tok/s | 비고 |
|--------|----------|----------|-------|------|
| **Q8_0** | 3.8GB | 24.5s | **9.0** | GPU 연산 최적화됨 |
| Q6_K | 3.2GB | 24.5s | 4.2 | GPU에서 디퀀타이즈 오버헤드 |
| Q4_0 | 2.3GB | 17.6s | 5.1 | 로딩은 빠르나 연산 느림 |

**결론:** Adreno OpenCL 백엔드에서는 Q8_0이 가장 빠름. Q4_0/Q6_K는 GPU에서 직접 연산이 비효율적이라 오히려 느려짐. Q8_0 → Q4_0 재양자화는 llama.cpp에서 지원 안 함 (FP16/BF16 원본에서만 가능).

**참고:** Q8_0에서 재양자화 시도 → `requantizing from type q8_0 is disabled` 에러. BF16 원본(7.7GB)에서 Q4_0 변환 성공.

### 모델 크기 비교: Qwen3-1.7B vs Phi-4-mini (Day 4 후반)

Phi-4-mini (3.8B)가 Q8_0 최적이라는 걸 확인한 후, 더 작은 모델로 전환 시도.

**Qwen3-1.7B Q8_0 (1.8GB)** 테스트 결과:

| | Phi-4-mini (3.8B) | Qwen3-1.7B (1.7B) |
|---|---|---|
| 모델 크기 | 3.8GB | **1.8GB** |
| 로딩 시간 | 24.5초 | **14.4초** |
| 생성 속도 | 9.0 tok/s | **16.6 tok/s** |
| 150토큰 생성 | 16.8초 | **9.1초** |
| 몹 이름 품질 | "몬스터이름" (프롬프트 그대로) | **"토끼"** (실제 이름 생성) ✅ |
| JSON 구조 | 유효 | 유효 ✅ |

**Qwen3-1.7B를 기본 모델로 채택.** 모든 면에서 우세:
- 속도 **1.8배** (9.0 → 16.6 tok/s)
- 모델 크기 **53% 절감** (3.8GB → 1.8GB)
- 출력 품질 **더 우수** (실제 몹 이름 생성)
- 로딩 시간 **41% 단축** (24.5s → 14.4s)

### Day 4 전체 성과 요약

1. UI 레이아웃 수정 — ScrollView 제거, OutputText 직접 배치
2. 양자화 비교 완료 — Q8_0 > Q4_0 > Q6_K (Adreno OpenCL 기준)
3. **Qwen3-1.7B 채택** — 속도 2배, 크기 절반, 품질 향상
4. 한글 폰트 깨짐 일부 잔존 (NotoSansKR SDF 적용했으나 일부 유니코드 누락)

### Unity 온디바이스 LLM 파이프라인 구조

```
[Unity C# (게임 로직)]
    │
    │ P/Invoke
    ▼
[unity_bridge.c (C 래퍼)]
    │
    │ llama.h C API 호출
    ▼
[libllama.so + libggml-opencl.so]
    │
    │ OpenCL 커널 실행
    ▼
[Adreno GPU (Snapdragon)]
    │
    ▼
[GGUF 모델 파일 (/data/local/tmp/)]
```

**빌드 파이프라인:**
1. **llama.cpp** — NDK + CMake로 Android ARM64 크로스 컴파일 (`-DGGML_OPENCL=ON -DGGML_OPENMP=OFF`)
2. **unity_bridge.c** — llama.h의 복잡한 구조체를 숨기고 단순한 C 함수로 래핑
3. **NDK clang** — unity_bridge.c를 `libunity_llama.so`로 컴파일
4. **.so 배치** — `libllama.so`, `libggml*.so`, `libunity_llama.so` → `Assets/Plugins/Android/libs/arm64-v8a/`
5. **C# P/Invoke** — `[DllImport("unity_llama")]`로 Unity에서 직접 호출

**핵심 래퍼 함수 (unity_bridge.c):**
```c
unity_llama_backend_init()          // 백엔드 초기화
unity_llama_model_load(path, gpu)   // 모델 로드 (GPU 레이어 오프로드)
unity_llama_context_create(model)   // 컨텍스트 생성
unity_llama_tokenize(vocab, text)   // 텍스트 → 토큰
unity_llama_decode_batch(ctx, tok)  // 토큰 디코딩
unity_llama_sample_greedy(ctx)      // 다음 토큰 샘플링
unity_llama_token_to_text(vocab)    // 토큰 → 텍스트
```

**왜 이 구조인가:**
- Unity C#에서 llama.h 구조체를 직접 마샬링하면 메모리 레이아웃 불일치로 크래시
- C 래퍼로 구조체를 내부 처리하고 opaque 포인터만 노출하면 안정적
- OpenCL은 디바이스 시스템 라이브러리(`libOpenCL.so`)를 런타임에 로드 → 별도 번들 불필요
- 모델은 APK 외부에 배치 (4GB+ APK 불가) → `adb push` 또는 앱 내 다운로드

### 현재 최종 스택

| 항목 | 선택 |
|------|------|
| 추론 엔진 | llama.cpp (OpenCL, Adreno 최적화) |
| 모델 | Qwen3-1.7B Q8_0 (1.8GB) |
| GPU 백엔드 | OpenCL (Adreno 750) |
| 성능 | 16.6 tok/s, 9.1초/150토큰 |
| Unity 연동 | C 래퍼 (unity_bridge.c) + P/Invoke |

### 결론: QNN EP는 필수, 그러나 사전 변환 모델 필요
S24 Ultra (Snapdragon 8 Gen 3) CPU only로 8분 43초, QNN 런타임 컴파일로도 8분 10초. 둘 다 게임에 사용 불가.

### 다음 단계: Qualcomm AI Engine Direct SDK
QNN EP를 가장 현실적으로 적용하는 경로는 **Qualcomm AI Engine Direct SDK**:
1. AI Engine Direct SDK 설치 (Qualcomm 개발자 포털)
2. QNN 백엔드 포함 ONNX Runtime 빌드 또는 QNN context binary 생성
3. 모델을 QNN 호환 포맷으로 변환 (Qualcomm AI Hub 활용 가능)
4. 커스텀 AAR을 Unity에 통합
5. `SessionOptions.AppendExecutionProvider_QNN()` 적용

NPU 가속 시 토큰당 수십~수백ms까지 개선 기대. S24 Ultra급이면 10~30초 내 생성 가능성 있음.

---

## 오늘의 성과 요약

### 성공한 것
- ONNX Runtime + Unity Android 연동
- C# tiktoken BPE 토크나이저 구현
- KV 캐시 기반 LLM 추론 엔진 구현
- Phi-4-mini로 유효한 JSON 생성 확인
- Android 실기기 배포 파이프라인 구축

### 배운 것
- 3.8B 모델 CPU 추론은 게임에 쓰기엔 너무 느림 (토큰당 ~1-2초)
- StreamingAssets에 대용량 파일 넣으면 Gradle 빌드 터짐
- Unity TMP 기본 폰트에 한글 없음
- New Input System / Legacy Input 공존 이슈
- Editor 씬 빌더에서 `AddListener` vs `AddPersistentListener` 차이
- NNAPI ≠ NPU 풀 가속 → QNN EP 필요

### 다음 목표
- **QNN EP 적용 (최우선)** — Qualcomm AI Engine Direct SDK로 NPU 가속
- NPU 가속 전후 성능 비교 (현재 기준점: CPU 523초)
- UI 출력 디버깅 (폰트/결과 문제)
- 더 작은 모델 (SmolLM2-1.7B 등) 비교 테스트
- 프롬프트 엔지니어링 (몹 이름 오류 수정)
