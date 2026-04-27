# Changelog

---

## 2026-04-27 — Harden segmentation thread against crash on empty frame or unexpected Vision result

### Fixed
- **Empty frame guard** — if `cvtColor` produces an empty or zero-size `bgraMat` (e.g., during startup before the camera has delivered the first frame), `CVPixelBufferCreate` would be called with width=0/height=0. Guard added: skip the frame if `bgraMat.empty()` or either dimension is zero.
- **Vision result type validation** — `request.results[0]` was cast directly to `VNPixelBufferObservation *` with no type check. If Vision ever returns a result of a different class, dereferencing `obs.pixelBuffer` would be an unrecognised-selector crash. Added `isKindOfClass:[VNPixelBufferObservation class]` check before the cast, plus a nil guard on `obs` and `obs.pixelBuffer`.

---

## 2026-04-27 — Fix Flow Direction Color (H) grayscale and Flow Ripple (J) no-color

### Fixed
- **H shows only grayscale / J shows no ripple color** — optical flow is computed at `FLOW_SCALE = 0.25×` resolution but after `cv::resize` back to full size the vector values remained in small-scale pixel units. `flowSensitivity = 10` was effectively "10px at 0.25× = 2.5px full-scale", making saturation near-zero (→ grayscale in H). The J mode injection threshold `mag < 0.5` was almost never met, so no color was ever injected. Fix: multiply `flowMap` by `1/FLOW_SCALE` after resize to convert vectors to full-resolution pixel units.

---

## 2026-04-27 — Fix Datamosh (Y) black screen on entry

### Fixed
- **Datamosh shows black screen or occasional flashes on switch** — diff was computed between adjacent buffer slots (`bufIdx-1` vs `bufIdx-2`). At 60fps objects barely move between consecutive frames, so diffs are near-zero and the IIR accumulator decays to black faster than it builds. With `updateSpeed > 1`, adjacent slots are duplicate camera frames (diff = 0 identically). Fix: compare `MOTION_LOOKBACK` (10) frames apart, matching the other motion modes.
- **`DATAMOSH_BOOST` reduced 5.0 → 1.5** — with the larger 10-frame diff, the old boost caused immediate saturation to white; 1.5 gives trails that build naturally and remain tuneable with Up/Down.

---

## 2026-04-27 — Fix segmentation thread freeze after ~8 minutes

### Fixed
- **G/K image freezes after ~8 minutes regardless of spacing** — `VNSequenceRequestHandler` accumulates internal temporal state every frame; after ~28 k frames (~8 min at 60 fps) that state exhausts GPU memory and `performRequests` hangs indefinitely. Fix: reset the handler every 1800 frames (~30 s) to flush accumulated state.
- **ObjC memory leak on handler reset** — without ARC, reassigning `seqHandler` leaked the old object. Added `-fobjc-arc` to `CXXFLAGS` so the old handler is released automatically on reassignment.

---

## 2026-04-25 — Fix segmentation thread stall at high echo spacing

### Fixed
- **G/K image freezes after a few minutes at high `tghostSpacing`** — two root causes:
  1. `CVPixelBufferCreateWithBytes` gave Vision a pointer into `bgraMat.data` with no release callback. Vision can hold GPU references to the buffer past `performRequests` return; the next frame then overwrites that memory, corrupting Vision's state over time. Fix: `CVPixelBufferCreate` (Vision-owned allocation) + `memcpy`.
  2. Creating a new `VNImageRequestHandler` every frame accumulates GPU/Metal resources and eventually stalls. Fix: `VNSequenceRequestHandler` reused across all frames — the correct API for video, avoids per-frame resource churn.

---

## 2026-04-25 — Replace Python/MediaPipe segmentation with Apple Vision framework

### Changed
- **G and K modes no longer require Python or MediaPipe** — person segmentation now uses `VNGeneratePersonSegmentationRequest` (built into macOS 12+). No installation needed on any target machine.
- **Removed Python subprocess** — `launchSegServer()`, `seg_server.py`, and `.tflite` model file eliminated entirely. `segmentLoop()` now wraps each frame as a `CVPixelBuffer`, runs the Vision request synchronously on the background thread, and reads back the `OneComponent8` mask.
- **Source file renamed** `time_mirror.cpp` → `time_mirror.mm` (Objective-C++ required for Vision API)
- **Makefile** adds `-framework Vision -framework Foundation -framework CoreVideo`; removes `seg_server.py` and `.tflite` copy steps from `app` target
- **Ambiguous `Size`/`Point` names** qualified as `cv::Size`/`cv::Point` to resolve conflict with MacTypes.h definitions pulled in by the ObjC headers

---

## 2026-04-25 — Rainbow Ghost (K) tuning

### Changed
- **K: Rainbow Ghost — no temporal fade** — removed the 30%→100% brightness ramp across echoes; all echoes now render at full brightness scaled only by mask alpha
- **K: Rainbow Ghost — hue direction reversed** — hue now advances from newest echo to oldest (was oldest to newest), so color leads the moving subject rather than trailing it
- **K: Rainbow Ghost — Up/Down controls spacing** — arrow keys now adjust `tghostSpacing` (frames between echoes) instead of `rainbowSpeed`; spacing shared with G mode
- **Echo count raised to 7** — `TGHOST_ECHOES` increased from 6 to 7 (affects both G and K)
- **Default `tghostSpacing` doubled to 20** — was 10; R resets to 20 in both G and K

---

## 2026-04-25 — Temporal Ghost fix, distribution pipeline

### Fixed
- **Temporal Ghost (G) stale mask bug** — `segmentLoop` always jumped to the latest frame, leaving skipped buffer slots with masks from the previous wrap (~3.3s ago). Echo indices landing on those slots showed the person in an old position. Fix: after writing `maskBuffer[latest]`, propagate the same mask to all slots skipped since `lastSeg`.
- **App closes immediately on first launch** — seg server pipe failure (`seg_server.py` not found when running as bundle) set `running = false`, killing the whole app. Fix: pipe errors in `segmentLoop` now silently exit the thread without touching `running`.
- **`seg_server.py` not found in app bundle** — binary used a relative path (`"seg_server.py"`) which resolved from `/` when launched as a bundle. Fix: use `_NSGetExecutablePath` to find the executable's directory, then look in `../Resources/` (with fallback to `./` for dev builds). Script is now copied to `Contents/Resources/` in the Makefile.
- **Camera access denied on first launch** — `VideoCapture::open(0)` triggered the permission dialog but returned immediately before the user could approve. Fix: retry loop (20 × 500ms = 10s window).
- **Duplicate LC_RPATH crash on other machines** — `dylibbundler` replaced multiple existing rpaths each with `@executable_path/libs/`, leaving duplicate LC_RPATH entries that macOS 15 rejects. Fix: Makefile now loops `install_name_tool -delete_rpath` until all copies are removed, then adds exactly one back.
- **App bundle signing order** — `Info.plist` was written after `codesign`, producing `bundle format unrecognised`. Fix: `Info.plist` is now written before any signing step.

### Added
- **`make notarize` target** — full pipeline: build → sign → zip → submit to Apple → wait → staple → produce `TimeMirror_notarized.zip`
- **`make notarize-setup` target** — one-time credential storage in Keychain via `xcrun notarytool store-credentials`
- **`make app` target** — builds signed `TimeMirror.app` using Developer ID Application certificate with hardened runtime + camera entitlement
- **`entitlements.plist`** — `com.apple.security.device.camera` required for hardened runtime camera access
- **DISTRIBUTION.md, MODES.md, CHANGELOG.md** — project documentation

---

## 2026-04-23 — Mode expansion, performance tuning

### Added (committed)
- **N: Turbulence** — motion history accumulation drives displacement, chroma shift, and saturation (`turbShift`, adjustable)
- **Y: Datamosh** — IIR diff accumulation leaves decaying colour trails (`datamoshDecay = 0.92`)
- **E: Ghost Echo** — 7 motion-masked temporal echoes on black (`ghostSpacing = 8`)
- **B: Background Removal** — motion mask isolates foreground on black
- **G: Temporal Ghost** — 6 neural-segmented person echoes through time (`tghostSpacing = 10`, requires `seg_server.py`)
- **J: Flow Color Ripple** — directional hue advected with optical flow, IIR decay
- macOS app bundle support (`TimeMirror.app`)

### Changed
- Mode defaults and parameter ranges tuned (committed as "Tune mode defaults and ranges")
- Removed **Scan Glitch** mode (previously N); replaced by Turbulence

---

## Earlier commits

| Hash | Summary |
|------|---------|
| `5570a33` | Optimise rendering: dedicated capture thread, OpenMP, pre-alloc output Mat, `CAP_PROP_BUFFERSIZE=1` |
| `2fe6299` | Add `dist` build target, dylibbundler packaging, `.gitignore` |
| `2c45a02` | Initial C++ port: WASD direction modes, combo detection, fullscreen, speed control |
