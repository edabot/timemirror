# Changelog

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
