# Mode Reference

Technical details for each effect mode in `time_mirror.cpp`. For the user-facing description see README_CPP.md.

---

## Direction Modes

These modes select which frame to pull for each row or column based on its distance from an origin line.

| Key | Mode string | Origin | Direction |
|-----|-------------|--------|-----------|
| S | `"s"` | Top | Newest at top, oldest at bottom |
| W | `"w"` | Bottom | Newest at bottom, oldest at top |
| D | `"d"` | Left | Newest at left, oldest at right |
| A | `"a"` | Right | Newest at right, oldest at left |
| W+S | `"ws"` | Centre (horizontal) | Newest at centre, oldest at top/bottom |
| A+D | `"ad"` | Centre (vertical) | Newest at centre, oldest at left/right |

**Frame index formula (S mode example):**
```
frameOffset = bufIdx - (y * BUFFER_SIZE / height)
```
Row 0 (top) = current frame; row `height-1` = `BUFFER_SIZE` frames ago.

**Combo detection:** pressing the second key within 0.5s of the first triggers the centre-out variant (`checkForCombo()`).

---

## Special Effect Modes

### Z — Motion Adaptive
```
motionMap = GaussianBlur(absdiff(frame[now], frame[now - MOTION_LOOKBACK]))
offset = motionMap[pixel] * updateSpeed * BUFFER_SIZE / height
```
- `MOTION_LOOKBACK = 10` frames
- Still pixels → same displacement as base direction mode
- Moving pixels → larger temporal offset (more past)
- Up/Down adjusts `updateSpeed`

### X — Chromatic Time Shift
```
B channel ← frame[bufIdx - 1]
G channel ← frame[bufIdx - 1 - chromaOffset]
R channel ← frame[bufIdx - 1 - chromaOffset * 2]
```
- Fixed per-channel temporal split; `chromaOffset` = 3 frames (not runtime-adjustable)
- Effect is uniform across the frame (no motion dependency)

### J — Motion Chromatic
```
spread = motionMap[pixel] * chromaSpread
B ← frame[bufIdx - 1]
G ← frame[bufIdx - 1 - spread]
R ← frame[bufIdx - 1 - spread * 2]
```
- Default `chromaSpread = 40` frames at full motion
- Still pixels: spread = 0, all channels from same frame → no colour shift
- Moving pixels: full spread → vivid RGB temporal split
- Up/Down adjusts `chromaSpread`

### T — Prismatic Echo
```
6 echoes spaced echoSpacing frames apart
Tint: Red → Yellow → Green → Cyan → Blue → Magenta
Output: average of 3 echoes per channel → no colour cast on still images
```
- Default `echoSpacing = 23` frames
- Up/Down adjusts `echoSpacing` (range: 1 – `BUFFER_SIZE/3`)

### K — Flow Direction Color
```
Farneback optical flow at FLOW_SCALE (0.25×) resolution
hue    = atan2(vy, vx)          (flow direction)
sat    = min(mag / flowSensitivity, 1.0)
value  = pixel brightness
Output = HSV → BGR inline (no intermediate Mat)
```
- Default `flowSensitivity = 10.0` px/frame
- Still pixels → desaturated (sat ≈ 0)
- Up/Down adjusts `flowSensitivity`

### Y — Flow Color Ripple
```
Per-pixel hue assigned by flow direction; colour advects with flow vectors
IIR decay: rippleBuffer = rippleBuffer * rippleDecay + newColour
```
- Default `rippleDecay = 0.93` (~1 second fade at 60 fps)
- Colours persist and drift with motion; still areas fade to grey

### I — Turbulence
```
turbulenceMap = IIR accumulation of absdiff(frame[now], frame[now - MOTION_LOOKBACK])
displacement  = turbulenceMap[pixel] * turbShift  (pixel offset into past frames)
chroma shift  = displacement * chromaScale
saturation    = turbulenceMap[pixel] (motion = vivid colour)
```
- Default `turbShift = 20.0` px; Up/Down adjusts it
- Still areas: no displacement, desaturated
- Moving areas: displaced + chromatic + saturated

### U — Datamosh
```
diff = absdiff(frame[now], frame[now - MOTION_LOOKBACK])
boost = DATAMOSH_BOOST_K × (1 − datamoshDecay)   // linked so steady-state stays constant
datamoshAccum = datamoshAccum × datamoshDecay + diff × boost
Output = datamoshAccum clamped to [0, 255]
```
- Default `datamoshDecay = 0.92` (~14 frame half-life at 60 fps); ceiling 0.992 (~1.5 s)
- `DATAMOSH_BOOST_K = 6.5625` — boost scales down as decay increases to prevent white saturation
- Diff taken `MOTION_LOOKBACK` (10) frames apart — adjacent frames are near-zero at 60 fps
- Computed in one fused OpenMP pass (no intermediate buffers)
- Motion leaves bright colour trails that decay over time; still areas fade to black

### C — Ghost Echo
```
7 echoes spaced ghostSpacing frames apart, masked by current motion mask
Each echo composited with opacity 1/e (newest = full, oldest = faint)
Background = black
```
- Default `ghostSpacing = 8` frames
- Uses the same `motionMap` as M/X modes
- Up/Down adjusts `ghostSpacing`

### H — Temporal Ghost
```
7 echoes spaced tghostSpacing frames apart
Each echo uses maskBuffer[fi[e]] (Vision person mask) to isolate the subject
Fade: oldest = 30% brightness, newest = 100%
Background = black
```
- Default `tghostSpacing = 20` frames
- `segReady` used as base (never `bufIdx-1`) to avoid data race with segment thread
- `maskBuffer` gaps propagated to avoid stale masks from previous buffer wrap
- Up/Down adjusts `tghostSpacing` (range: 1 – `BUFFER_SIZE / TGHOST_ECHOES`)

### G — Rainbow Ghost
```
7 echoes spaced tghostSpacing frames apart
Each echo tinted a single hue spaced RAINBOW_HUE_STEP (45°) apart
Hue assigned newest→oldest: echo 0 = rainbowHue, echo 1 = rainbowHue - 45°, …
rainbowHue advances rainbowSpeed (30 °/s) each frame so colors animate
No temporal fade — all echoes rendered at full brightness (maskAlpha only)
Background = black
```
- Default `tghostSpacing = 20` frames (shared with H mode)
- Up/Down adjusts `tghostSpacing` (range: 1 – `BUFFER_SIZE / TGHOST_ECHOES`)

### B — Tunnel Ghost
```
s[e]      = 1.0 + (tunnelScale − 1.0) × e / (TGHOST_ECHOES − 1)
inv_s[e]  = 1 / s[e]
src_x = cx + (x − cx) × inv_s[e]      (per-pixel, per-echo)
src_y = cy + (y − cy) × inv_s[e]      (per-row, per-echo — hoisted)
```
- Newest echo (e=0): s=1.0, full-frame (inv_s=1.0, no zoom)
- Oldest echo (e=6): s=tunnelScale, zoomed in (inv_s<1.0, samples a centre crop — person appears larger than frame)
- Rendered back-to-front so nearest (fullest) echo paints last
- Same rainbow hue palette as G mode; `rainbowHue` advances at `rainbowSpeed`
- Default `tunnelScale = 3.0` (oldest echo 3× zoomed)
- Up/Down adjusts `tunnelScale` (range: 1.2 – 8.0, step 0.5)
- `tghostSpacing` shared with G/H modes for echo frame spacing

### Expanding ring backdrop (H, G, B modes)
```
dist  = sqrt((x − cx)² + (y − cy)²)
phase = fmod(dist − ringOffset, RING_SPACING)
pixel = 35 if phase < RING_SPACING/2 else 0   (dark gray / black alternation)
ringOffset advances RING_SPEED/60 px per frame
```
- `RING_SPEED = 75` px/s; `RING_SPACING = 200` px between ring centres
- Background drawn first; person echoes composited on top
- `dy²` hoisted per row; only `sqrtf` computed per pixel

---

### V — Flow Warp
```
(vx, vy) = Farneback optical flow at FLOW_SCALE (0.25×), scaled to full-res units
sx = x + vx × flowWarpScale
sy = y + vy × flowWarpScale
output[y][x] = frameBuffer[recent][sy][sx]
```
- Default `flowWarpScale = 10.0`
- Flow computed by preprocessLoop and double-buffered into `flowMapBuf`
- Fast motion → large displacement; still areas → identity (no warp)
- Up/Down adjusts `flowWarpScale` (range: 1 – 50)

### N — Wave Warp
```
Wave propagation (explicit Euler, per frame):
  new[y][x] = (N+S+E+W) × 0.5 − prev[y][x]   (wave equation)
  new[y][x] × WAVE_DAMP + motionMap[y][x] × WAVE_SEED
Displacement:
  dx = waveA[y][x+1] − waveA[y][x−1]
  dy = waveA[y+1][x] − waveA[y−1][x]
  output[y][x] = frameBuffer[recent][y + dy×refract][x + dx×refract]
```
- `WAVE_DAMP = 0.97`; `WAVE_SEED = 3.0`
- Default `waveRefract = 15.0`; Up/Down adjusts (range: 1 – 50)
- Motion map seeded from preprocessLoop; full frame (including person) is distorted
- Double-buffered `waveA`/`waveB` swapped each frame (`std::swap`, O(1))

### M — Chroma Wave
```
Three independent wave simulations — one per RGB channel.
Each seeded from its own channel's absdiff (not grayscale motion map):
  drR = |curr[y][x].R − prev[y][x].R| / 255
  drG = |curr[y][x].G − prev[y][x].G| / 255
  drB = |curr[y][x].B − prev[y][x].B| / 255
Per-channel displacement:
  output[y][x].R = frameBuffer[recent][syR][sxR].R
  output[y][x].G = frameBuffer[recent][syG][sxG].G
  output[y][x].B = frameBuffer[recent][syB][sxB].B
```
- Default `chromaWaveRefract = 15.0`; Up/Down adjusts (range: 1 – 50)
- R/G/B motion drives independent wave patterns; different-coloured motion creates distinct per-channel refraction
- Full frame (including person) distorted — no mask compositing
- Three wave propagations fused into one OMP pass

---

## Segmentation subsystem (H, G, B modes)

Person segmentation uses `VNGeneratePersonSegmentationRequest` from Apple's Vision framework (macOS 12+). No Python, MediaPipe, or external model files required.

**Pipeline (runs on dedicated background thread):**
1. Convert `frameBuffer[latest]` BGR→BGRA; copy into a Vision-owned `CVPixelBuffer` (`CVPixelBufferCreate` + `memcpy` — Vision may hold GPU refs past `performRequests` return, so zero-copy is unsafe)
2. Feed to `VNSequenceRequestHandler` → `performRequests:onCVPixelBuffer:` (reused across frames; correct API for video, avoids per-frame GPU resource churn)
3. Read back `VNPixelBufferObservation` mask (`OneComponent8`: 255 = person, 0 = background)
4. Resize to full resolution (bilinear) and blur (σ=12) for soft edges
5. Propagate mask to any buffer slots skipped since last segmentation
6. Advance `segReady` (release) — render thread reads with `acquire`

`segReady` is only advanced after the full mask write completes, so the render thread can safely read `maskBuffer[segReady]` without a lock.

---

## Adding a new mode — checklist

1. Add mode string constant / `currentMode = "foo"` branch in key handler
2. Add rendering block in `applyTimeDisplacement()` (follow existing pattern)
3. Add `getModeName()` return string
4. Add Up/Down parameter adjustment in the arrow key handlers (×2)
5. Add R reset case in the reset handler
6. Update README_CPP.md controls table
7. Update CLAUDE.md Features and Key Algorithms sections
