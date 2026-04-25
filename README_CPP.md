# Time Mirror Effect - C++ Version

High-performance C++ implementation achieving 50–60 FPS at 1920×1080. Each row (or column) of the output is pulled from a different moment in the past, creating mesmerizing temporal distortions.

## Quick Start (macOS App Bundle)

Download `TimeMirror_notarized.zip`, unzip, and drag `TimeMirror.app` to your Applications folder. Double-click to open — grant camera access when prompted.

> **Important:** Always move the app to `/Applications` before opening. Opening it directly from a zip or Downloads folder will trigger macOS App Translocation and the app will fail to launch.

## Build from Source

### Prerequisites

```bash
brew install opencv libomp
```

### Compile

```bash
make
```

Or directly:
```bash
g++ -std=c++17 -O3 -march=native -Xpreprocessor -fopenmp \
  -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp \
  time_mirror.cpp -o time_mirror `pkg-config --cflags --libs opencv4`
```

If you get `opencv4.pc not found`, replace `opencv4` with `opencv`.

### Run

```bash
make run
# or
./time_mirror
```

### Build a signed + notarized app bundle

```bash
make notarize
```

Produces `TimeMirror_notarized.zip` — trusted by Gatekeeper on any Mac.

---

## Controls

### Direction Modes

| Key | Effect |
|-----|--------|
| **S** | Top to bottom (newest at top) |
| **W** | Bottom to top (newest at bottom) |
| **D** | Left to right (newest at left) |
| **A** | Right to left (newest at right) |
| **W then S** (within 0.5s) | Center-out vertical |
| **A then D** (within 0.5s) | Center-out horizontal |

### Special Effect Modes

| Key | Mode | Description |
|-----|------|-------------|
| **M** | Motion Adaptive | Displacement intensity scales with per-pixel motion |
| **C** | Chromatic Time Shift | R/G/B channels pulled from slightly different frame offsets |
| **X** | Motion Chromatic | Still areas = no color shift; moving areas = vivid RGB temporal split |
| **P** | Prismatic Echo | 6 spectral echoes (red→magenta) spaced through the buffer; motion creates rainbow trails |
| **H** | Flow Direction Color | Optical flow direction → hue, magnitude → saturation |
| **J** | Flow Color Ripple | Directional color that advects with optical flow and decays over time |
| **N** | Turbulence | Motion history drives displacement + chroma + saturation |
| **Y** | Datamosh | IIR motion diff accumulation — moving subjects leave bright color trails |
| **E** | Ghost Echo | 7 motion-masked temporal echoes composited on black |
| **G** | Temporal Ghost | 7 person silhouettes at different moments in time, on black |
| **K** | Rainbow Ghost | Like G but each echo tinted a cycling hue; color flows newest→oldest |

> **G and K modes** use Apple's Vision framework (built into macOS 12+) — no Python or MediaPipe required.

### Other Controls

| Key | Action |
|-----|--------|
| **Up / Down Arrow** | Increase / decrease speed (or adjust mode-specific parameter) |
| **R** | Reset speed / parameter to default |
| **F** | Toggle fullscreen |
| **Q / ESC** | Quit |

Speed and parameter changes show a temporary on-screen indicator for 3 seconds.

---

## Performance

| Mode | FPS (1920×1080) |
|------|-----------------|
| Up/Down (W, S, WS) | 50–60 FPS |
| Left/Right (A, D, AD) | 40–50 FPS |

Three optimisations working together:
- **Dedicated capture thread** — camera grab runs concurrently with rendering (lock-free via `std::atomic`)
- **OpenMP parallel loops** — displacement step distributed across all CPU cores
- **Compiler flags** `-O3 -march=native` — auto-vectorisation and SIMD

---

## Configuration

Edit these constants in `time_mirror.cpp`:

```cpp
const int FRAME_WIDTH  = 1920;  // Camera resolution width
const int FRAME_HEIGHT = 1080;  // Camera resolution height
const int BUFFER_SIZE  = 200;   // Frames stored (~3–4 seconds at 60 fps)
```

---

## Troubleshooting

**App closes immediately after launch**
- Make sure the app is in `/Applications`, not run directly from a zip
- Grant camera access when prompted; the app will retry for ~10 seconds

**"opencv4.pc not found"**
- Replace `opencv4` with `opencv` in the compile command

**Camera permission denied (Terminal)**
- System Settings → Privacy & Security → Camera → enable for Terminal

**Segmentation fault**
- Usually an OpenCV version mismatch: `brew upgrade opencv`

**Xcode / compiler errors**
- Install command line tools: `xcode-select --install`
