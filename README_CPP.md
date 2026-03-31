# Time Mirror Effect - C++ Version

High-performance C++ implementation achieving 50–60 FPS at 1920×1080. Each row (or column) of the output is pulled from a different moment in the past, creating mesmerizing temporal distortions.

## Installation on Mac

1. Install Homebrew (if not already installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install OpenCV and libomp:
```bash
brew install opencv libomp
```

3. Compile with OpenMP (recommended — ~2–4x faster displacement):
```bash
g++ -std=c++17 -O3 -march=native -Xpreprocessor -fopenmp \
  -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp \
  time_mirror.cpp -o time_mirror `pkg-config --cflags --libs opencv4`
```

Or without OpenMP (still fast thanks to capture thread and -O3):
```bash
g++ -std=c++17 -O3 -march=native time_mirror.cpp -o time_mirror `pkg-config --cflags --libs opencv4`
```

If you get an error about `opencv4.pc not found`, replace `opencv4` with `opencv`.

4. Run:
```bash
./time_mirror
```

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
| **C** | Chromatic Aberration | R/G/B channels pulled from slightly different frame offsets |
| **X** | Motion Chromatic | Still areas = no color shift; moving areas = vivid RGB temporal split |
| **P** | Prismatic Echo | 6 spectral echoes (red→magenta) through the buffer; motion creates rainbow trails |
| **H** | Flow Direction Color | Optical flow direction → hue, magnitude → saturation; directional color from motion |
| **N** | Scan Glitch | Bands jump to random past frames with CRT shift; occasional full-frame flicker |

### Other Controls
| Key | Action |
|-----|--------|
| **Up / Down Arrow** | Increase / decrease speed (lines per frame) |
| **R** | Reset speed to 1 |
| **F** | Toggle fullscreen |
| **Q / ESC** | Quit |

Speed changes display a temporary on-screen indicator for 3 seconds.

## Performance

| Mode | FPS (1920×1080) |
|------|-----------------|
| Up/Down (W, S, WS) | 50–60 FPS |
| Left/Right (A, D, AD) | 40–50 FPS |

Performance comes from three optimisations working together:
- **Dedicated capture thread** — camera grab runs concurrently with rendering (lock-free via `std::atomic`)
- **OpenMP parallel loops** — displacement step distributed across all CPU cores
- **Compiler flags** `-O3 -march=native` — auto-vectorisation and SIMD

## Troubleshooting

**"opencv4.pc not found"**
- Replace `opencv4` with `opencv` in the compile command

**Camera permission denied**
- System Preferences → Security & Privacy → Camera → enable for Terminal

**Segmentation fault**
- Usually an OpenCV version mismatch: `brew upgrade opencv`

**Xcode / compiler errors**
- Install command line tools: `xcode-select --install`

## Configuration

Edit these constants at the top of `time_mirror.cpp`:

```cpp
const int FRAME_WIDTH  = 1920;  // Camera resolution width
const int FRAME_HEIGHT = 1080;  // Camera resolution height
const int BUFFER_SIZE  = 200;   // Frames stored (~3–4 seconds at 60 fps)
```
