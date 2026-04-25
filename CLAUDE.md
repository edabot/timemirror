# Time Mirror Effect - Project Documentation

A real-time video effect that creates a "funhouse mirror" by displaying different horizontal or vertical strips from different moments in time, creating mesmerizing temporal distortions.

## Overview

This project captures live webcam footage and applies a time-displacement effect where each line (row or column) of the output comes from a different frame in the past. The effect creates beautiful visual distortions as objects move through the frame.

## Project Evolution

### Development Journey
- **Started with Processing (Java)** - Initial prototype for quick visualization
- **Moved to Python/OpenCV** - Better performance and control
- **C++ Implementation** - 2-3x performance improvement, hitting target framerates
- **C++ Optimized** - Threaded capture + OpenMP + compiler flags targeting 60 FPS

### Performance Progression
| Version | Language | Resolution | FPS (Up/Down) | FPS (Left/Right) |
|---------|----------|------------|---------------|------------------|
| v1-v6   | Processing | 640×480 | 15-20 | 10-15 |
| v7-v23  | Python | 1280×720 | 25 | 15-20 |
| C++ v1  | C++ | 1920×1080 | 25+ | 20-25 |
| C++ v2  | C++ + OpenMP | 1920×1080 | 50-60 | 40-50 |

## Technical Details

### How It Works

1. **Circular Buffer**: Stores the last 200 frames in memory
2. **Line Selection**: Each output line pulls from a different frame based on:
   - Current mode (W/S/A/D)
   - Distance from origin point
   - Buffer position
3. **Real-time Processing**: Applies effect at 50-60 FPS at 1080p

### Architecture

```
┌─────────────────────────────────┐   ┌──────────────────────────────────────┐
│        Capture Thread           │   │             Main Thread              │
│                                 │   │                                      │
│  Camera Input (1920×1080)       │   │  Read writeIndex (atomic acquire)    │
│      ↓                          │   │      ↓                               │
│  Mirror Horizontally (flip)     │   │  Apply Time Displacement             │
│      ↓                          │   │    (OpenMP parallel row/col copy)    │
│  Write to frameBuffer[idx]      │   │      ↓                               │
│      ↓                          │   │  Draw UI Overlays                    │
│  Advance writeIndex (release)◄──┼───┼──────↓                               │
│      ↓                          │   │  Display Output                      │
│  (loop immediately)             │   │      ↓                               │
│                                 │   │  Handle Keyboard Input               │
└─────────────────────────────────┘   └──────────────────────────────────────┘
```

### Key Algorithms

**Top-Down Mode (S):**
- Top line = current frame
- Bottom line = 200 frames ago
- Each line interpolates between

**Center-Out Mode (W+S):**
- Center line = current frame
- Top/Bottom edges = oldest frames
- Radial time gradient from center

**Motion Chromatic Mode (X):**
- Per-frame: `absdiff` between current and `MOTION_LOOKBACK` frames ago → Gaussian blur → CV_32F motionMap (0–1)
- Per-pixel: motion value scales a `chromaSpread` offset (default 20 frames)
- Blue channel pulled from `idx - spread`, Green from `idx`, Red from `idx + spread`
- Still pixels: all three channels from same frame → no color shift
- Moving pixels: full `chromaSpread` offset → vivid RGB split

**Prismatic Echo Mode (P):**
- 6 temporal echoes, spaced `echoSpacing` frames apart (default 15)
- Each echo is tinted with a spectral color: Red → Yellow → Green → Cyan → Blue → Magenta
- Output averages 3 echoes per channel, so still images reproduce faithfully with no color cast
- Moving subjects: echoes separate in time → rainbow trails follow motion

**Flow Direction Color Mode (H):**
- Farneback optical flow computed at `FLOW_SCALE = 0.25` resolution, resized back up
- Per-pixel: `atan2(vy, vx)` → hue (direction encodes color), flow magnitude → saturation, pixel brightness → value
- Converted HSV → BGR inline without creating intermediate Mat
- Still areas produce desaturated output; motion reveals directional color

## Features

### Direction Modes
- **W** - Bottom to Top (newest at bottom)
- **S** - Top to Bottom (newest at top)
- **A** - Right to Left (newest at right)
- **D** - Left to Right (newest at left)

### Combination Modes
- **W+S** - Center-out vertical (press W then S within 0.5s)
- **A+D** - Center-out horizontal (press A then D within 0.5s)

### Special Effect Modes
- **M** - Motion Adaptive (displacement intensity scales with per-pixel motion)
- **C** - Chromatic Aberration (R/G/B channels pulled from slightly different frames)
- **X** - Motion Chromatic (motion-driven per-pixel RGB temporal split; still=no color, moving=full chroma spread)
- **P** - Prismatic Echo (6 spectral echoes spaced through the buffer; moving subjects leave rainbow trails)
- **H** - Flow Direction Color (optical flow direction → hue, magnitude → saturation; directional color from motion)
- **N** - Turbulence (motion history accumulation drives pixel displacement + chroma split + saturation boost)
- **G** - Temporal Ghost (7 person echoes through time on black; uses Apple Vision framework, no Python needed)
- **K** - Rainbow Ghost (like G but each echo tinted a cycling hue; color flows newest→oldest; no temporal fade)

### Controls
- **Arrow Up/Down** - Adjust speed (lines per frame)
- **R** - Reset speed to 1
- **F** - Toggle fullscreen
- **Q/ESC** - Quit

### UI Features
- Clean fullscreen view (no permanent overlays)
- Speed indicator appears for 3 seconds when changed
- Terminal logging of FPS and current mode

## Installation & Setup

### Prerequisites

**macOS:**
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OpenCV
brew install opencv

# Install Xcode Command Line Tools
xcode-select --install
```

**Raspberry Pi:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopencv-dev
```

### Compilation

```bash
# Navigate to project directory
cd /path/to/project

# Recommended: with OpenMP (parallel displacement, ~2-4x faster)
brew install libomp
g++ -std=c++17 -O3 -march=native -Xpreprocessor -fopenmp \
  -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp \
  time_mirror.mm -o time_mirror `pkg-config --cflags --libs opencv4`

# Without OpenMP (still benefits from -O3 and capture thread)
g++ -std=c++17 -O3 -march=native time_mirror.mm -o time_mirror `pkg-config --cflags --libs opencv4`

# Run
./time_mirror
```

### Troubleshooting

**"opencv4.pc not found"**
```bash
# Try opencv instead of opencv4
g++ -std=c++17 -O3 -march=native time_mirror.mm -o time_mirror `pkg-config --cflags --libs opencv`
```

**Camera permission denied (macOS)**
- System Preferences → Security & Privacy → Camera
- Enable for Terminal

**Segmentation fault**
- Usually means OpenCV version mismatch
- Try: `brew upgrade opencv`

## Configuration

Edit these constants in `time_mirror.mm`:

```cpp
const int FRAME_WIDTH = 1920;   // Camera resolution width
const int FRAME_HEIGHT = 1080;  // Camera resolution height
const int BUFFER_SIZE = 200;    // Number of frames to store
```

### Performance Tuning

| Setting | Effect | Trade-off |
|---------|--------|-----------|
| Lower resolution | Higher FPS | Lower quality |
| Smaller buffer | Less memory, faster | Shorter time span |
| Reduce update speed | Smoother | Less dramatic effect |

## Technical Challenges Solved

### 1. Memory Layout Performance
**Problem:** Column operations were 40% slower than row operations  
**Solution:** Rows are contiguous in memory, columns are not. Tried transposition but overhead was worse. Accepted the difference and optimized what we could.

### 2. Python Performance Ceiling
**Problem:** Python couldn't hit 30 FPS at 1080p  
**Solution:** Rewrote in C++ for 2-3x speedup. Eliminated interpreter overhead and used direct OpenCV C++ API.

### 3. Center Line Timing
**Problem:** Center line in combo modes showed old frame instead of newest  
**Solution:** Used `buffer_index - 1` since buffer_index points to next write location, not most recent.

### 4. Arrow Key Detection on Mac
**Problem:** Standard OpenCV key codes didn't work
**Solution:** Mac uses codes 0/1 for up/down arrows instead of 82/84.

### 5. Single-threaded Pipeline Stalling at Camera
**Problem:** `cap >> frame` blocks until the camera delivers a frame, eating directly into the render budget and capping throughput at camera FPS regardless of processing speed.
**Solution:** Moved capture into a dedicated thread. Capture and render now run concurrently. `writeIndex` uses `std::atomic` with `memory_order_release` on write and `memory_order_acquire` on read — lock-free and safe with no tearing.

### 6. Per-frame Heap Allocation
**Problem:** `Mat::zeros(height, width, ...)` allocated and zero-filled 6 MB on the heap every frame.
**Solution:** Pre-allocate `output` Mat once before the loop. Since every pixel is overwritten by `copyTo`, no zeroing is needed.

### 7. Stale Camera Frames
**Problem:** OpenCV's default internal camera queue holds 3-4 frames, so the grabbed frame could be up to 4 frames old even with a 60 fps camera.
**Solution:** `cap.set(CAP_PROP_BUFFERSIZE, 1)` reduces the queue to 1, ensuring the capture thread always gets the freshest frame.

### 8. Displacement Loop Not Parallelised
**Problem:** Row/column copies in `applyTimeDisplacement` ran serially on one core even though every iteration is independent.
**Solution:** `#pragma omp parallel for schedule(static)` distributes the work across all available cores. On a 4-core machine this alone approaches a 4x speedup for the displacement step.

## Code Structure

### Main Components

```cpp
// Core data structures
vector<Mat>  frameBuffer;         // Circular buffer of frames (pre-allocated)
atomic<int>  writeIndex;          // Next write slot; release/acquire for thread safety
atomic<int>  updateSpeed;         // Lines advanced per captured frame
atomic<bool> running;             // Shared stop signal for capture thread
string       currentMode;         // Current direction mode (main thread only)

// Special mode parameters
int chromaSpread = 20;            // X mode: max frame spread at full motion
int echoSpacing  = 15;            // P mode: frames between each of 6 echoes
float flowSensitivity = 10.0f;   // H mode: flow magnitude scaling

// Optical flow buffers (H mode)
Mat flowMap, smallFlowBuf, smallPrevBGR, smallCurrBGR, prevGraySmall, currGraySmall;

// Key functions
void captureLoop(VideoCapture&)   // Capture thread: grab, flip, write to buffer
void applyTimeDisplacement()      // Apply effect (OpenMP parallel loops)
void drawSpeedOverlay()           // Show speed indicator
string checkForCombo()            // Detect key combinations
```

### Processing Pipeline

```
Capture thread (runs concurrently):
    1. Grab frame from camera
    2. Mirror horizontally (flip)
    3. Write to frameBuffer[writeIndex]
    4. Advance writeIndex (atomic release)

Main thread (render loop):
    1. Read writeIndex (atomic acquire)
    2. Apply time displacement in parallel (OpenMP)
    3. Draw UI overlays if needed
    4. Display result
    5. Handle keyboard input
```

## Version History

### Processing Versions (v1-v6)
- Initial prototypes
- Basic direction controls
- Speed controls added
- Webcam integration working

### Python Versions (v7-v23)
- OpenCV integration
- WASD controls
- Combo mode detection
- Resolution optimization attempts
- GPU acceleration attempt (not effective on Mac)

### C++ Version (v1)
- Complete rewrite for performance
- Fullscreen toggle
- Clean UI with temporary overlays
- Mac-specific key code fixes
- Bounds checking and safety improvements

### C++ Version (v2 — 60 FPS optimisation)
- Dedicated capture thread (producer/consumer, lock-free via `std::atomic`)
- OpenMP parallel displacement loops (`-fopenmp`, all modes)
- Pre-allocated output `Mat` (eliminates per-frame 6 MB heap allocation)
- `CAP_PROP_BUFFERSIZE = 1` (always grab freshest camera frame)
- Compiler flags `-O3 -march=native` (auto-vectorisation, SIMD)
- Startup log shows OpenMP thread count for verification

### C++ Version (v3 — Special Effect Modes)
- **M: Motion Adaptive** — per-pixel motion map (absdiff + Gaussian blur) drives displacement intensity
- **C: Chromatic Aberration** — R/G/B channels pulled from slightly different frame offsets
- **X: Motion Chromatic** — motion map scales per-pixel RGB channel temporal split; still areas show no color, moving areas show vivid chromatic split
- **P: Prismatic Echo** — 6 spectral echoes (red/yellow/green/cyan/blue/magenta) spaced through buffer; still images reproduce faithfully, motion creates rainbow trails
- **H: Flow Direction Color** — Farneback optical flow direction mapped to hue, magnitude to saturation; directional color reveals motion flow patterns

## Modes Tried and Dropped

These modes were implemented and tested but removed after evaluation.

### T: Color Motion Trails
- **Approach:** IIR accumulation buffer with cycling hue tint applied each frame; motion caused hue to shift
- **Why dropped:** The effect looked like a color-cycled version of the standard direction modes. The hue cycling was not tied to motion magnitude or direction in a visually meaningful way — it just gradually shifted all colors uniformly.
- **Lesson:** Color should be driven by motion properties (direction, magnitude), not time alone.

### U: Motion Hue Rotation
- **Approach:** Motion intensity rotated per-pixel hue in HSV space; high-motion pixels shifted more
- **Why dropped:** Visually interesting but not as compelling as Prismatic Echo or Flow Direction Color. Dropped when user chose P and H as the two keepers from a batch of five new modes.

### G: Temporal Hue Gradient
- **Approach:** Each frame in the buffer assigned a hue based on its age; output pixels tinted by the hue of the frame they were drawn from
- **Why dropped:** Same batch drop as U. The time-based color gradient was subtle and less dramatic than the motion-reactive modes.

### Z: Chromatic Ripple
- **Approach:** Sine-wave ripple pattern applied to per-pixel frame offsets, with R/G/B channels rippling at slightly different phases
- **Why dropped:** Same batch drop as U and G. The ripple pattern felt mechanical and didn't interact with the video content meaningfully.

### O: Optical Flow Trails
- **Approach:** Farneback flow vectors used to warp an accumulation buffer; pixels smeared in flow direction over time
- **Why dropped:** Explicitly removed by user request after initial implementation. The smearing created a murky look rather than clean trails.

### V: Flow Direction Split
- **Approach:** Flow direction (left/right vs. up/down component) determined which of two frame buffers to draw from per pixel
- **Why dropped:** Explicitly removed alongside O. The two-buffer split created jarring seams rather than smooth chromatic separation.

---

## Future Enhancements

### Potential Features
- [ ] Record output to video file
- [ ] Multiple buffer sizes for different effects
- [ ] Diagonal sweep modes
- [ ] Radial/polar coordinate modes
- [ ] MIDI controller integration for live performance
- [ ] Multiple camera support
- [ ] GPU acceleration with Metal (Mac) or CUDA (NVIDIA)

### Mode Ideas (Not Yet Implemented)
- [ ] **Long Exposure** — accumulate N frames with equal weight; moving subjects blur into streaks, still areas stay sharp
- [ ] **Echo Ghost** — overlay current frame with 3-5 semi-transparent older frames; sharp ghost images rather than blurred trails
- [ ] **Directional Chroma** — like X (Motion Chromatic) but channel split direction follows motion direction (horizontal motion → horizontal split, vertical → vertical)
- [ ] **Angle Clock** — divide frame into radial sectors; each sector sweeps through time like a clock hand, sectors at different angles show different moments
- ~~**Scan Glitch**~~ Removed — replaced by Turbulence (N)

### Performance Ideas
- Shader-based implementation (OpenGL/Metal)
- ~~Multi-threaded processing~~ ✓ Done (capture thread + OpenMP)
- Hardware-accelerated video encoding
- Transposed buffer for column modes (make A/D/AD as fast as W/S/WS)
- GPU displacement via `cv::UMat` + OpenCL

## Hardware Recommendations

### Minimum Requirements
- CPU: Intel i5 or equivalent
- RAM: 4GB
- Camera: 720p @ 30fps
- OS: macOS 10.14+, Linux, Windows 10+

### Recommended Setup
- CPU: Intel i7 / Apple M1 or better
- RAM: 8GB+
- Camera: 1080p @ 60fps
- OS: macOS 12+ / Ubuntu 22.04+

### Budget Option
- Raspberry Pi 5 (4GB model) - ~$60
- Should achieve 20-25 FPS at 1080p

## Credits

**Developed by:** Ed Lewis  
**Assistant:** Claude (Anthropic)  
**Technology Stack:** C++17, OpenCV 4.x  
**Inspired by:** Slit-scan photography, time-displacement effects

## License

This project is open source. Feel free to use, modify, and distribute.

## Contact & Support

For questions or contributions, please reach out through the project repository.

---

*Last updated: February 2026 (v3 — special effect modes)*
