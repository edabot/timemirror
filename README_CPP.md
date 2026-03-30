# Time Mirror Effect - C++ Version

High-performance C++ implementation for 60+ FPS at 1280x720.

## Installation on Mac

1. Install Homebrew (if not already installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install OpenCV:
```bash
brew install opencv
```

3. Compile the program:
```bash
g++ -std=c++17 time_mirror.cpp -o time_mirror `pkg-config --cflags --libs opencv4`
```

If you get an error about opencv4 not found, try:
```bash
g++ -std=c++17 time_mirror.cpp -o time_mirror `pkg-config --cflags --libs opencv`
```

4. Run the program:
```bash
./time_mirror
```

## Controls

- **W** - Bottom to top
- **S** - Top to bottom (default)
- **A** - Right to left
- **D** - Left to right
- **W then S** (within 0.5s) - Center-out vertical
- **A then D** (within 0.5s) - Center-out horizontal
- **Up Arrow** - Increase speed
- **Down Arrow** - Decrease speed
- **R** - Reset speed to 1
- **I** - Toggle info overlay
- **Q or ESC** - Quit

## Expected Performance

C++ version should achieve:
- **60+ FPS** for up/down modes (S, W, WS combo)
- **40-50 FPS** for left/right modes (A, D, AD combo)

This is 2-3x faster than the Python version!

## Troubleshooting

**Error: "opencv4.pc not found"**
- Try: `brew info opencv` to see the installed version
- Use `opencv` instead of `opencv4` in the compile command

**Camera permission denied**
- Go to System Preferences → Security & Privacy → Camera
- Grant permission to Terminal

**Compilation errors**
- Make sure you have Xcode command line tools: `xcode-select --install`
