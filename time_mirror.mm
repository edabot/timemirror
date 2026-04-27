/*
Time-based Funhouse Mirror Effect - C++ Version

Compilation on Mac:
  brew install opencv

  # Without OpenMP (simpler):
  g++ -std=c++17 -O3 -march=native time_mirror.cpp -o time_mirror `pkg-config --cflags --libs opencv4`

  # With OpenMP (recommended, ~2-4x faster displacement):
  brew install libomp
  g++ -std=c++17 -O3 -march=native -Xpreprocessor -fopenmp \
    -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp \
    time_mirror.cpp -o time_mirror `pkg-config --cflags --libs opencv4`

  ./time_mirror

Usage:
  W/S/A/D - Direction controls
  W+S within 0.5s - Center-out vertical
  A+D within 0.5s - Center-out horizontal
  Up/Down arrows - Speed control
  R - Reset speed
  F - Toggle fullscreen
  Q/ESC - Quit
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <thread>
#include <atomic>
#import <Vision/Vision.h>
#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace cv;
using namespace std;
using namespace chrono;
// Configuration
const int FRAME_WIDTH = 1920;
const int FRAME_HEIGHT = 1080;
const int BUFFER_SIZE = 200;

// Motion mode settings
const int MOTION_LOOKBACK = 10;  // Frames back for motion comparison (higher = reacts to slower motion)
const int MOTION_BLUR_SIZE = 31; // Spatial blur radius — larger spreads halos further (must be odd)

// Chromatic time shift settings (runtime-adjustable with Up/Down in chroma mode)
int chromaOffset = 23; // Frames between each colour channel (B=now, G=now-N, R=now-2N)

// Motion adaptive mode settings (runtime-adjustable with Up/Down in motion mode)
int motionMaxOffset = 40; // Max frames back at full motion (higher = more dreamy, lower = subtle)

// Motion-chromatic mode settings (runtime-adjustable with Up/Down in mchroma mode)
int chromaSpread = 40; // Max frames between channels at full motion (0-motion = no split)

// Ghost Echo mode (E) — motion-masked stack of 7 temporal echoes.
// Still areas → black. Moving subjects → layered ghost images from past frames.
int ghostSpacing = 8; // Frames between each of 7 echoes (Up/Down adjustable)

// Background Removal mode (B) — MOG2 Gaussian mixture background subtractor.
// Learns background automatically; only updates background model for background pixels,
// so a stationary subject does not bleed into the background estimate over time.

// Shared between capture thread and main thread.
// writeIndex uses release/acquire semantics so the main thread always
// sees a fully-written frame before the index advances past it.
vector<Mat> frameBuffer;
atomic<int> writeIndex{0};
atomic<int> updateSpeed{1};
atomic<bool> running{true};

// Motion mode working buffers — pre-allocated in main, used only in main thread
Mat motionMap;   // CV_32F, 0.0 (still) → 1.0 (max motion), per-pixel
Mat diffMat;     // CV_8UC3 scratch for absdiff
Mat grayDiff;    // CV_8U  scratch for grayscale + blur
Mat motionSmall; // CV_8U  scratch for motion blur at FLOW_SCALE resolution

// Prismatic echo settings (runtime-adjustable with Up/Down in prismatic mode)
int echoSpacing = 23; // Frames between each of 6 hue-tinted echoes (1 – BUFFER_SIZE/3)

// Optical flow mode working buffers — pre-allocated in main
// Flow is computed at FLOW_SCALE of full resolution for performance, then resized up.
const float FLOW_SCALE = 0.25f; // compute flow at 1/4 linear resolution (~16x fewer pixels)
Mat flowMap;                    // CV_32FC2, full-size interpolated flow vectors (vx, vy)
Mat smallFlowBuf;               // CV_32FC2, flow at reduced resolution
Mat smallPrevBGR;               // downscaled previous frame for flow input
Mat smallCurrBGR;               // downscaled current frame for flow input
Mat prevGraySmall;              // grayscale at small scale
Mat currGraySmall;              // grayscale at small scale
float flowSensitivity = 10.0f;  // flow px/frame (small-scale) that maps to full time offset

// Datamosh mode settings
// Accumulates per-pixel frame differences with IIR decay.
// Still areas → black; motion → vivid color trails that persist then fade.
Mat datamoshAccum;                 // CV_32FC3, signed accumulator
Mat datamoshDiffF;                 // CV_32FC3 scratch for the per-frame diff
float datamoshDecay = 0.92f;       // IIR decay per frame (higher = longer trails)
const float DATAMOSH_BOOST = 1.5f; // diff amplification before accumulation

// Flow Ripple mode (J) — directional color that advects with optical flow and decays over ~1 second.
// rippleBuffer holds accumulated per-pixel BGR color as float [0–255].
// Each frame: advect with remap, decay by rippleDecay, inject new color where flow is strong.
Mat rippleBuffer;          // CV_32FC3, full res, persists between frames
Mat rippleTmp;             // CV_32FC3, scratch for remap output
Mat rippleMapX;            // CV_32F,   backward-warp x coords built from flowMap
Mat rippleMapY;            // CV_32F,   backward-warp y coords built from flowMap
Mat ripple8;               // CV_8UC3,  converted for additive compositing
float rippleDecay = 0.93f; // per-frame IIR decay (~1s at 60fps: 0.93^60 ≈ 0.014)

// Turbulence mode (T) — per-pixel motion history drives pixel displacement + color separation.
// turbulenceMap accumulates motion over ~2 seconds via IIR decay.
// Still areas → near grayscale. Active areas → vivid distortion with chromatic split.
Mat turbulenceMap;        // CV_32F, 0–1, accumulated per-pixel motion history
vector<float> turbNoiseX; // pre-allocated sinf lookup table, size = actualWidth
vector<float> turbNoiseY; // pre-allocated cosf lookup table, size = actualWidth
int turbFrame = 0;
float turbDecay = 0.992f; // IIR decay (~2s half-life at 60fps)
float turbShift = 20.0f;  // max pixel displacement at full turbulence (Up/Down adjustable)

// Temporal Ghost mode (G) ─────────────────────────────────────────────────────
// maskBuffer is a circular buffer of CV_8U masks parallel to frameBuffer.
// segmentLoop() runs on a dedicated thread, feeding frames to the macOS
// Vision framework (VNGeneratePersonSegmentationRequest) and storing the
// resulting person masks in maskBuffer at full resolution.
// segReady tracks the most recently completed mask slot.
vector<Mat> maskBuffer;   // CV_8U, same slots as frameBuffer
atomic<int> segReady{-1}; // index of last written mask; -1 = not started
int tghostSpacing = 20;   // frames between each of TGHOST_ECHOES echoes
const int TGHOST_ECHOES = 7;
float rainbowHue = 0.0f;         // K mode: current base hue (0–360, cycles each frame)
float rainbowSpeed = 30.0f;      // K mode: degrees per second the hue advances
const int RAINBOW_HUE_STEP = 45; // degrees between consecutive echoes

// Main-thread-only state
string currentMode = "s";
map<char, steady_clock::time_point> lastKeyTime;
const double COMBO_WINDOW = 0.5;

// ── Capture thread ────────────────────────────────────────────────────────────
// Runs independently so cap >> frame never stalls the render loop.
void captureLoop(VideoCapture &cap)
{
    Mat frame;
    while (running)
    {
        cap >> frame;
        if (frame.empty())
        {
            running = false;
            break;
        }

        flip(frame, frame, 1);

        // Write the captured frame once, then copy from the warm slot for
        // subsequent speed steps — avoids re-reading the source frame from memory.
        int speed = updateSpeed.load(memory_order_relaxed);
        int idx = writeIndex.load(memory_order_relaxed);
        frame.copyTo(frameBuffer[idx]);
        writeIndex.store((idx + 1) % BUFFER_SIZE, memory_order_release);
        for (int i = 1; i < speed; i++)
        {
            int prev = idx;
            idx = writeIndex.load(memory_order_relaxed);
            frameBuffer[prev].copyTo(frameBuffer[idx]);
            writeIndex.store((idx + 1) % BUFFER_SIZE, memory_order_release);
        }
    }
}

// ── Segmentation thread (Vision framework) ────────────────────────────────────
// Uses VNGeneratePersonSegmentationRequest (built into macOS 12+, no Python needed).
// Feeds the latest captured frame as a CVPixelBuffer, reads back the person mask,
// upscales to full resolution, and stores in maskBuffer.
// segReady is only advanced after a complete write so the render thread reads safely.
void segmentLoop(int actualWidth, int actualHeight)
{
    @autoreleasepool {
        VNGeneratePersonSegmentationRequest *request =
            [[VNGeneratePersonSegmentationRequest alloc] init];
        request.qualityLevel = VNGeneratePersonSegmentationRequestQualityLevelFast;
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8;

        // VNSequenceRequestHandler accumulates internal temporal state each frame.
        // After ~28 k frames (~8 min at 60 fps) that state exhausts GPU memory and
        // performRequests hangs indefinitely. Reset every SEG_RESET_INTERVAL frames
        // to flush accumulated state; -fobjc-arc ensures the old handler is released.
        const int SEG_RESET_INTERVAL = 1800; // ~30 s at 60 fps
        VNSequenceRequestHandler *seqHandler = [[VNSequenceRequestHandler alloc] init];
        int segFrameCount = 0;

        Mat bgraMat;
        int lastSeg = -1;

        while (running)
        {
            int latest = (writeIndex.load(memory_order_acquire) - 1 + BUFFER_SIZE) % BUFFER_SIZE;
            if (latest == lastSeg)
            {
                this_thread::yield();
                continue;
            }

            @autoreleasepool {
                if (++segFrameCount % SEG_RESET_INTERVAL == 0)
                    seqHandler = [[VNSequenceRequestHandler alloc] init];
                cv::cvtColor(frameBuffer[latest], bgraMat, COLOR_BGR2BGRA);

                // Allocate a Vision-owned pixel buffer and copy the frame into it.
                // CVPixelBufferCreateWithBytes (zero-copy) is unsafe here because
                // Vision may retain GPU references to the buffer asynchronously
                // past performRequests return, corrupting bgraMat on the next frame.
                CVPixelBufferRef pixelBuffer = nullptr;
                CVReturn status = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    (size_t)bgraMat.cols, (size_t)bgraMat.rows,
                    kCVPixelFormatType_32BGRA,
                    nullptr, &pixelBuffer);

                if (status != kCVReturnSuccess || !pixelBuffer)
                {
                    lastSeg = latest;
                    continue;
                }

                CVPixelBufferLockBaseAddress(pixelBuffer, 0);
                uint8_t *dst = (uint8_t *)CVPixelBufferGetBaseAddress(pixelBuffer);
                size_t dstStride = CVPixelBufferGetBytesPerRow(pixelBuffer);
                for (int row = 0; row < bgraMat.rows; row++)
                    memcpy(dst + row * dstStride, bgraMat.ptr(row), (size_t)bgraMat.cols * 4);
                CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);

                NSError *err = nil;
                [seqHandler performRequests:@[request]
                            onCVPixelBuffer:pixelBuffer
                                      error:&err];
                CVPixelBufferRelease(pixelBuffer);

                if (!err && request.results.count > 0)
                {
                    VNPixelBufferObservation *obs =
                        (VNPixelBufferObservation *)request.results[0];
                    CVPixelBufferRef maskPB = obs.pixelBuffer;

                    CVPixelBufferLockBaseAddress(maskPB, kCVPixelBufferLock_ReadOnly);
                    int maskW = (int)CVPixelBufferGetWidth(maskPB);
                    int maskH = (int)CVPixelBufferGetHeight(maskPB);
                    void *base = CVPixelBufferGetBaseAddress(maskPB);
                    size_t stride = CVPixelBufferGetBytesPerRow(maskPB);

                    // Resize mask to full resolution and feather edges.
                    // Vision returns 255=person, 0=background — matches maskBuffer convention.
                    Mat maskSmall(maskH, maskW, CV_8U, base, stride);
                    cv::resize(maskSmall, maskBuffer[latest],
                               cv::Size(actualWidth, actualHeight), 0, 0, INTER_LINEAR);
                    CVPixelBufferUnlockBaseAddress(maskPB, kCVPixelBufferLock_ReadOnly);

                    cv::GaussianBlur(maskBuffer[latest], maskBuffer[latest], cv::Size(0, 0), 12.0);

                    // Propagate mask to slots skipped since lastSeg so echo indices
                    // never land on stale masks from a previous buffer wrap.
                    if (lastSeg >= 0)
                    {
                        int span = (latest - lastSeg + BUFFER_SIZE) % BUFFER_SIZE;
                        for (int i = 1; i < span; i++)
                            maskBuffer[latest].copyTo(maskBuffer[(lastSeg + i) % BUFFER_SIZE]);
                    }

                    lastSeg = latest;
                    segReady.store(latest, memory_order_release);
                }
                else
                {
                    lastSeg = latest;
                }
            }
        }
    }
}

// ── Key combo detection ───────────────────────────────────────────────────────
string checkForCombo(char keyPressed)
{
    auto currentTime = steady_clock::now();
    lastKeyTime[keyPressed] = currentTime;

    if (keyPressed == 'w' || keyPressed == 's')
    {
        char other = (keyPressed == 'w') ? 's' : 'w';
        if (lastKeyTime.count(other))
        {
            double dt = duration<double>(currentTime - lastKeyTime[other]).count();
            if (dt < COMBO_WINDOW)
                return "ws";
        }
        return string(1, keyPressed);
    }
    if (keyPressed == 'a' || keyPressed == 'd')
    {
        char other = (keyPressed == 'a') ? 'd' : 'a';
        if (lastKeyTime.count(other))
        {
            double dt = duration<double>(currentTime - lastKeyTime[other]).count();
            if (dt < COMBO_WINDOW)
                return "ad";
        }
        return string(1, keyPressed);
    }
    return string(1, keyPressed);
}

// ── Time displacement ─────────────────────────────────────────────────────────
// output is pre-allocated by the caller (no heap allocation per frame).
// bufIdx is the current writeIndex snapshot; every pixel is overwritten so
// output does not need to be zeroed.
//
// OpenMP parallelises the per-row / per-column loops across all cores.
// Each iteration is independent (different row/col of output), so there are
// no data races.
void applyTimeDisplacement(Mat &output, int width, int height, int bufIdx)
{
    if (currentMode == "ws")
    {
        int centerY = height / 2;
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            int dist = abs(y - centerY);
            int frameOffset = (bufIdx - 1 - (dist * BUFFER_SIZE / max(centerY, 1)) + BUFFER_SIZE * 2) % BUFFER_SIZE;
            frameBuffer[frameOffset].row(y).copyTo(output.row(y));
        }
    }
    else if (currentMode == "ad")
    {
        int centerX = width / 2;
        // Precompute frame index per column, then copy row-by-row (cache-friendly)
        vector<int> colFrame(width);
        for (int x = 0; x < width; x++)
        {
            int dist = abs(x - centerX);
            colFrame[x] = (bufIdx - 1 - (dist * BUFFER_SIZE / max(centerX, 1)) + BUFFER_SIZE * 2) % BUFFER_SIZE;
        }
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            Vec3b *outRow = output.ptr<Vec3b>(y);
            for (int x = 0; x < width; x++)
                outRow[x] = frameBuffer[colFrame[x]].ptr<Vec3b>(y)[x];
        }
    }
    else if (currentMode == "w")
    {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            int frameOffset = (bufIdx - ((height - 1 - y) * BUFFER_SIZE / height) + BUFFER_SIZE) % BUFFER_SIZE;
            frameBuffer[frameOffset].row(y).copyTo(output.row(y));
        }
    }
    else if (currentMode == "s")
    {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            int frameOffset = (bufIdx - (y * BUFFER_SIZE / height) + BUFFER_SIZE) % BUFFER_SIZE;
            frameBuffer[frameOffset].row(y).copyTo(output.row(y));
        }
    }
    else if (currentMode == "a")
    {
        vector<int> colFrame(width);
        for (int x = 0; x < width; x++)
            colFrame[x] = (bufIdx - ((width - 1 - x) * BUFFER_SIZE / width) + BUFFER_SIZE) % BUFFER_SIZE;
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            Vec3b *outRow = output.ptr<Vec3b>(y);
            for (int x = 0; x < width; x++)
                outRow[x] = frameBuffer[colFrame[x]].ptr<Vec3b>(y)[x];
        }
    }
    else if (currentMode == "d")
    {
        vector<int> colFrame(width);
        for (int x = 0; x < width; x++)
            colFrame[x] = (bufIdx - (x * BUFFER_SIZE / width) + BUFFER_SIZE) % BUFFER_SIZE;
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            Vec3b *outRow = output.ptr<Vec3b>(y);
            for (int x = 0; x < width; x++)
                outRow[x] = frameBuffer[colFrame[x]].ptr<Vec3b>(y)[x];
        }
    }
    else if (currentMode == "motion")
    {
// Per-pixel: still areas sample the current frame, moving areas sample
// further back in time — motion creates long temporal trails.
// motionMap must be computed by the caller before this function.
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            const float *motionRow = motionMap.ptr<float>(y);
            Vec3b *outRow = output.ptr<Vec3b>(y);
            for (int x = 0; x < width; x++)
            {
                // motion=0 → bufIdx-1 (most recent); motion=1 → bufIdx (oldest)
                int offset = (int)(motionRow[x] * motionMaxOffset);
                int idx = (bufIdx - 1 - offset + BUFFER_SIZE * 2) % BUFFER_SIZE;
                outRow[x] = frameBuffer[idx].ptr<Vec3b>(y)[x];
            }
        }
    }
    else if (currentMode == "chroma")
    {
        // B channel from most recent frame, G from CHROMA_OFFSET frames ago,
        // R from 2×CHROMA_OFFSET frames ago. Still objects look normal; moving
        // objects leave blue→green→red colour trails.
        // mixChannels does this in a single pass with no intermediate allocations.
        int idx0 = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
        int idx1 = (bufIdx - 1 - chromaOffset + BUFFER_SIZE * 2) % BUFFER_SIZE;
        int idx2 = (bufIdx - 1 - chromaOffset * 2 + BUFFER_SIZE * 2) % BUFFER_SIZE;

        const Mat sources[] = {frameBuffer[idx0], frameBuffer[idx1], frameBuffer[idx2]};
        const int fromTo[] = {0, 0, 4, 1, 8, 2}; // B←frame0, G←frame1, R←frame2
        cv::mixChannels(sources, 3, &output, 1, fromTo, 3);
    }
    else if (currentMode == "mchroma")
    {
        // Per-pixel motion-adaptive chromatic aberration.
        // Still pixels (motion=0) show the current frame unchanged.
        // Moving pixels get B/G/R channels pulled from progressively older frames,
        // with the spread (in frames) scaling linearly with local motion intensity.
        // motionMap must be computed by the caller before this function.
        // idxB is constant for the whole frame — hoist out of both loops
        int idxB = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            const float *motionRow = motionMap.ptr<float>(y);
            Vec3b *outRow = output.ptr<Vec3b>(y);
            const Vec3b *rowB = frameBuffer[idxB].ptr<Vec3b>(y);
            for (int x = 0; x < width; x++)
            {
                int spread = (int)(motionRow[x] * chromaSpread);
                int idxG = (bufIdx - 1 - spread + BUFFER_SIZE * 2) % BUFFER_SIZE;
                int idxR = (bufIdx - 1 - spread * 2 + BUFFER_SIZE * 4) % BUFFER_SIZE;
                outRow[x][0] = rowB[x][0];                            // B ← newest
                outRow[x][1] = frameBuffer[idxG].ptr<Vec3b>(y)[x][1]; // G ← spread ago
                outRow[x][2] = frameBuffer[idxR].ptr<Vec3b>(y)[x][2]; // R ← 2× spread ago
            }
        }
    }
    else if (currentMode == "prismatic")
    {
        // 6 temporal echoes, each tinted with an evenly-spaced hue (red→yellow→green→
        // cyan→blue→magenta). Echoes are additively combined per channel.
        // Still areas: all echoes overlap → image reproduces faithfully.
        // Moving areas: echoes separate in space → rainbow prismatic smear.
        // Channel assignments keep each channel covered by exactly 3 echoes so
        // normalising by /3 reproduces the original colour in still regions.
        //   R ← echoes 0(red), 1(yellow), 5(magenta)
        //   G ← echoes 1(yellow), 2(green), 3(cyan)
        //   B ← echoes 3(cyan),  4(blue),  5(magenta)
        // Frame indices are independent of y — hoist out of the parallel loop
        int fi[6];
        for (int e = 0; e < 6; e++)
            fi[e] = (bufIdx - 1 - e * echoSpacing + BUFFER_SIZE * 4) % BUFFER_SIZE;
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            const Vec3b *src[6];
            for (int e = 0; e < 6; e++)
                src[e] = frameBuffer[fi[e]].ptr<Vec3b>(y);
            Vec3b *outRow = output.ptr<Vec3b>(y);
            for (int x = 0; x < width; x++)
            {
                float sumR = src[0][x][2] + src[1][x][2] + src[5][x][2];
                float sumG = src[1][x][1] + src[2][x][1] + src[3][x][1];
                float sumB = src[3][x][0] + src[4][x][0] + src[5][x][0];
                outRow[x][0] = (uchar)min(255.0f, sumB * (1.0f / 3.0f));
                outRow[x][1] = (uchar)min(255.0f, sumG * (1.0f / 3.0f));
                outRow[x][2] = (uchar)min(255.0f, sumR * (1.0f / 3.0f));
            }
        }
    }
    else if (currentMode == "datamosh")
    {
        // datamoshAccum is a CV_32FC3 float buffer updated by the preprocessing block.
        // convertTo with CV_8UC3 saturates values above 255 to 255 — no manual clamping needed.
        // Still areas: accum → 0 → black.  Moving areas: bright colour that decays over time.
        datamoshAccum.convertTo(output, CV_8UC3);
    }
    else if (currentMode == "ghostecho")
    {
        // 7 temporal echoes stacked with triangular weights (newest = brightest).
        // motionMap masks the output: still areas → black, moving areas → ghost stack.
        // Frame indices are hoisted out of the parallel loop — one per echo.
        int fi[7];
        for (int e = 0; e < 7; e++)
            fi[e] = (bufIdx - 1 - e * ghostSpacing + BUFFER_SIZE * 10) % BUFFER_SIZE;
        // Triangular weights: e=0 → 7/28, e=1 → 6/28 … e=6 → 1/28
        static const float w[7] = {7 / 28.f, 6 / 28.f, 5 / 28.f, 4 / 28.f, 3 / 28.f, 2 / 28.f, 1 / 28.f};
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            const float *motRow = motionMap.ptr<float>(y);
            Vec3b *outRow = output.ptr<Vec3b>(y);
            for (int x = 0; x < width; x++)
            {
                float m = motRow[x];
                if (m < 0.05f)
                {
                    outRow[x] = Vec3b(0, 0, 0);
                    continue;
                }
                float B = 0, G = 0, R = 0;
                for (int e = 0; e < 7; e++)
                {
                    const Vec3b &p = frameBuffer[fi[e]].ptr<Vec3b>(y)[x];
                    B += p[0] * w[e];
                    G += p[1] * w[e];
                    R += p[2] * w[e];
                }
                // Ramp up to full brightness as motion increases
                float boost = min(m * 3.0f, 1.0f);
                outRow[x][0] = (uchar)min(255.f, B * boost * 2.0f);
                outRow[x][1] = (uchar)min(255.f, G * boost * 2.0f);
                outRow[x][2] = (uchar)min(255.f, R * boost * 2.0f);
            }
        }
    }
    else if (currentMode == "timeghost")
    {
        // Composite TGHOST_ECHOES silhouettes of the person from different moments in time.
        // maskBuffer provides per-pixel person masks (255=person, 0=background).
        // Echoes are painted oldest→newest; newest always wins at overlapping pixels.
        // Fade: oldest echo = 30% brightness, newest = 100%.
        // Use segReady as the base — it is only advanced AFTER a full mask write
        // completes, so maskBuffer[base] is always fully written and safe to read.
        // Using bufIdx-1 instead would race with the segment thread writing to that
        // same slot, producing horizontal banding artifacts.
        int base = segReady.load(memory_order_acquire);
        if (base < 0)
        {
            output.setTo(Scalar(0, 0, 0));
            return;
        }
        int fi[TGHOST_ECHOES];
        for (int e = 0; e < TGHOST_ECHOES; e++)
            fi[e] = (base - e * tghostSpacing + BUFFER_SIZE * 10) % BUFFER_SIZE;

#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            Vec3b *outRow = output.ptr<Vec3b>(y);
            for (int x = 0; x < width; x++)
            {
                outRow[x] = Vec3b(0, 0, 0);
                // Paint oldest to newest so newest overwrites at current position.
                // Soft mask alpha feathers edges; temporal fade dims older echoes.
                for (int e = TGHOST_ECHOES - 1; e >= 0; e--)
                {
                    float maskAlpha = maskBuffer[fi[e]].ptr<uchar>(y)[x] / 255.0f;
                    if (maskAlpha < 0.05f)
                        continue;
                    float fade = 0.30f + 0.70f * (1.0f - (float)e / (TGHOST_ECHOES - 1));
                    float total = maskAlpha * fade;
                    const Vec3b &p = frameBuffer[fi[e]].ptr<Vec3b>(y)[x];
                    outRow[x][0] = (uchar)(p[0] * total);
                    outRow[x][1] = (uchar)(p[1] * total);
                    outRow[x][2] = (uchar)(p[2] * total);
                }
            }
        }
    }
    else if (currentMode == "rainbowghost")
    {
        // Like Temporal Ghost but each echo is tinted a single hue instead of using
        // natural colour. Hues are spaced RAINBOW_HUE_STEP degrees apart and the base
        // hue advances each frame so colours appear to travel down through the echoes.
        // Per-echo BGR precomputed outside the pixel loop (6 values, not per-pixel).
        int base = segReady.load(memory_order_acquire);
        if (base < 0)
        {
            output.setTo(Scalar(0, 0, 0));
            return;
        }

        int fi[TGHOST_ECHOES];
        for (int e = 0; e < TGHOST_ECHOES; e++)
            fi[e] = (base - e * tghostSpacing + BUFFER_SIZE * 10) % BUFFER_SIZE;

        // Precompute hue→BGR for each echo (no fade — all echoes at full brightness)
        float eB[TGHOST_ECHOES], eG[TGHOST_ECHOES], eR[TGHOST_ECHOES];
        for (int e = 0; e < TGHOST_ECHOES; e++)
        {
            float hue = fmod(rainbowHue - e * RAINBOW_HUE_STEP + 360.0f * TGHOST_ECHOES, 360.0f);
            float h6 = hue / 60.0f;
            int hi = (int)h6 % 6;
            float f = h6 - (int)h6;
            float q = 1.0f - f;
            switch (hi)
            {
            case 0:
                eR[e] = 1;
                eG[e] = f;
                eB[e] = 0;
                break;
            case 1:
                eR[e] = q;
                eG[e] = 1;
                eB[e] = 0;
                break;
            case 2:
                eR[e] = 0;
                eG[e] = 1;
                eB[e] = f;
                break;
            case 3:
                eR[e] = 0;
                eG[e] = q;
                eB[e] = 1;
                break;
            case 4:
                eR[e] = f;
                eG[e] = 0;
                eB[e] = 1;
                break;
            default:
                eR[e] = 1;
                eG[e] = 0;
                eB[e] = q;
                break;
            }
        }

#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            Vec3b *outRow = output.ptr<Vec3b>(y);
            for (int x = 0; x < width; x++)
            {
                outRow[x] = Vec3b(0, 0, 0);
                for (int e = TGHOST_ECHOES - 1; e >= 0; e--)
                {
                    float maskAlpha = maskBuffer[fi[e]].ptr<uchar>(y)[x] / 255.0f;
                    if (maskAlpha < 0.05f)
                        continue;
                    const Vec3b &p = frameBuffer[fi[e]].ptr<Vec3b>(y)[x];
                    float lum = (0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2]) / 255.0f;
                    float brightness = lum * maskAlpha * 255.0f;
                    outRow[x][0] = (uchar)min(255.0f, eB[e] * brightness);
                    outRow[x][1] = (uchar)min(255.0f, eG[e] * brightness);
                    outRow[x][2] = (uchar)min(255.0f, eR[e] * brightness);
                }
            }
        }
    }
    else if (currentMode == "flowripple")
    {
        // rippleBuffer is maintained by the preprocessing block (advect + decay + inject).
        // Convert to 8-bit (saturating) and add additively over the current frame so
        // still areas show the live video and moving areas glow with directional ripple color.
        int recent = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
        rippleBuffer.convertTo(ripple8, CV_8UC3, 1.0);
        cv::add(frameBuffer[recent], ripple8, output);
    }
    else if (currentMode == "turbulence")
    {
        // turbulenceMap is maintained by the preprocessing block.
        // Per-pixel: turbulence level (0=still, 1=max) drives:
        //   • animated sine-wave displacement (scaled by turbShift)
        //   • chromatic split: B/G/R sampled from different x offsets
        //   • saturation: 0 turbulence → grayscale, full → 2.5× vivid
        // xNoiseX/Y are precomputed per frame to avoid trig in the inner loop.
        int recent = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
        float ta = turbFrame * 0.04f;
        for (int x = 0; x < width; x++)
        {
            turbNoiseX[x] = sinf(x * 0.04f + ta * 1.1f);
            turbNoiseY[x] = cosf(x * 0.04f - ta * 0.7f);
        }
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            float rowNX = cosf(y * 0.04f + ta * 0.6f);
            float rowNY = sinf(y * 0.04f - ta * 0.8f);
            const float *turbRow = turbulenceMap.ptr<float>(y);
            Vec3b *outRow = output.ptr<Vec3b>(y);
            for (int x = 0; x < width; x++)
            {
                float t = turbRow[x];
                if (t < 0.01f)
                {
                    Vec3b pix = frameBuffer[recent].ptr<Vec3b>(y)[x];
                    int g = (int)(0.114f * pix[0] + 0.587f * pix[1] + 0.299f * pix[2]);
                    outRow[x] = Vec3b((uchar)g, (uchar)g, (uchar)g);
                    continue;
                }
                float nx = (turbNoiseX[x] + rowNX) * 0.5f;
                float ny = (turbNoiseY[x] + rowNY) * 0.5f;
                int dx = (int)(nx * t * turbShift);
                int dy = (int)(ny * t * turbShift);
                // Each channel displaced in a different direction for vivid separation.
                // B: push opposite to main displacement; R: amplified main direction;
                // G: perpendicular. Spread scales with turbulence (max ~50px per channel).
                float spread = t * 50.0f;
                int sxB = max(0, min(x + dx - (int)(spread), width - 1));
                int syB = max(0, min(y + dy + (int)(spread * 0.5f), height - 1));
                int sxG = max(0, min(x + dx + (int)(spread * 0.3f), width - 1));
                int syG = max(0, min(y + dy - (int)(spread * 0.6f), height - 1));
                int sxR = max(0, min(x + dx + (int)(spread), width - 1));
                int syR = max(0, min(y + dy + (int)(spread * 0.4f), height - 1));
                float B = frameBuffer[recent].ptr<Vec3b>(syB)[sxB][0];
                float G = frameBuffer[recent].ptr<Vec3b>(syG)[sxG][1];
                float R = frameBuffer[recent].ptr<Vec3b>(syR)[sxR][2];
                float gray = 0.114f * B + 0.587f * G + 0.299f * R;
                float satFactor = t * 2.5f;
                outRow[x][0] = (uchar)max(0, min((int)(gray + (B - gray) * satFactor), 255));
                outRow[x][1] = (uchar)max(0, min((int)(gray + (G - gray) * satFactor), 255));
                outRow[x][2] = (uchar)max(0, min((int)(gray + (R - gray) * satFactor), 255));
            }
        }
    }
    else if (currentMode == "flowhue")
    {
        // Each pixel: hue = optical flow direction, saturation = flow speed,
        // value = pixel brightness from current frame.
        // Still areas → grayscale. Moving areas → vivid directional colour:
        // right=red, down=yellow, left=cyan, up=blue, diagonals=in-between.
        // flowMap must be computed by the caller before this function.
        const float PI2 = 2.0f * 3.14159265f;
        int recent = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            const Vec2f *flowRow = flowMap.ptr<Vec2f>(y);
            const Vec3b *srcRow = frameBuffer[recent].ptr<Vec3b>(y);
            Vec3b *outRow = output.ptr<Vec3b>(y);
            for (int x = 0; x < width; x++)
            {
                float vx = flowRow[x][0];
                float vy = flowRow[x][1];
                float mag = sqrtf(vx * vx + vy * vy);
                float sat = min(mag / flowSensitivity, 1.0f);
                Vec3b pix = srcRow[x];
                float val = (pix[0] * 0.114f + pix[1] * 0.587f + pix[2] * 0.299f) * (1.0f / 255.0f);
                float hue = (atan2f(vy, vx) + 3.14159265f) / PI2 * 360.0f;
                float c = val * sat;
                float h6 = hue / 60.0f;
                float xc = c * (1.0f - fabsf(fmodf(h6, 2.0f) - 1.0f));
                float m = val - c;
                float rp, gp, bp;
                switch ((int)h6 % 6)
                {
                case 0:
                    rp = c;
                    gp = xc;
                    bp = 0;
                    break;
                case 1:
                    rp = xc;
                    gp = c;
                    bp = 0;
                    break;
                case 2:
                    rp = 0;
                    gp = c;
                    bp = xc;
                    break;
                case 3:
                    rp = 0;
                    gp = xc;
                    bp = c;
                    break;
                case 4:
                    rp = xc;
                    gp = 0;
                    bp = c;
                    break;
                default:
                    rp = c;
                    gp = 0;
                    bp = xc;
                    break;
                }
                outRow[x][0] = (uchar)((bp + m) * 255.0f);
                outRow[x][1] = (uchar)((gp + m) * 255.0f);
                outRow[x][2] = (uchar)((rp + m) * 255.0f);
            }
        }
    }
}

string getModeName()
{
    if (currentMode == "w")
        return "W: Bottom to Top";
    if (currentMode == "s")
        return "S: Top to Bottom";
    if (currentMode == "a")
        return "A: Right to Left";
    if (currentMode == "d")
        return "D: Left to Right";
    if (currentMode == "ws")
        return "W+S: Center-Out Vertical";
    if (currentMode == "ad")
        return "A+D: Center-Out Horizontal";
    if (currentMode == "motion")
        return "M: Motion Adaptive";
    if (currentMode == "chroma")
        return "C: Chromatic Time Shift";
    if (currentMode == "mchroma")
        return "X: Motion Chromatic";
    if (currentMode == "prismatic")
        return "P: Prismatic Echo";
    if (currentMode == "flowhue")
        return "H: Flow Direction Color";
    if (currentMode == "flowripple")
        return "J: Flow Color Ripple";
    if (currentMode == "datamosh")
        return "Y: Datamosh";
    if (currentMode == "ghostecho")
        return "E: Ghost Echo";
    if (currentMode == "timeghost")
        return "G: Temporal Ghost";
    if (currentMode == "rainbowghost")
        return "K: Rainbow Ghost";
    if (currentMode == "turbulence")
        return "N: Turbulence";
    return "Unknown";
}

void drawValueOverlay(Mat &output, const string &text, double timeSinceChange)
{
    const double DISPLAY_DURATION = 3.0;
    const double FADE_DURATION = 0.5;
    if (timeSinceChange > DISPLAY_DURATION)
        return;

    string speedText = text;
    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.5;
    int thickness = 3;
    int baseline = 0;
    cv::Size textSize = getTextSize(speedText, fontFace, fontScale, thickness, &baseline);

    int padding = 40;
    cv::Point textPos(output.cols - textSize.width - padding, output.rows - padding);
    putText(output, speedText, textPos, fontFace, fontScale, Scalar(255, 255, 255), thickness);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main()
{
    cout << "========================================" << endl;
    cout << "Time Mirror Effect - C++ Version" << endl;
    cout << "High Performance Implementation" << endl;
#ifdef _OPENMP
    cout << "OpenMP enabled (" << omp_get_max_threads() << " threads)" << endl;
#else
    cout << "OpenMP not enabled (single-threaded displacement)" << endl;
#endif
    cout << "========================================" << endl;

    VideoCapture cap;
    for (int attempt = 0; attempt < 20; ++attempt)
    {
        cap.open(0);
        if (cap.isOpened())
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    if (!cap.isOpened())
    {
        cerr << "ERROR: Cannot open camera" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    cap.set(CAP_PROP_FPS, 60);
    cap.set(CAP_PROP_BUFFERSIZE, 1); // Always grab the freshest frame

    int actualWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    int actualHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "Camera initialized: " << actualWidth << "x" << actualHeight << endl;
    if (actualWidth != FRAME_WIDTH || actualHeight != FRAME_HEIGHT)
        cout << "Note: Requested " << FRAME_WIDTH << "x" << FRAME_HEIGHT
             << " but using " << actualWidth << "x" << actualHeight << endl;

    // Pre-allocate circular buffer
    frameBuffer.resize(BUFFER_SIZE);
    for (int i = 0; i < BUFFER_SIZE; i++)
        frameBuffer[i] = Mat::zeros(actualHeight, actualWidth, CV_8UC3);
    cout << "Buffer initialized: " << BUFFER_SIZE << " frames" << endl;

    // Pre-allocate output frame — reused every iteration, no per-frame heap alloc
    Mat output(actualHeight, actualWidth, CV_8UC3);

    // Pre-allocate motion mode working buffers
    motionMap = Mat::zeros(actualHeight, actualWidth, CV_32F);
    diffMat = Mat::zeros(actualHeight, actualWidth, CV_8UC3);
    grayDiff = Mat::zeros(actualHeight, actualWidth, CV_8U);

    // Pre-allocate optical flow working buffers (computed at reduced resolution)
    int flowW = max(1, (int)(actualWidth * FLOW_SCALE));
    int flowH = max(1, (int)(actualHeight * FLOW_SCALE));
    flowMap = Mat::zeros(actualHeight, actualWidth, CV_32FC2);
    smallFlowBuf = Mat::zeros(flowH, flowW, CV_32FC2);
    smallPrevBGR = Mat::zeros(flowH, flowW, CV_8UC3);
    smallCurrBGR = Mat::zeros(flowH, flowW, CV_8UC3);
    prevGraySmall = Mat::zeros(flowH, flowW, CV_8U);
    currGraySmall = Mat::zeros(flowH, flowW, CV_8U);
    motionSmall = Mat::zeros(flowH, flowW, CV_8U);
    cout << "Flow buffers: " << flowW << "x" << flowH << " (1/" << (int)(1 / FLOW_SCALE) << " scale)" << endl;

    // Pre-allocate datamosh working buffers
    datamoshAccum = Mat::zeros(actualHeight, actualWidth, CV_32FC3);
    datamoshDiffF = Mat::zeros(actualHeight, actualWidth, CV_32FC3);

    // Pre-allocate flow ripple working buffers
    rippleBuffer = Mat::zeros(actualHeight, actualWidth, CV_32FC3);
    rippleTmp = Mat::zeros(actualHeight, actualWidth, CV_32FC3);
    rippleMapX = Mat::zeros(actualHeight, actualWidth, CV_32F);
    rippleMapY = Mat::zeros(actualHeight, actualWidth, CV_32F);
    ripple8 = Mat::zeros(actualHeight, actualWidth, CV_8UC3);

    // Pre-allocate turbulence mode buffers
    turbulenceMap = Mat::zeros(actualHeight, actualWidth, CV_32F);
    turbNoiseX.assign(actualWidth, 0.0f);
    turbNoiseY.assign(actualWidth, 0.0f);

    // Pre-allocate background removal mask; MOG2 subtractor created on mode entry

    // Pre-allocate temporal ghost mask buffer — one CV_8U mask per frame slot
    maskBuffer.resize(BUFFER_SIZE);
    for (int i = 0; i < BUFFER_SIZE; i++)
        maskBuffer[i] = Mat::zeros(actualHeight, actualWidth, CV_8U);

    namedWindow("Time Mirror Effect", WINDOW_NORMAL);
    resizeWindow("Time Mirror Effect", 1280, 720);

    cout << "\nKEYBOARD CONTROLS:" << endl;
    cout << "  W/S/A/D      - Direction controls" << endl;
    cout << "  W+S or A+D   - Combo modes (press within 0.5s)" << endl;
    cout << "  M            - Motion adaptive mode" << endl;
    cout << "  C            - Chromatic time shift mode" << endl;
    cout << "  X            - Motion chromatic (still=normal, moving=RGB time split)" << endl;
    cout << "  P            - Prismatic echo (6 hue-tinted temporal echoes)" << endl;
    cout << "  H            - Flow direction color (direction→hue, speed→saturation)" << endl;
    cout << "  J            - Flow color ripple (directional color that lingers and drifts)" << endl;
    cout << "  N            - Turbulence (motion history → displacement + chroma + saturation)" << endl;
    cout << "  Y            - Datamosh (motion trails via IIR diff accumulation)" << endl;
    cout << "  E            - Ghost Echo (7 motion-masked temporal echoes on black)" << endl;
    cout << "  B            - Background Removal (isolate moving foreground on black)" << endl;
    cout << "  G            - Temporal Ghost (7 person silhouettes through time on black)" << endl;
    cout << "  K            - Rainbow Ghost (like G but echoes tinted with cycling hues)" << endl;
    cout << "  Up/Down      - Speed / Chroma / Flow sens / Spread / Echo / Band ht" << endl;
    cout << "  R            - Reset the above to defaults" << endl;
    cout << "  F            - Toggle fullscreen" << endl;
    cout << "  Q/ESC        - Quit" << endl;
    cout << "\nStarting...\n"
         << endl;

    // Start capture thread — decouples camera I/O from render loop
    thread captureThread(captureLoop, ref(cap));

    // Start segmentation thread — uses Vision framework (built into macOS 12+).
    // No subprocess or external dependencies required.
    thread segThread(segmentLoop, actualWidth, actualHeight);

    // Wait for first frame before rendering
    while (writeIndex.load(memory_order_acquire) == 0 && running)
        this_thread::sleep_for(milliseconds(10));

    double fps = 0;
    auto fpsStartTime = steady_clock::now();
    int fpsFrameCount = 0;
    bool isFullscreen = false;

    auto lastOverlayTime = steady_clock::now();
    bool overlayActive = false;
    string overlayText;

    while (running)
    {
        // acquire: see all frameBuffer writes that happened before this index
        int bufIdx = writeIndex.load(memory_order_acquire);

        // Build motion map — used by motion and mchroma modes
        if (currentMode == "motion" || currentMode == "mchroma" || currentMode == "ghostecho")
        {
            int recent = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
            int older = (bufIdx - 1 - MOTION_LOOKBACK + BUFFER_SIZE * 2) % BUFFER_SIZE;
            cv::absdiff(frameBuffer[recent], frameBuffer[older], diffMat);
            cv::cvtColor(diffMat, grayDiff, COLOR_BGR2GRAY);
            // Blur at FLOW_SCALE resolution (~16× fewer pixels) then upsample
            cv::resize(grayDiff, motionSmall, motionSmall.size(), 0, 0, INTER_LINEAR);
            cv::GaussianBlur(motionSmall, motionSmall, cv::Size(MOTION_BLUR_SIZE, MOTION_BLUR_SIZE), 0);
            cv::resize(motionSmall, grayDiff, grayDiff.size(), 0, 0, INTER_LINEAR);
            grayDiff.convertTo(motionMap, CV_32F, 1.0 / 255.0);
        }

        // Update datamosh accumulator — used by datamosh mode.
        // Diffs MOTION_LOOKBACK frames apart (not adjacent) so pixel displacement
        // is large enough to be visible at 60fps. Adjacent frames produce near-zero
        // diffs that collapse the accumulator to black before trails can build up.
        // Reuses diffMat (CV_8UC3) scratch buffer already declared for motion mode.
        if (currentMode == "datamosh")
        {
            int curr = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
            int prev = (bufIdx - 1 - MOTION_LOOKBACK + BUFFER_SIZE * 2) % BUFFER_SIZE;
            cv::absdiff(frameBuffer[curr], frameBuffer[prev], diffMat);
            diffMat.convertTo(datamoshDiffF, CV_32FC3);
            cv::addWeighted(datamoshAccum, datamoshDecay,
                            datamoshDiffF, DATAMOSH_BOOST, 0.0, datamoshAccum);
        }

        // Update turbulence accumulator — used by turbulence mode.
        // Diffs current vs MOTION_LOOKBACK frames ago, blurs spatially, then IIR-decays
        // into turbulenceMap. Still areas fade to 0 over ~2s; motion spikes quickly to 1.
        if (currentMode == "turbulence")
        {
            int recent = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
            int older = (bufIdx - 1 - MOTION_LOOKBACK + BUFFER_SIZE * 2) % BUFFER_SIZE;
            cv::absdiff(frameBuffer[recent], frameBuffer[older], diffMat);
            cv::cvtColor(diffMat, grayDiff, COLOR_BGR2GRAY);
            cv::resize(grayDiff, motionSmall, motionSmall.size(), 0, 0, INTER_LINEAR);
            cv::GaussianBlur(motionSmall, motionSmall, cv::Size(MOTION_BLUR_SIZE, MOTION_BLUR_SIZE), 0);
            cv::resize(motionSmall, grayDiff, grayDiff.size(), 0, 0, INTER_LINEAR);
            cv::threshold(grayDiff, grayDiff, 20, 255, THRESH_TOZERO); // suppress camera noise
            grayDiff.convertTo(motionMap, CV_32F, 0.5 / 255.0);        // reuse motionMap as scratch
            turbulenceMap *= turbDecay;
            cv::add(turbulenceMap, motionMap, turbulenceMap);
            cv::min(turbulenceMap, 1.0f, turbulenceMap);
            turbFrame++;
        }

        // Advance rainbow hue — used by rainbowghost mode.
        if (currentMode == "rainbowghost")
            rainbowHue = fmod(rainbowHue + rainbowSpeed / 60.0f, 360.0f);

        // Compute optical flow — used by flowhue and flowripple modes.
        // Farneback runs at FLOW_SCALE resolution then is resized up for speed.
        if (currentMode == "flowhue" || currentMode == "flowripple")
        {
            int recent = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
            int older = (bufIdx - 2 + BUFFER_SIZE * 2) % BUFFER_SIZE;
            cv::resize(frameBuffer[older], smallPrevBGR, smallPrevBGR.size());
            cv::resize(frameBuffer[recent], smallCurrBGR, smallCurrBGR.size());
            cv::cvtColor(smallPrevBGR, prevGraySmall, COLOR_BGR2GRAY);
            cv::cvtColor(smallCurrBGR, currGraySmall, COLOR_BGR2GRAY);
            cv::calcOpticalFlowFarneback(prevGraySmall, currGraySmall, smallFlowBuf,
                                         0.5, 3, 15, 3, 5, 1.2, 0);
            cv::resize(smallFlowBuf, flowMap, flowMap.size());
            flowMap *= (1.0f / FLOW_SCALE); // convert small-scale px → full-resolution px
        }

        // Update flow ripple buffer — advect, decay, inject — used by flowripple mode.
        // 1. Build per-pixel backward-warp maps from flowMap.
        // 2. remap advects existing color content forward (in the flow direction).
        // 3. Decay: multiply by rippleDecay (~1 second lifetime at 60fps).
        // 4. Inject: where flow is strong, add a fresh saturated directional color additively.
        if (currentMode == "flowripple")
        {
            const float PI2 = 2.0f * 3.14159265f;
            const float invSens = 1.0f / flowSensitivity;
            const float invPI2 = 360.0f / PI2;

// Step 1: build backward-warp maps (pixel at (x,y) came from (x-vx, y-vy))
#pragma omp parallel for schedule(static)
            for (int y = 0; y < actualHeight; y++)
            {
                const Vec2f *flowRow = flowMap.ptr<Vec2f>(y);
                float *mx = rippleMapX.ptr<float>(y);
                float *my = rippleMapY.ptr<float>(y);
                for (int x = 0; x < actualWidth; x++)
                {
                    mx[x] = (float)x - flowRow[x][0];
                    my[x] = (float)y - flowRow[x][1];
                }
            }

            // Step 2: advect — shifts existing colors in the flow direction
            remap(rippleBuffer, rippleTmp, rippleMapX, rippleMapY,
                  INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

            // Step 3: decay
            rippleTmp *= rippleDecay;

// Step 4: inject fresh directional color where motion exceeds threshold
#pragma omp parallel for schedule(static)
            for (int y = 0; y < actualHeight; y++)
            {
                const Vec2f *flowRow = flowMap.ptr<Vec2f>(y);
                Vec3f *ripRow = rippleTmp.ptr<Vec3f>(y);
                for (int x = 0; x < actualWidth; x++)
                {
                    float vx = flowRow[x][0];
                    float vy = flowRow[x][1];
                    float mag = sqrtf(vx * vx + vy * vy);
                    if (mag < 0.5f)
                        continue;

                    float alpha = min(mag * invSens, 1.0f);
                    float hue = (atan2f(vy, vx) + 3.14159265f) * invPI2;

                    // HSV→BGR inline: S=1, V=1
                    float h6 = hue / 60.0f;
                    float xc = 1.0f - fabsf(fmodf(h6, 2.0f) - 1.0f);
                    float rp, gp, bp;
                    switch ((int)h6 % 6)
                    {
                    case 0:
                        rp = 1.f;
                        gp = xc;
                        bp = 0;
                        break;
                    case 1:
                        rp = xc;
                        gp = 1.f;
                        bp = 0;
                        break;
                    case 2:
                        rp = 0;
                        gp = 1.f;
                        bp = xc;
                        break;
                    case 3:
                        rp = 0;
                        gp = xc;
                        bp = 1.f;
                        break;
                    case 4:
                        rp = xc;
                        gp = 0;
                        bp = 1.f;
                        break;
                    default:
                        rp = 1.f;
                        gp = 0;
                        bp = xc;
                        break;
                    }

                    float brightness = alpha * 255.0f;
                    ripRow[x][0] = min(255.0f, ripRow[x][0] + bp * brightness);
                    ripRow[x][1] = min(255.0f, ripRow[x][1] + gp * brightness);
                    ripRow[x][2] = min(255.0f, ripRow[x][2] + rp * brightness);
                }
            }

            rippleTmp.copyTo(rippleBuffer);
        }

        applyTimeDisplacement(output, actualWidth, actualHeight, bufIdx);

        // FPS
        if (++fpsFrameCount >= 10)
        {
            auto now = steady_clock::now();
            fps = fpsFrameCount / duration<double>(now - fpsStartTime).count();
            fpsStartTime = now;
            fpsFrameCount = 0;
            cout << "FPS: " << (int)fps << " | Mode: " << getModeName() << endl;
        }

        if (overlayActive)
        {
            double dt = duration<double>(steady_clock::now() - lastOverlayTime).count();
            drawValueOverlay(output, overlayText, dt);
            if (dt > 3.0)
                overlayActive = false;
        }

        imshow("Time Mirror Effect", output);

        int key = waitKey(1);
        if (key == 'q' || key == 27)
        {
            running = false;
            break;
        }

        if (key == 'w')
        {
            currentMode = checkForCombo('w');
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 's')
        {
            currentMode = checkForCombo('s');
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'a')
        {
            currentMode = checkForCombo('a');
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'd')
        {
            currentMode = checkForCombo('d');
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'm')
        {
            currentMode = "motion";
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'c')
        {
            currentMode = "chroma";
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'x')
        {
            currentMode = "mchroma";
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'p')
        {
            currentMode = "prismatic";
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'h')
        {
            currentMode = "flowhue";
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'j')
        {
            currentMode = "flowripple";
            rippleBuffer.setTo(0); // clear lingering state on entry
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'n')
        {
            currentMode = "turbulence";
            turbulenceMap.setTo(0);
            turbFrame = 0;
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'y')
        {
            currentMode = "datamosh";
            datamoshAccum.setTo(0); // fresh slate each entry
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'e')
        {
            currentMode = "ghostecho";
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'g')
        {
            currentMode = "timeghost";
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'k')
        {
            currentMode = "rainbowghost";
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'f')
        {
            isFullscreen = !isFullscreen;
            setWindowProperty("Time Mirror Effect", WND_PROP_FULLSCREEN,
                              isFullscreen ? WINDOW_FULLSCREEN : WINDOW_NORMAL);
            cout << "Fullscreen: " << (isFullscreen ? "ON" : "OFF") << endl;
        }
        else if (key == 'r')
        {
            if (currentMode == "motion")
            {
                motionMaxOffset = 40;
                overlayText = "Depth: " + to_string(motionMaxOffset);
            }
            else if (currentMode == "chroma")
            {
                chromaOffset = 23;
                overlayText = "Chroma: " + to_string(chromaOffset);
            }
            else if (currentMode == "flowhue")
            {
                flowSensitivity = 10.0f;
                overlayText = "Flow: " + to_string((int)flowSensitivity);
            }
            else if (currentMode == "mchroma")
            {
                chromaSpread = 40;
                overlayText = "Spread: " + to_string(chromaSpread);
            }
            else if (currentMode == "prismatic")
            {
                echoSpacing = 23;
                overlayText = "Echo: " + to_string(echoSpacing);
            }
            else if (currentMode == "datamosh")
            {
                datamoshDecay = 0.92f;
                overlayText = "Trail: " + to_string((int)(datamoshDecay * 100));
            }
            else if (currentMode == "flowripple")
            {
                rippleDecay = 0.93f;
                overlayText = "Persist: " + to_string((int)(rippleDecay * 100));
            }
            else if (currentMode == "turbulence")
            {
                turbShift = 20.0f;
                overlayText = "Shift: " + to_string((int)turbShift);
            }
            else if (currentMode == "ghostecho")
            {
                ghostSpacing = 8;
                overlayText = "Spacing: " + to_string(ghostSpacing);
            }
            else if (currentMode == "timeghost")
            {
                tghostSpacing = 20;
                overlayText = "Spacing: " + to_string(tghostSpacing);
            }
            else if (currentMode == "rainbowghost")
            {
                tghostSpacing = 20;
                overlayText = "Spacing: " + to_string(tghostSpacing);
            }
            else
            {
                updateSpeed.store(1);
                overlayText = "Speed: 1";
            }
            cout << overlayText << endl;
            lastOverlayTime = steady_clock::now();
            overlayActive = true;
        }
        else if (key == 0)
        { // Up arrow (Mac)
            if (currentMode == "motion")
            {
                motionMaxOffset = min(BUFFER_SIZE - 1, motionMaxOffset + 5);
                overlayText = "Depth: " + to_string(motionMaxOffset);
            }
            else if (currentMode == "chroma")
            {
                chromaOffset = min(BUFFER_SIZE / 2 - 1, chromaOffset + 1);
                overlayText = "Chroma: " + to_string(chromaOffset);
            }
            else if (currentMode == "flowhue")
            {
                flowSensitivity = min(50.0f, flowSensitivity + 2.0f);
                overlayText = "Flow: " + to_string((int)flowSensitivity);
            }
            else if (currentMode == "mchroma")
            {
                chromaSpread = min(BUFFER_SIZE / 2 - 1, chromaSpread + 2);
                overlayText = "Spread: " + to_string(chromaSpread);
            }
            else if (currentMode == "prismatic")
            {
                echoSpacing = min(BUFFER_SIZE / 3, echoSpacing + 2);
                overlayText = "Echo: " + to_string(echoSpacing);
            }
            else if (currentMode == "datamosh")
            {
                datamoshDecay = min(0.98f, datamoshDecay + 0.02f);
                overlayText = "Trail: " + to_string((int)(datamoshDecay * 100));
            }
            else if (currentMode == "flowripple")
            {
                rippleDecay = min(0.99f, rippleDecay + 0.01f);
                overlayText = "Persist: " + to_string((int)(rippleDecay * 100));
            }
            else if (currentMode == "turbulence")
            {
                turbShift = min(60.0f, turbShift + 2.0f);
                overlayText = "Shift: " + to_string((int)turbShift);
            }
            else if (currentMode == "ghostecho")
            {
                ghostSpacing = min(BUFFER_SIZE / 7, ghostSpacing + 1);
                overlayText = "Spacing: " + to_string(ghostSpacing);
            }
            else if (currentMode == "timeghost")
            {
                tghostSpacing = min(BUFFER_SIZE / TGHOST_ECHOES, tghostSpacing + 1);
                overlayText = "Spacing: " + to_string(tghostSpacing);
            }
            else if (currentMode == "rainbowghost")
            {
                tghostSpacing = min(BUFFER_SIZE / TGHOST_ECHOES, tghostSpacing + 1);
                overlayText = "Spacing: " + to_string(tghostSpacing);
            }
            else
            {
                updateSpeed.store(min(BUFFER_SIZE, updateSpeed.load() + 1));
                overlayText = "Speed: " + to_string(updateSpeed.load());
            }
            cout << overlayText << endl;
            lastOverlayTime = steady_clock::now();
            overlayActive = true;
        }
        else if (key == 1)
        { // Down arrow (Mac)
            if (currentMode == "motion")
            {
                motionMaxOffset = max(5, motionMaxOffset - 5);
                overlayText = "Depth: " + to_string(motionMaxOffset);
            }
            else if (currentMode == "chroma")
            {
                chromaOffset = max(1, chromaOffset - 1);
                overlayText = "Chroma: " + to_string(chromaOffset);
            }
            else if (currentMode == "flowhue")
            {
                flowSensitivity = max(2.0f, flowSensitivity - 2.0f);
                overlayText = "Flow: " + to_string((int)flowSensitivity);
            }
            else if (currentMode == "mchroma")
            {
                chromaSpread = max(1, chromaSpread - 2);
                overlayText = "Spread: " + to_string(chromaSpread);
            }
            else if (currentMode == "prismatic")
            {
                echoSpacing = max(1, echoSpacing - 2);
                overlayText = "Echo: " + to_string(echoSpacing);
            }
            else if (currentMode == "datamosh")
            {
                datamoshDecay = max(0.70f, datamoshDecay - 0.02f);
                overlayText = "Trail: " + to_string((int)(datamoshDecay * 100));
            }
            else if (currentMode == "flowripple")
            {
                rippleDecay = max(0.70f, rippleDecay - 0.01f);
                overlayText = "Persist: " + to_string((int)(rippleDecay * 100));
            }
            else if (currentMode == "turbulence")
            {
                turbShift = max(2.0f, turbShift - 2.0f);
                overlayText = "Shift: " + to_string((int)turbShift);
            }
            else if (currentMode == "ghostecho")
            {
                ghostSpacing = max(1, ghostSpacing - 1);
                overlayText = "Spacing: " + to_string(ghostSpacing);
            }
            else if (currentMode == "timeghost")
            {
                tghostSpacing = max(1, tghostSpacing - 1);
                overlayText = "Spacing: " + to_string(tghostSpacing);
            }
            else if (currentMode == "rainbowghost")
            {
                tghostSpacing = max(1, tghostSpacing - 1);
                overlayText = "Spacing: " + to_string(tghostSpacing);
            }
            else
            {
                updateSpeed.store(max(1, updateSpeed.load() - 1));
                overlayText = "Speed: " + to_string(updateSpeed.load());
            }
            cout << overlayText << endl;
            lastOverlayTime = steady_clock::now();
            overlayActive = true;
        }
    }

    running = false;
    captureThread.join();
    segThread.join();

    cap.release();
    destroyAllWindows();
    cout << "\nProgram ended" << endl;
    return 0;
}
