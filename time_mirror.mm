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

// ── Single source of truth for all adjustable parameters ──────────────────────
// Each Param holds the live value plus its default, range, step, and display format.
// R key calls .reset(); Up/Down call .up()/.down(); render loop reads .value or .i().
enum class ParamFmt { INT, PCT, F1, F1X };
struct ModeParam {
    float value;
    const float def, vmin, vmax, step;
    const char* const label;
    const ParamFmt fmt;
    constexpr ModeParam(float d, float lo, float hi, float s, const char* l,
                    ParamFmt f = ParamFmt::INT)
        : value(d), def(d), vmin(lo), vmax(hi), step(s), label(l), fmt(f) {}
    void reset() { value = def; }
    void up()    { value = std::min(vmax, value + step); }
    void down()  { value = std::max(vmin, value - step); }
    int  i()     const { return (int)value; }
    std::string display() const {
        char buf[64];
        switch (fmt) {
            case ParamFmt::INT: snprintf(buf,64,"%s: %d",label,(int)value); break;
            case ParamFmt::PCT: snprintf(buf,64,"%s: %d",label,(int)(value*100+0.5f)); break;
            case ParamFmt::F1:  snprintf(buf,64,"%s: %.1f",label,value); break;
            case ParamFmt::F1X: snprintf(buf,64,"%s: %.1fx",label,value); break;
        }
        return buf;
    }
};

// ── Parameter instances — edit only here to change defaults, ranges, or steps ─
ModeParam P_chromaOffset {23,    1,    float(BUFFER_SIZE/2-1), 1,    "Chroma"};
ModeParam P_motionDepth  {40,    5,    float(BUFFER_SIZE-1),   5,    "Depth"};
ModeParam P_chromaSpread {40,    1,    float(BUFFER_SIZE/2-1), 2,    "Spread"};
ModeParam P_ghostSpace   {8,     1,    float(BUFFER_SIZE/7),   1,    "Spacing"};
ModeParam P_flowWarp     {10,    1,    50,    2,    "Warp"};
ModeParam P_waveRefract  {15,    1,    50,    2,    "Refract"};
ModeParam P_chromaWave   {2,    1,    15,    1,    "Refract"};
ModeParam P_echoSpacing  {23,    1,    float(BUFFER_SIZE/3),   2,    "Echo"};
ModeParam P_flowSens     {10,    2,    50,    2,    "Flow"};
ModeParam P_datamosh     {0.92f, 0.70f, 0.992f, 0.02f, "Trail",   ParamFmt::PCT};
ModeParam P_rippleDecay  {0.93f, 0.70f, 0.99f,  0.01f, "Persist", ParamFmt::PCT};
ModeParam P_turbShift    {20,    2,    60,    2,    "Shift"};
ModeParam P_tghostSpace  {20,    1,    float(BUFFER_SIZE/7),   1,    "Spacing"};
ModeParam P_tunnelScale  {3.0f,  1.2f, 8.0f,  0.5f, "Zoom",    ParamFmt::F1X};
ModeParam P_glowBoost    {0.5f,  0.25f, 6.0f, 0.25f, "Glow",   ParamFmt::F1};

// Motion mode settings
const int MOTION_LOOKBACK = 10;  // Frames back for motion comparison (higher = reacts to slower motion)
const int MOTION_BLUR_SIZE = 31; // Spatial blur radius — larger spreads halos further (must be odd)

// Wave Warp mode (N) — 2D wave simulation seeded by motion.
// Standard wave equation: new = (neighbors sum)*0.5 - prev, with IIR damping.
// Wave gradient displaces camera sample coordinates for a refracting water surface.
Mat waveA, waveB;              // CV_32F double-buffer: waveA=current, waveB=previous
const float WAVE_DAMP = 0.97f; // per-frame damping (lower = shorter ripples)
const float WAVE_SEED = 3.0f;  // motion map intensity seeded into wave per frame

// Chroma Wave mode (M) — three independent wave simulations, one per RGB channel.
// Each wave is seeded from its own colour channel's absdiff so R/G/B motion drives
// independent wave patterns; each channel's displacement is rotated 120° apart so
// even similar wave shapes produce vivid colour separation in different spatial directions.
Mat waveAr, waveBr, waveAg, waveBg, waveAb, waveBb; // per-channel double buffers
const float CWAVE_DAMP = 0.985f; // less decay than N → bigger sustained waves
const float CWAVE_SEED = 8.0f;   // stronger injection than N → larger amplitudes


// Shared between capture thread and main thread.
// writeIndex uses release/acquire semantics so the main thread always
// sees a fully-written frame before the index advances past it.
vector<Mat> frameBuffer;
atomic<int> writeIndex{0};
atomic<int> updateSpeed{1};
atomic<bool> running{true};

// Motion mode working buffers — pre-allocated in main
Mat motionMap;      // CV_32F, shallow-copy alias into motionMapBuf[prepBuf] each frame
Mat diffMat;        // CV_8UC3 scratch (main thread only)
Mat grayDiff;       // CV_8U  scratch (main thread only)
Mat motionSmall;    // CV_8U  scratch for motion blur at FLOW_SCALE resolution (main thread only)
Mat turbScratch;    // CV_32F scratch for turbulence accumulation (replaces motionMap reuse)

// Double-buffered preprocessing maps — written by preprocessLoop thread, read by main thread.
// prepBuf is the buffer index safe to read; the thread writes to the opposite index.
// Float-level data races on the handover frame are benign for a visual-art application.
Mat motionMapBuf[2]; // CV_32F
Mat flowMapBuf[2];   // CV_32FC2
atomic<int> prepBuf{0};

// Optical flow mode working buffers — pre-allocated in main
// Flow is computed at FLOW_SCALE of full resolution for performance, then resized up.
const float FLOW_SCALE = 0.25f; // compute flow at 1/4 linear resolution (~16x fewer pixels)
Mat flowMap;                    // CV_32FC2, shallow-copy alias into flowMapBuf[prepBuf] each frame

// Datamosh mode buffers
Mat datamoshAccum;  // CV_32FC3, signed accumulator
Mat datamoshDiffF;  // CV_32FC3 scratch for the per-frame diff
// Boost scales inversely with (1-decay) to hold steady-state brightness constant.
const float DATAMOSH_BOOST_K = 6.5625f;

// Flow Ripple mode buffers
Mat rippleBuffer; // CV_32FC3, full res, persists between frames
Mat rippleTmp;    // CV_32FC3, scratch for remap output
Mat rippleMapX;   // CV_32F,   backward-warp x coords built from flowMap
Mat rippleMapY;   // CV_32F,   backward-warp y coords built from flowMap
Mat ripple8;      // CV_8UC3,  converted for additive compositing

// Turbulence mode buffers
Mat turbulenceMap;        // CV_32F, 0–1, accumulated per-pixel motion history
vector<float> turbNoiseX; // pre-allocated sinf lookup table, size = actualWidth
vector<float> turbNoiseY; // pre-allocated cosf lookup table, size = actualWidth
int turbFrame = 0;
const float turbDecay = 0.992f; // IIR decay (~2s half-life at 60fps), not user-adjustable

// Temporal Ghost / segmentation state
vector<Mat> maskBuffer;   // CV_8U, same slots as frameBuffer
atomic<int> segReady{-1}; // index of last written mask; -1 = not started
const int TGHOST_ECHOES = 7;
float rainbowHue  = 0.0f;  // G/T-alt mode: current base hue (0–360, cycles each frame)
float rainbowSpeed = 30.0f; // degrees per second the hue advances
const int RAINBOW_HUE_STEP = 45; // degrees between consecutive echoes
float ringOffset = 0.0f;   // expanding ring backdrop phase (px, advances each frame)
const float RING_SPACING = 200.0f;
const float RING_SPEED   = 75.0f;

// ── Mode enum — single authoritative list of all mode identifiers ─────────────
// Using enum class prevents silent typos: an invalid name is a compile error.
enum class Mode {
    S, W, A, D, WS, AD,
    MOTION, CHROMA, MCHROMA,
    PRISMATIC, PRISMATICGHOST,
    FLOWRIPPLE, FLOWHUE,
    DATAMOSH, TURBULENCE,
    GHOSTECHO, CHROMAGHOSTECHO,
    TIMEGHOST, TUNNELTIMEGHOST,
    RAINBOWGHOST, TUNNELGHOST,
    FLOWWARP, WAVEWARP, CHROMAWAVE
};

// Main-thread-only state
Mode currentMode = Mode::S;
// Toggle-pair memory: remembers which variant was last active so returning to a key
// restores the exact mode the user left, not always the primary variant.
Mode lastT = Mode::PRISMATIC;
Mode lastY = Mode::FLOWRIPPLE;
Mode lastG = Mode::RAINBOWGHOST;
Mode lastH = Mode::TIMEGHOST;
Mode lastC = Mode::GHOSTECHO;
Mode lastK = Mode::WAVEWARP;
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
                if (bgraMat.empty() || bgraMat.cols == 0 || bgraMat.rows == 0)
                {
                    lastSeg = latest;
                    continue;
                }

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

                if (!err && request.results.count > 0 &&
                    [request.results[0] isKindOfClass:[VNPixelBufferObservation class]])
                {
                    VNPixelBufferObservation *obs =
                        (VNPixelBufferObservation *)request.results[0];
                    if (!obs || !obs.pixelBuffer) { lastSeg = latest; continue; }
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

// ── Preprocessing thread ──────────────────────────────────────────────────────
// Computes motion map (M/X/E) and optical flow (H/J) on a dedicated thread so
// these sequential OpenCV operations overlap with the previous frame's render+display.
// Writes into motionMapBuf[writeBuf] / flowMapBuf[writeBuf], then atomically advances
// prepBuf so the main thread picks up the result on its next iteration.
// All scratch buffers are private to this thread — no sharing with main thread.
void preprocessLoop(int actualWidth, int actualHeight)
{
    const int smallW = max(1, (int)(actualWidth  * FLOW_SCALE));
    const int smallH = max(1, (int)(actualHeight * FLOW_SCALE));

    // Private scratch — never accessed by main thread
    Mat pDiff(actualHeight, actualWidth, CV_8UC3);
    Mat pGray(actualHeight, actualWidth, CV_8U);
    Mat pSmall(smallH, smallW, CV_8U);
    Mat pPrevBGR(smallH, smallW, CV_8UC3);
    Mat pCurrBGR(smallH, smallW, CV_8UC3);
    Mat pPrevGray(smallH, smallW, CV_8U);
    Mat pCurrGray(smallH, smallW, CV_8U);
    Mat pFlowSmall(smallH, smallW, CV_32FC2);

    int lastPrepped = -1;
    int writeBuf = 1; // start writing to buf 1 (render reads buf 0 initially)

    while (running)
    {
        int latest = (writeIndex.load(memory_order_acquire) - 1 + BUFFER_SIZE) % BUFFER_SIZE;
        if (latest == lastPrepped) { this_thread::yield(); continue; }

        Mode mode = currentMode;

        if (mode == Mode::MOTION || mode == Mode::MCHROMA || mode == Mode::GHOSTECHO || mode == Mode::CHROMAGHOSTECHO || mode == Mode::WAVEWARP)
        {
            int recent = latest;
            int older  = (latest - MOTION_LOOKBACK + BUFFER_SIZE * 2) % BUFFER_SIZE;
            cv::absdiff(frameBuffer[recent], frameBuffer[older], pDiff);
            cv::cvtColor(pDiff, pGray, COLOR_BGR2GRAY);
            cv::resize(pGray, pSmall, pSmall.size(), 0, 0, INTER_LINEAR);
            cv::GaussianBlur(pSmall, pSmall, cv::Size(MOTION_BLUR_SIZE, MOTION_BLUR_SIZE), 0);
            cv::resize(pSmall, pGray, pGray.size(), 0, 0, INTER_LINEAR);
            pGray.convertTo(motionMapBuf[writeBuf], CV_32F, 1.0 / 255.0);
            prepBuf.store(writeBuf, memory_order_release);
            writeBuf = 1 - writeBuf;
        }
        else if (mode == Mode::FLOWHUE || mode == Mode::FLOWRIPPLE || mode == Mode::FLOWWARP)
        {
            int recent = latest;
            int older  = (latest - 1 + BUFFER_SIZE) % BUFFER_SIZE;
            cv::resize(frameBuffer[older],  pPrevBGR,  pPrevBGR.size());
            cv::resize(frameBuffer[recent], pCurrBGR,  pCurrBGR.size());
            cv::cvtColor(pPrevBGR, pPrevGray, COLOR_BGR2GRAY);
            cv::cvtColor(pCurrBGR, pCurrGray, COLOR_BGR2GRAY);
            cv::calcOpticalFlowFarneback(pPrevGray, pCurrGray, pFlowSmall,
                                         0.5, 3, 15, 3, 5, 1.2, 0);
            cv::resize(pFlowSmall, flowMapBuf[writeBuf], flowMapBuf[writeBuf].size());
            flowMapBuf[writeBuf] *= (1.0f / FLOW_SCALE);
            prepBuf.store(writeBuf, memory_order_release);
            writeBuf = 1 - writeBuf;
        }
        // Other modes need no map preprocessing — just note the frame as handled.

        lastPrepped = latest;
    }
}

// ── Key combo detection ───────────────────────────────────────────────────────
Mode checkForCombo(char keyPressed)
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
                return Mode::WS;
        }
        return (keyPressed == 'w') ? Mode::W : Mode::S;
    }
    if (keyPressed == 'a' || keyPressed == 'd')
    {
        char other = (keyPressed == 'a') ? 'd' : 'a';
        if (lastKeyTime.count(other))
        {
            double dt = duration<double>(currentTime - lastKeyTime[other]).count();
            if (dt < COMBO_WINDOW)
                return Mode::AD;
        }
        return (keyPressed == 'a') ? Mode::A : Mode::D;
    }
    return Mode::S; // unreachable
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
    if (currentMode == Mode::WS)
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
    else if (currentMode == Mode::AD)
    {
        int centerX = width / 2;
        vector<int> colFrame(width);
        for (int x = 0; x < width; x++)
        {
            int dist = abs(x - centerX);
            colFrame[x] = (bufIdx - 1 - (dist * BUFFER_SIZE / max(centerX, 1)) + BUFFER_SIZE * 2) % BUFFER_SIZE;
        }
        // Group consecutive columns with the same source frame into strips and
        // parallelise over strips. Each strip reads one frame sequentially across
        // all rows (~32 KB), fits in L1 and lets the hardware prefetcher work —
        // much lower cache-miss cost than jumping across 200 frames per row.
        struct Strip { int frame, x0, x1; }; // x1 is exclusive
        vector<Strip> strips;
        strips.reserve(BUFFER_SIZE * 2 + 2);
        for (int i = 0, j; i < width; i = j) {
            int f = colFrame[i];
            for (j = i + 1; j < width && colFrame[j] == f; ++j) {}
            strips.push_back({f, i, j});
        }
#pragma omp parallel for schedule(static)
        for (int si = 0; si < (int)strips.size(); ++si)
        {
            const Strip& s = strips[si];
            size_t off = (size_t)s.x0 * 3, nb = (size_t)(s.x1 - s.x0) * 3;
            for (int y = 0; y < height; ++y)
                memcpy(output.ptr(y) + off, frameBuffer[s.frame].ptr(y) + off, nb);
        }
    }
    else if (currentMode == Mode::W)
    {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            int frameOffset = (bufIdx - 1 - ((height - 1 - y) * BUFFER_SIZE / height) + BUFFER_SIZE * 2) % BUFFER_SIZE;
            frameBuffer[frameOffset].row(y).copyTo(output.row(y));
        }
    }
    else if (currentMode == Mode::S)
    {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            int frameOffset = (bufIdx - 1 - (y * BUFFER_SIZE / height) + BUFFER_SIZE * 2) % BUFFER_SIZE;
            frameBuffer[frameOffset].row(y).copyTo(output.row(y));
        }
    }
    else if (currentMode == Mode::A)
    {
        vector<int> colFrame(width);
        for (int x = 0; x < width; x++)
            colFrame[x] = (bufIdx - 1 - ((width - 1 - x) * BUFFER_SIZE / width) + BUFFER_SIZE * 2) % BUFFER_SIZE;
        struct Strip { int frame, x0, x1; };
        vector<Strip> strips;
        strips.reserve(BUFFER_SIZE + 2);
        for (int i = 0, j; i < width; i = j) {
            int f = colFrame[i];
            for (j = i + 1; j < width && colFrame[j] == f; ++j) {}
            strips.push_back({f, i, j});
        }
#pragma omp parallel for schedule(static)
        for (int si = 0; si < (int)strips.size(); ++si)
        {
            const Strip& s = strips[si];
            size_t off = (size_t)s.x0 * 3, nb = (size_t)(s.x1 - s.x0) * 3;
            for (int y = 0; y < height; ++y)
                memcpy(output.ptr(y) + off, frameBuffer[s.frame].ptr(y) + off, nb);
        }
    }
    else if (currentMode == Mode::D)
    {
        vector<int> colFrame(width);
        for (int x = 0; x < width; x++)
            colFrame[x] = (bufIdx - 1 - (x * BUFFER_SIZE / width) + BUFFER_SIZE * 2) % BUFFER_SIZE;
        struct Strip { int frame, x0, x1; };
        vector<Strip> strips;
        strips.reserve(BUFFER_SIZE + 2);
        for (int i = 0, j; i < width; i = j) {
            int f = colFrame[i];
            for (j = i + 1; j < width && colFrame[j] == f; ++j) {}
            strips.push_back({f, i, j});
        }
#pragma omp parallel for schedule(static)
        for (int si = 0; si < (int)strips.size(); ++si)
        {
            const Strip& s = strips[si];
            size_t off = (size_t)s.x0 * 3, nb = (size_t)(s.x1 - s.x0) * 3;
            for (int y = 0; y < height; ++y)
                memcpy(output.ptr(y) + off, frameBuffer[s.frame].ptr(y) + off, nb);
        }
    }
    else if (currentMode == Mode::MOTION)
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
                int offset = (int)(motionRow[x] * P_motionDepth.value);
                int idx = (bufIdx - 1 - offset + BUFFER_SIZE * 2) % BUFFER_SIZE;
                outRow[x] = frameBuffer[idx].ptr<Vec3b>(y)[x];
            }
        }
    }
    else if (currentMode == Mode::CHROMA)
    {
        // B channel from most recent frame, G from CHROMA_OFFSET frames ago,
        // R from 2×CHROMA_OFFSET frames ago. Still objects look normal; moving
        // objects leave blue→green→red colour trails.
        // mixChannels does this in a single pass with no intermediate allocations.
        int idx0 = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
        int idx1 = (bufIdx - 1 - P_chromaOffset.i() + BUFFER_SIZE * 2) % BUFFER_SIZE;
        int idx2 = (bufIdx - 1 - P_chromaOffset.i() * 2 + BUFFER_SIZE * 2) % BUFFER_SIZE;

        const Mat sources[] = {frameBuffer[idx0], frameBuffer[idx1], frameBuffer[idx2]};
        const int fromTo[] = {0, 0, 4, 1, 8, 2}; // B←frame0, G←frame1, R←frame2
        cv::mixChannels(sources, 3, &output, 1, fromTo, 3);
    }
    else if (currentMode == Mode::MCHROMA)
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
                int spread = (int)(motionRow[x] * P_chromaSpread.i());
                int idxG = (bufIdx - 1 - spread + BUFFER_SIZE * 2) % BUFFER_SIZE;
                int idxR = (bufIdx - 1 - spread * 2 + BUFFER_SIZE * 4) % BUFFER_SIZE;
                outRow[x][0] = rowB[x][0];                            // B ← newest
                outRow[x][1] = frameBuffer[idxG].ptr<Vec3b>(y)[x][1]; // G ← spread ago
                outRow[x][2] = frameBuffer[idxR].ptr<Vec3b>(y)[x][2]; // R ← 2× spread ago
            }
        }
    }
    else if (currentMode == Mode::PRISMATIC)
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
            fi[e] = (bufIdx - 1 - e * P_echoSpacing.i() + BUFFER_SIZE * 4) % BUFFER_SIZE;
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
    else if (currentMode == Mode::DATAMOSH)
    {
        // Output already written by the fused preprocessing loop — nothing to do here.
    }
    else if (currentMode == Mode::GHOSTECHO)
    {
        // 7 temporal echoes stacked with triangular weights (newest = brightest).
        // motionMap masks the output: still areas → black, moving areas → ghost stack.
        // Frame indices are hoisted out of the parallel loop — one per echo.
        int fi[7];
        for (int e = 0; e < 7; e++)
            fi[e] = (bufIdx - 1 - e * P_ghostSpace.i() + BUFFER_SIZE * 10) % BUFFER_SIZE;
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
    else if (currentMode == Mode::CHROMAGHOSTECHO)
    {
        // Ghost Echo variant: each echo tinted a cycling hue (like Rainbow Ghost but motion-masked).
        // Per-echo luma × hue replaces the neutral weighted blend; rainbowHue cycles each frame.
        int fi[7];
        for (int e = 0; e < 7; e++)
            fi[e] = (bufIdx - 1 - e * P_ghostSpace.i() + BUFFER_SIZE * 10) % BUFFER_SIZE;

        static const float w[7] = {7/28.f, 6/28.f, 5/28.f, 4/28.f, 3/28.f, 2/28.f, 1/28.f};

        float eB[7], eG[7], eR[7];
        for (int e = 0; e < 7; e++)
        {
            float hue = fmod(rainbowHue - e * RAINBOW_HUE_STEP + 360.0f * 7, 360.0f);
            float h6 = hue / 60.0f;
            int hi = (int)h6 % 6;
            float f = h6 - (int)h6, q = 1.0f - f;
            switch (hi)
            {
            case 0: eR[e]=1; eG[e]=f; eB[e]=0; break;
            case 1: eR[e]=q; eG[e]=1; eB[e]=0; break;
            case 2: eR[e]=0; eG[e]=1; eB[e]=f; break;
            case 3: eR[e]=0; eG[e]=q; eB[e]=1; break;
            case 4: eR[e]=f; eG[e]=0; eB[e]=1; break;
            default: eR[e]=1; eG[e]=0; eB[e]=q; break;
            }
        }

#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            const float *motRow = motionMap.ptr<float>(y);
            Vec3b *outRow = output.ptr<Vec3b>(y);
            for (int x = 0; x < width; x++)
            {
                float m = motRow[x];
                if (m < 0.05f) { outRow[x] = Vec3b(0, 0, 0); continue; }
                float B = 0, G = 0, R = 0;
                for (int e = 0; e < 7; e++)
                {
                    const Vec3b &p = frameBuffer[fi[e]].ptr<Vec3b>(y)[x];
                    float lum = (0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2]) * (1.0f/255.0f);
                    B += eB[e] * lum * w[e];
                    G += eG[e] * lum * w[e];
                    R += eR[e] * lum * w[e];
                }
                float boost = min(m * 3.0f, 1.0f) * 2.0f * 255.0f;
                outRow[x] = Vec3b(
                    (uchar)min(255.f, B * boost),
                    (uchar)min(255.f, G * boost),
                    (uchar)min(255.f, R * boost));
            }
        }
    }
    else if (currentMode == Mode::PRISMATICGHOST)
    {
        // Person masks from Vision segmentation, each echo tinted a spectral hue (like T but
        // on masked silhouettes). Echoes blended additively so overlaps become brighter;
        // glow applied via sqrtf(alpha) falloff so mask edges bloom.
        int base = segReady.load(memory_order_acquire);
        if (base < 0) { output.setTo(Scalar(0, 0, 0)); return; }

        int fi[TGHOST_ECHOES];
        for (int e = 0; e < TGHOST_ECHOES; e++)
            fi[e] = (base - e * P_tghostSpace.i() + BUFFER_SIZE * 10) % BUFFER_SIZE;

        // Precompute per-echo hue→BGR (cycling like rainbow ghost)
        float eB[TGHOST_ECHOES], eG[TGHOST_ECHOES], eR[TGHOST_ECHOES];
        for (int e = 0; e < TGHOST_ECHOES; e++)
        {
            float hue = fmod(rainbowHue - e * RAINBOW_HUE_STEP + 360.0f * TGHOST_ECHOES, 360.0f);
            float h6 = hue / 60.0f;
            int hi = (int)h6 % 6;
            float f = h6 - (int)h6, q = 1.0f - f;
            switch (hi)
            {
            case 0: eR[e]=1; eG[e]=f; eB[e]=0; break;
            case 1: eR[e]=q; eG[e]=1; eB[e]=0; break;
            case 2: eR[e]=0; eG[e]=1; eB[e]=f; break;
            case 3: eR[e]=0; eG[e]=q; eB[e]=1; break;
            case 4: eR[e]=f; eG[e]=0; eB[e]=1; break;
            default: eR[e]=1; eG[e]=0; eB[e]=q; break;
            }
        }

        float gb      = P_glowBoost.value;
        float cx      = (width  - 1) * 0.5f;
        float cy      = (height - 1) * 0.5f;
        float maxDist = sqrtf(cx * cx + cy * cy); // corner distance for brightness ramp
        float ro      = ringOffset;
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            Vec3b *outRow = output.ptr<Vec3b>(y);
            const uchar *maskRows[TGHOST_ECHOES];
            const Vec3b *frameRows[TGHOST_ECHOES];
            for (int e = 0; e < TGHOST_ECHOES; e++)
            {
                maskRows[e]  = maskBuffer[fi[e]].ptr<uchar>(y);
                frameRows[e] = frameBuffer[fi[e]].ptr<Vec3b>(y);
            }
            float dy2 = (y - cy) * (y - cy);
            for (int x = 0; x < width; x++)
            {
                // Ring backdrop: light sections are black at centre, brighten toward edges
                float dist  = sqrtf(dy2 + (x - cx) * (x - cx));
                float phase = fmodf(dist - ro + RING_SPACING * 1000.0f, RING_SPACING);
                float rv    = (phase < RING_SPACING * 0.5f) ? (dist / maxDist) * 35.0f : 0.0f;
                float B = rv, G = rv, R = rv;

                // Additive echo accumulation
                for (int e = 0; e < TGHOST_ECHOES; e++)
                {
                    float alpha = maskRows[e][x] * (1.0f / 255.0f);
                    if (alpha < 0.05f) continue;
                    const Vec3b &p = frameRows[e][x];
                    float lum = (0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2]) * (1.0f / 255.0f);
                    // sqrtf(alpha) blooms soft mask edges; multiply by lum and boost
                    float glow = sqrtf(alpha) * lum * gb;
                    B += eB[e] * glow * 255.0f;
                    G += eG[e] * glow * 255.0f;
                    R += eR[e] * glow * 255.0f;
                }
                outRow[x] = Vec3b(
                    (uchar)(B > 255.0f ? 255 : (int)B),
                    (uchar)(G > 255.0f ? 255 : (int)G),
                    (uchar)(R > 255.0f ? 255 : (int)R));
            }
        }
    }
    else if (currentMode == Mode::TIMEGHOST)
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
            fi[e] = (base - e * P_tghostSpace.i() + BUFFER_SIZE * 10) % BUFFER_SIZE;

        // Precompute per-echo fade (depends only on echo index, not on pixel)
        float echoFade[TGHOST_ECHOES];
        for (int e = 0; e < TGHOST_ECHOES; e++)
            echoFade[e] = 0.30f + 0.70f * (1.0f - (float)e / (TGHOST_ECHOES - 1));

        {
            float cx2 = (width  - 1) * 0.5f;
            float cy2 = (height - 1) * 0.5f;
            float ro = ringOffset;
#pragma omp parallel for schedule(static)
            for (int y = 0; y < height; y++)
            {
                const uchar *maskRows[TGHOST_ECHOES];
                const Vec3b *frameRows[TGHOST_ECHOES];
                for (int e = 0; e < TGHOST_ECHOES; e++) {
                    maskRows[e]  = maskBuffer[fi[e]].ptr<uchar>(y);
                    frameRows[e] = frameBuffer[fi[e]].ptr<Vec3b>(y);
                }
                Vec3b *outRow = output.ptr<Vec3b>(y);
                float dy2 = (y - cy2) * (y - cy2);
                for (int x = 0; x < width; x++)
                {
                    float dist  = sqrtf(dy2 + (x - cx2) * (x - cx2));
                    float phase = fmodf(dist - ro + RING_SPACING * 1000.0f, RING_SPACING);
                    uchar rv    = (phase < RING_SPACING * 0.5f) ? 35 : 0;
                    outRow[x]   = Vec3b(rv, rv, rv);
                    for (int e = TGHOST_ECHOES - 1; e >= 0; e--)
                    {
                        int alpha = maskRows[e][x];
                        if (alpha < 13) continue;
                        float total = (alpha / 255.0f) * echoFade[e];
                        const Vec3b &p = frameRows[e][x];
                        outRow[x] = Vec3b((uchar)(p[0]*total), (uchar)(p[1]*total), (uchar)(p[2]*total));
                    }
                }
            }
        }
    }
    else if (currentMode == Mode::RAINBOWGHOST)
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
            fi[e] = (base - e * P_tghostSpace.i() + BUFFER_SIZE * 10) % BUFFER_SIZE;

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

        // Expanding ring backdrop: concentric gray/black rings that slowly grow outward.
        // Each pixel's ring color is determined by its distance from centre modulo RING_SPACING.
        // ringOffset advances each frame so rings appear to expand.
        float cx = (width  - 1) * 0.5f;
        float cy = (height - 1) * 0.5f;
        float ro = ringOffset; // local copy — safe to read on main thread
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            const uchar *maskRows[TGHOST_ECHOES];
            const Vec3b *frameRows[TGHOST_ECHOES];
            for (int e = 0; e < TGHOST_ECHOES; e++) {
                maskRows[e]  = maskBuffer[fi[e]].ptr<uchar>(y);
                frameRows[e] = frameBuffer[fi[e]].ptr<Vec3b>(y);
            }
            Vec3b *outRow = output.ptr<Vec3b>(y);
            float dy2 = (y - cy) * (y - cy);
            for (int x = 0; x < width; x++)
            {
                // Ring backdrop
                float dist  = sqrtf(dy2 + (x - cx) * (x - cx));
                float phase = fmodf(dist - ro + RING_SPACING * 1000.0f, RING_SPACING);
                uchar rv    = (phase < RING_SPACING * 0.5f) ? 35 : 0;
                outRow[x]   = Vec3b(rv, rv, rv);

                // Echo overlay (oldest → newest so newest paints on top)
                for (int e = TGHOST_ECHOES - 1; e >= 0; e--)
                {
                    int alpha = maskRows[e][x];
                    if (alpha < 13) continue;
                    const Vec3b &p = frameRows[e][x];
                    float brightness = (0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2])
                                       * (alpha / 255.0f);
                    outRow[x] = Vec3b(
                        (uchar)min(255.0f, eB[e] * brightness),
                        (uchar)min(255.0f, eG[e] * brightness),
                        (uchar)min(255.0f, eR[e] * brightness));
                }
            }
        }
    }
    else if (currentMode == Mode::TUNNELTIMEGHOST)
    {
        // Temporal Ghost variant: each older echo is scaled up toward centre (same tunnel
        // geometry as Tunnel Ghost) but uses natural colour + brightness fade like Temporal Ghost.
        int base = segReady.load(memory_order_acquire);
        if (base < 0) { output.setTo(Scalar(0, 0, 0)); return; }

        int fi[TGHOST_ECHOES];
        for (int e = 0; e < TGHOST_ECHOES; e++)
            fi[e] = (base - e * P_tghostSpace.i() + BUFFER_SIZE * 10) % BUFFER_SIZE;

        float echoFade[TGHOST_ECHOES];
        for (int e = 0; e < TGHOST_ECHOES; e++)
            echoFade[e] = 0.30f + 0.70f * (1.0f - (float)e / (TGHOST_ECHOES - 1));

        float inv_s[TGHOST_ECHOES];
        for (int e = 0; e < TGHOST_ECHOES; e++)
        {
            float s = 1.0f + (P_tunnelScale.value - 1.0f) * (float)e / (TGHOST_ECHOES - 1);
            inv_s[e] = 1.0f / s;
        }

        float cx = (width  - 1) * 0.5f;
        float cy = (height - 1) * 0.5f;
        float ro = ringOffset;
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            Vec3b *outRow = output.ptr<Vec3b>(y);

            const uchar *maskRowE[TGHOST_ECHOES]  = {};
            const Vec3b *frameRowE[TGHOST_ECHOES] = {};
            float ixBias[TGHOST_ECHOES];
            float dy2 = (y - cy) * (y - cy);
            for (int e = 0; e < TGHOST_ECHOES; e++)
            {
                float sy = cy + (y - cy) * inv_s[e];
                int iy = (int)(sy + 0.5f);
                if ((unsigned)iy < (unsigned)height)
                {
                    maskRowE[e]  = maskBuffer[fi[e]].ptr<uchar>(iy);
                    frameRowE[e] = frameBuffer[fi[e]].ptr<Vec3b>(iy);
                }
                ixBias[e] = cx * (1.0f - inv_s[e]);
            }

            for (int x = 0; x < width; x++)
            {
                float dist  = sqrtf(dy2 + (x - cx) * (x - cx));
                float phase = fmodf(dist - ro + RING_SPACING * 1000.0f, RING_SPACING);
                uchar rv    = (phase < RING_SPACING * 0.5f) ? 35 : 0;
                outRow[x]   = Vec3b(rv, rv, rv);

                for (int e = TGHOST_ECHOES - 1; e >= 0; e--)
                {
                    if (!maskRowE[e]) continue;
                    int ix = (int)(ixBias[e] + x * inv_s[e] + 0.5f);
                    if ((unsigned)ix >= (unsigned)width) continue;
                    int alpha = maskRowE[e][ix];
                    if (alpha < 13) continue;
                    float total = (alpha / 255.0f) * echoFade[e];
                    const Vec3b &p = frameRowE[e][ix];
                    outRow[x] = Vec3b(
                        (uchar)(p[0] * total),
                        (uchar)(p[1] * total),
                        (uchar)(p[2] * total));
                }
            }
        }
    }
    else if (currentMode == Mode::TUNNELGHOST)
    {
        // Rainbow Ghost variant where each older echo is scaled down toward the image centre,
        // creating a receding tunnel of coloured person silhouettes.
        // Newest echo (e=0) = full size; oldest (e=TGHOST_ECHOES-1) = P_tunnelScale.value × full size.
        // Rendered back-to-front so the nearest (largest) echo paints last.
        int base = segReady.load(memory_order_acquire);
        if (base < 0) { output.setTo(Scalar(0, 0, 0)); return; }

        int fi[TGHOST_ECHOES];
        for (int e = 0; e < TGHOST_ECHOES; e++)
            fi[e] = (base - e * P_tghostSpace.i() + BUFFER_SIZE * 10) % BUFFER_SIZE;

        // Same hue palette as Rainbow Ghost
        float eB[TGHOST_ECHOES], eG[TGHOST_ECHOES], eR[TGHOST_ECHOES];
        for (int e = 0; e < TGHOST_ECHOES; e++)
        {
            float hue = fmod(rainbowHue - e * RAINBOW_HUE_STEP + 360.0f * TGHOST_ECHOES, 360.0f);
            float h6 = hue / 60.0f;
            int hi = (int)h6 % 6;
            float f = h6 - (int)h6, q = 1.0f - f;
            switch (hi)
            {
            case 0: eR[e]=1; eG[e]=f; eB[e]=0; break;
            case 1: eR[e]=q; eG[e]=1; eB[e]=0; break;
            case 2: eR[e]=0; eG[e]=1; eB[e]=f; break;
            case 3: eR[e]=0; eG[e]=q; eB[e]=1; break;
            case 4: eR[e]=f; eG[e]=0; eB[e]=1; break;
            default: eR[e]=1; eG[e]=0; eB[e]=q; break;
            }
        }

        // Per-echo display scale: newest (e=0)=1.0 (full frame), oldest=P_tunnelScale.value (>1 = zoomed in)
        // inv_s < 1 for older echoes: samples a smaller centre crop → person appears larger than frame
        float inv_s[TGHOST_ECHOES]; // precompute 1/scale to avoid division in the pixel loop
        for (int e = 0; e < TGHOST_ECHOES; e++)
        {
            float s = 1.0f + (P_tunnelScale.value - 1.0f) * (float)e / (TGHOST_ECHOES - 1);
            inv_s[e] = 1.0f / s;
        }

        float cx = (width  - 1) * 0.5f;
        float cy = (height - 1) * 0.5f;

        float ro = ringOffset;
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            Vec3b *outRow = output.ptr<Vec3b>(y);

            const uchar *maskRowE[TGHOST_ECHOES]  = {};
            const Vec3b *frameRowE[TGHOST_ECHOES] = {};
            float ixBias[TGHOST_ECHOES];
            float dy2 = (y - cy) * (y - cy);
            for (int e = 0; e < TGHOST_ECHOES; e++)
            {
                float sy = cy + (y - cy) * inv_s[e];
                int iy = (int)(sy + 0.5f);
                if ((unsigned)iy < (unsigned)height)
                {
                    maskRowE[e]  = maskBuffer[fi[e]].ptr<uchar>(iy);
                    frameRowE[e] = frameBuffer[fi[e]].ptr<Vec3b>(iy);
                }
                ixBias[e] = cx * (1.0f - inv_s[e]);
            }

            for (int x = 0; x < width; x++)
            {
                float dist  = sqrtf(dy2 + (x - cx) * (x - cx));
                float phase = fmodf(dist - ro + RING_SPACING * 1000.0f, RING_SPACING);
                uchar rv    = (phase < RING_SPACING * 0.5f) ? 35 : 0;
                outRow[x]   = Vec3b(rv, rv, rv);

                for (int e = TGHOST_ECHOES - 1; e >= 0; e--)
                {
                    if (!maskRowE[e]) continue;
                    int ix = (int)(ixBias[e] + x * inv_s[e] + 0.5f);
                    if ((unsigned)ix >= (unsigned)width) continue;
                    int alpha = maskRowE[e][ix];
                    if (alpha < 13) continue;
                    const Vec3b &p = frameRowE[e][ix];
                    float brightness = (0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2])
                                       * (alpha / 255.0f);
                    outRow[x] = Vec3b(
                        (uchar)min(255.0f, eB[e] * brightness),
                        (uchar)min(255.0f, eG[e] * brightness),
                        (uchar)min(255.0f, eR[e] * brightness));
                }
            }
        }
    }
    else if (currentMode == Mode::FLOWWARP)
    {
        // Optical flow displaces the backdrop sample coordinates per pixel.
        // Output (x,y) reads from the live frame at (x + vx*scale, y + vy*scale),
        // smearing the background in the direction of motion.
        // Person cutout (Vision mask) composited on top unwarped.
        int recent = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
        bool hasFlow = !flowMap.empty() && flowMap.rows == height && flowMap.cols == width;
        float ws  = P_flowWarp.value;
        int sr = segReady.load(memory_order_acquire);
#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++)
        {
            Vec3b       *outRow   = output.ptr<Vec3b>(y);
            const Vec2f *flowRow  = hasFlow ? flowMap.ptr<Vec2f>(y) : nullptr;
            const uchar *maskRow  = (sr >= 0) ? maskBuffer[sr].ptr<uchar>(y)  : nullptr;
            const Vec3b *frameRow = (sr >= 0) ? frameBuffer[sr].ptr<Vec3b>(y) : nullptr;
            for (int x = 0; x < width; x++)
            {
                // Warped backdrop sample
                int sx = x, sy = y;
                if (flowRow)
                {
                    sx = (int)(x + flowRow[x][0] * ws + 0.5f);
                    sy = (int)(y + flowRow[x][1] * ws + 0.5f);
                    sx = max(0, min(width  - 1, sx));
                    sy = max(0, min(height - 1, sy));
                }
                outRow[x] = frameBuffer[recent].ptr<Vec3b>(sy)[sx];

                // Person cutout on top, unwarped, soft edges from mask alpha
                if (maskRow && frameRow)
                {
                    int alpha = maskRow[x];
                    if (alpha > 12)
                    {
                        float a = alpha / 255.0f, ia = 1.0f - a;
                        const Vec3b &p  = frameRow[x];
                        const Vec3b &bg = outRow[x];
                        outRow[x] = Vec3b(
                            (uchar)(p[0] * a + bg[0] * ia),
                            (uchar)(p[1] * a + bg[1] * ia),
                            (uchar)(p[2] * a + bg[2] * ia));
                    }
                }
            }
        }
    }
    else if (currentMode == Mode::WAVEWARP)
    {
        // 2D wave simulation seeded by motion, displayed as camera refraction.
        // Wave equation: new[y][x] = (N+S+E+W)*0.5 - prev[y][x], damped each frame.
        // Motion map seeds new energy; wave gradient displaces camera sample per pixel.
        int recent = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
        bool hasMotion = !motionMap.empty() && motionMap.rows == height && motionMap.cols == width;
        float refract = P_waveRefract.value;

        // Propagate wave: read waveA (current), update waveB (previous) in-place, seed motion
#pragma omp parallel for schedule(static)
        for (int y = 1; y < height - 1; y++)
        {
            const float *rowAp = waveA.ptr<float>(y - 1);
            const float *rowA  = waveA.ptr<float>(y);
            const float *rowAn = waveA.ptr<float>(y + 1);
                  float *rowB  = waveB.ptr<float>(y);
            const float *motRow = hasMotion ? motionMap.ptr<float>(y) : nullptr;
            for (int x = 1; x < width - 1; x++)
            {
                float v = (rowAp[x] + rowAn[x] + rowA[x-1] + rowA[x+1]) * 0.5f - rowB[x];
                v *= WAVE_DAMP;
                if (motRow) v += motRow[x] * WAVE_SEED;
                rowB[x] = v;
            }
        }
        std::swap(waveA, waveB); // waveA now holds the freshly propagated result

        // Render: use waveA gradient to refract the full camera frame
#pragma omp parallel for schedule(static)
        for (int y = 1; y < height - 1; y++)
        {
            Vec3b       *outRow = output.ptr<Vec3b>(y);
            const float *waveP  = waveA.ptr<float>(y - 1);
            const float *waveC  = waveA.ptr<float>(y);
            const float *waveN  = waveA.ptr<float>(y + 1);
            for (int x = 1; x < width - 1; x++)
            {
                float dx = waveC[x + 1] - waveC[x - 1];
                float dy = waveN[x]     - waveP[x];
                int sx = (int)(x + dx * refract + 0.5f);
                int sy = (int)(y + dy * refract + 0.5f);
                sx = max(0, min(width  - 1, sx));
                sy = max(0, min(height - 1, sy));
                outRow[x] = frameBuffer[recent].ptr<Vec3b>(sy)[sx];
            }
            outRow[0]       = frameBuffer[recent].ptr<Vec3b>(y)[0];
            outRow[width-1] = frameBuffer[recent].ptr<Vec3b>(y)[width-1];
        }
        // Edge rows: copy direct from camera
        memcpy(output.ptr(0),          frameBuffer[recent].ptr(0),          width * 3);
        memcpy(output.ptr(height - 1), frameBuffer[recent].ptr(height - 1), width * 3);
    }
    else if (currentMode == Mode::CHROMAWAVE)
    {
        // Three independent wave simulations, each seeded from its own colour channel's diff.
        // R/G/B channels of the output each sample the camera at coordinates displaced by
        // their own wave's gradient — different-coloured motion creates independent ripple patterns.
        int recent  = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
        int prev_f  = (bufIdx - 1 - MOTION_LOOKBACK + BUFFER_SIZE * 4) % BUFFER_SIZE;
        float refract = P_chromaWave.value;

        // Propagate all three waves in one OMP pass; seed each from its own colour diff
#pragma omp parallel for schedule(static)
        for (int y = 1; y < height - 1; y++)
        {
            const float *rAp = waveAr.ptr<float>(y-1), *rA = waveAr.ptr<float>(y), *rAn = waveAr.ptr<float>(y+1); float *rB = waveBr.ptr<float>(y);
            const float *gAp = waveAg.ptr<float>(y-1), *gA = waveAg.ptr<float>(y), *gAn = waveAg.ptr<float>(y+1); float *gB = waveBg.ptr<float>(y);
            const float *bAp = waveAb.ptr<float>(y-1), *bA = waveAb.ptr<float>(y), *bAn = waveAb.ptr<float>(y+1); float *bB = waveBb.ptr<float>(y);
            const Vec3b *currRow = frameBuffer[recent].ptr<Vec3b>(y);
            const Vec3b *prevRow = frameBuffer[prev_f].ptr<Vec3b>(y);
            for (int x = 1; x < width - 1; x++)
            {
                float dr = abs((int)currRow[x][2] - (int)prevRow[x][2]) * (1.0f/255.0f);
                float dg = abs((int)currRow[x][1] - (int)prevRow[x][1]) * (1.0f/255.0f);
                float db = abs((int)currRow[x][0] - (int)prevRow[x][0]) * (1.0f/255.0f);
                rB[x] = ((rAp[x]+rAn[x]+rA[x-1]+rA[x+1])*0.5f - rB[x]) * CWAVE_DAMP + dr * CWAVE_SEED;
                gB[x] = ((gAp[x]+gAn[x]+gA[x-1]+gA[x+1])*0.5f - gB[x]) * CWAVE_DAMP + dg * CWAVE_SEED;
                bB[x] = ((bAp[x]+bAn[x]+bA[x-1]+bA[x+1])*0.5f - bB[x]) * CWAVE_DAMP + db * CWAVE_SEED;
            }
        }
        std::swap(waveAr, waveBr);
        std::swap(waveAg, waveBg);
        std::swap(waveAb, waveBb);

        // Render: each colour channel sampled at coordinates displaced by its own wave gradient
#pragma omp parallel for schedule(static)
        for (int y = 1; y < height - 1; y++)
        {
            Vec3b       *outRow  = output.ptr<Vec3b>(y);
            const float *rP = waveAr.ptr<float>(y-1), *rC = waveAr.ptr<float>(y), *rN = waveAr.ptr<float>(y+1);
            const float *gP = waveAg.ptr<float>(y-1), *gC = waveAg.ptr<float>(y), *gN = waveAg.ptr<float>(y+1);
            const float *bP = waveAb.ptr<float>(y-1), *bC = waveAb.ptr<float>(y), *bN = waveAb.ptr<float>(y+1);
            for (int x = 1; x < width - 1; x++)
            {
                auto clampW = [&](int v, int mx){ return v < 0 ? 0 : v >= mx ? mx-1 : v; };

                // Each channel's gradient is rotated 120° apart so they displace in
                // different spatial directions, guaranteeing vivid colour separation.
                // R: 0°, G: 120°, B: 240°  (√3/2 ≈ 0.866)
                float rdx = rC[x+1]-rC[x-1],  rdy = rN[x]-rP[x];
                float gdx = gC[x+1]-gC[x-1],  gdy = gN[x]-gP[x];
                float bdx = bC[x+1]-bC[x-1],  bdy = bN[x]-bP[x];

                int sxR = clampW((int)(x + rdx * refract + 0.5f), width);
                int syR = clampW((int)(y + rdy * refract + 0.5f), height);
                int sxG = clampW((int)(x + (-0.5f*gdx - 0.866f*gdy) * refract + 0.5f), width);
                int syG = clampW((int)(y + ( 0.866f*gdx - 0.5f*gdy) * refract + 0.5f), height);
                int sxB = clampW((int)(x + (-0.5f*bdx + 0.866f*bdy) * refract + 0.5f), width);
                int syB = clampW((int)(y + (-0.866f*bdx - 0.5f*bdy) * refract + 0.5f), height);

                uchar R = frameBuffer[recent].ptr<Vec3b>(syR)[sxR][2];
                uchar G = frameBuffer[recent].ptr<Vec3b>(syG)[sxG][1];
                uchar B = frameBuffer[recent].ptr<Vec3b>(syB)[sxB][0];
                outRow[x] = Vec3b(B, G, R);
            }
            outRow[0]       = frameBuffer[recent].ptr<Vec3b>(y)[0];
            outRow[width-1] = frameBuffer[recent].ptr<Vec3b>(y)[width-1];
        }
        memcpy(output.ptr(0),          frameBuffer[recent].ptr(0),          width * 3);
        memcpy(output.ptr(height - 1), frameBuffer[recent].ptr(height - 1), width * 3);
    }
    else if (currentMode == Mode::FLOWRIPPLE)
    {
        // rippleBuffer is maintained by the preprocessing block (advect + decay + inject).
        // Convert to 8-bit (saturating) and add additively over the current frame so
        // still areas show the live video and moving areas glow with directional ripple color.
        int recent = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
        rippleBuffer.convertTo(ripple8, CV_8UC3, 1.0);
        cv::add(frameBuffer[recent], ripple8, output);
    }
    else if (currentMode == Mode::TURBULENCE)
    {
        // turbulenceMap is maintained by the preprocessing block.
        // Per-pixel: turbulence level (0=still, 1=max) drives:
        //   • animated sine-wave displacement (scaled by P_turbShift.value)
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
                int dx = (int)(nx * t * P_turbShift.value);
                int dy = (int)(ny * t * P_turbShift.value);
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
    else if (currentMode == Mode::FLOWHUE)
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
                float sat = min(mag / P_flowSens.value, 1.0f);
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
    switch (currentMode) {
        case Mode::W:              return "W: Bottom to Top";
        case Mode::S:              return "S: Top to Bottom";
        case Mode::A:              return "A: Right to Left";
        case Mode::D:              return "D: Left to Right";
        case Mode::WS:             return "W+S: Center-Out Vertical";
        case Mode::AD:             return "A+D: Center-Out Horizontal";
        case Mode::MOTION:         return "Z: Motion Adaptive";
        case Mode::CHROMA:         return "X: Chromatic Time Shift";
        case Mode::MCHROMA:        return "J: Motion Chromatic";
        case Mode::PRISMATIC:      return "T: Prismatic Echo";
        case Mode::PRISMATICGHOST: return "T: Prismatic Ghost";
        case Mode::FLOWHUE:        return "Y: Flow Direction Color";
        case Mode::FLOWRIPPLE:     return "Y: Flow Color Ripple";
        case Mode::DATAMOSH:       return "U: Datamosh";
        case Mode::TURBULENCE:     return "I: Turbulence";
        case Mode::GHOSTECHO:      return "C: Ghost Echo";
        case Mode::CHROMAGHOSTECHO:return "C: Chroma Ghost Echo";
        case Mode::TIMEGHOST:      return "H: Temporal Ghost";
        case Mode::TUNNELTIMEGHOST:return "H: Tunnel Time Ghost";
        case Mode::RAINBOWGHOST:   return "G: Rainbow Ghost";
        case Mode::TUNNELGHOST:    return "G: Tunnel Ghost";
        case Mode::FLOWWARP:       return "V: Flow Warp";
        case Mode::WAVEWARP:       return "K: Wave Warp";
        case Mode::CHROMAWAVE:     return "K: Chroma Wave";
    }
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

    // Pre-allocate motion / flow working buffers (main thread scratch + double-buffered maps)
    int flowW = max(1, (int)(actualWidth  * FLOW_SCALE));
    int flowH = max(1, (int)(actualHeight * FLOW_SCALE));
    diffMat    = Mat::zeros(actualHeight, actualWidth, CV_8UC3);
    grayDiff   = Mat::zeros(actualHeight, actualWidth, CV_8U);
    motionSmall = Mat::zeros(flowH, flowW, CV_8U);
    turbScratch = Mat::zeros(actualHeight, actualWidth, CV_32F);
    // Double-buffered maps written by preprocessLoop, read by main thread
    motionMapBuf[0] = Mat::zeros(actualHeight, actualWidth, CV_32F);
    motionMapBuf[1] = Mat::zeros(actualHeight, actualWidth, CV_32F);
    flowMapBuf[0]   = Mat::zeros(actualHeight, actualWidth, CV_32FC2);
    flowMapBuf[1]   = Mat::zeros(actualHeight, actualWidth, CV_32FC2);
    // Aliases that main thread updates each frame to point at the active buffer
    motionMap = motionMapBuf[0];
    flowMap   = flowMapBuf[0];
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

    // Pre-allocate wave simulation buffers (Wave Warp mode)
    waveA = Mat::zeros(actualHeight, actualWidth, CV_32F);
    waveB = Mat::zeros(actualHeight, actualWidth, CV_32F);

    // Pre-allocate per-channel wave buffers (Chroma Wave mode)
    waveAr = Mat::zeros(actualHeight, actualWidth, CV_32F);
    waveBr = Mat::zeros(actualHeight, actualWidth, CV_32F);
    waveAg = Mat::zeros(actualHeight, actualWidth, CV_32F);
    waveBg = Mat::zeros(actualHeight, actualWidth, CV_32F);
    waveAb = Mat::zeros(actualHeight, actualWidth, CV_32F);
    waveBb = Mat::zeros(actualHeight, actualWidth, CV_32F);

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
    cout << "  T            - Prismatic echo (6 hue-tinted temporal echoes)" << endl;
    cout << "  Y            - Flow color ripple (directional color that lingers and drifts)" << endl;
    cout << "  U            - Datamosh (motion trails via IIR diff accumulation)" << endl;
    cout << "  I            - Turbulence (motion history → displacement + chroma + saturation)" << endl;
    cout << "  G            - Rainbow Ghost (like H but echoes tinted with cycling hues)" << endl;
    cout << "  H            - Temporal Ghost (7 person silhouettes through time on black)" << endl;
    cout << "  J            - Motion chromatic (still=normal, moving=RGB time split)" << endl;
    cout << "  K            - Flow direction color (direction→hue, speed→saturation)" << endl;
    cout << "  Z            - Motion Adaptive" << endl;
    cout << "  X            - Chromatic Time Shift" << endl;
    cout << "  C            - Ghost Echo (7 motion-masked temporal echoes on black)" << endl;
    cout << "  V            - Flow Warp (optical flow distorts live backdrop; person cutout on top)" << endl;
    cout << "  N            - Wave Warp (2D wave simulation seeded by motion; camera refraction)" << endl;
    cout << "  M            - Chroma Wave (3 independent waves, one per RGB channel; per-channel refraction)" << endl;
    cout << "  B            - Tunnel Ghost (rainbow ghost with echoes scaled into tunnel)" << endl;
    cout << "  Up/Down      - Speed / Chroma / Flow sens / Spread / Echo / Band ht" << endl;
    cout << "  R            - Reset the above to defaults" << endl;
    cout << "  F            - Toggle fullscreen" << endl;
    cout << "  Q/ESC        - Quit" << endl;
    cout << "\nStarting...\n"
         << endl;

    // Start capture thread — decouples camera I/O from render loop
    thread captureThread(captureLoop, ref(cap));
    thread prepThread(preprocessLoop, actualWidth, actualHeight);

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

        // Update motionMap/flowMap aliases to the latest buffer from preprocessLoop.
        // preprocessLoop runs concurrently and writes into motionMapBuf/flowMapBuf;
        // prepBuf (atomic) tells us which buffer is safe to read.
        {
            int rb = prepBuf.load(memory_order_acquire);
            motionMap = motionMapBuf[rb]; // O(1) shallow header copy
            flowMap   = flowMapBuf[rb];
        }

        // Update datamosh accumulator — used by datamosh mode.
        // Diffs MOTION_LOOKBACK frames apart (not adjacent) so pixel displacement
        // is large enough to be visible at 60fps. Adjacent frames produce near-zero
        // diffs that collapse the accumulator to black before trails can build up.
        // Reuses diffMat (CV_8UC3) scratch buffer already declared for motion mode.
        if (currentMode == Mode::DATAMOSH)
        {
            int curr = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
            int prev = (bufIdx - 1 - MOTION_LOOKBACK + BUFFER_SIZE * 2) % BUFFER_SIZE;
            float datamoshBoost = DATAMOSH_BOOST_K * (1.0f - P_datamosh.value);
            // Fuse absdiff + convertTo + addWeighted + convertTo(output) into one pass.
            // Eliminates datamoshDiffF scratch buffer and 3 extra full-frame traversals.
#pragma omp parallel for schedule(static)
            for (int y = 0; y < actualHeight; y++)
            {
                const uchar *currRow = frameBuffer[curr].ptr(y);
                const uchar *prevRow = frameBuffer[prev].ptr(y);
                float       *accumRow = datamoshAccum.ptr<float>(y);
                uchar       *outRow   = output.ptr(y);
                const int n = actualWidth * 3;
                for (int i = 0; i < n; i++)
                {
                    float diff = (float)(currRow[i] > prevRow[i]
                                         ? currRow[i] - prevRow[i]
                                         : prevRow[i] - currRow[i]);
                    float val = accumRow[i] * P_datamosh.value + diff * datamoshBoost;
                    accumRow[i] = val;
                    outRow[i]   = val < 255.0f ? (uchar)val : 255;
                }
            }
        }

        // Update turbulence accumulator — used by turbulence mode.
        // Diffs current vs MOTION_LOOKBACK frames ago, blurs spatially, then IIR-decays
        // into turbulenceMap. Still areas fade to 0 over ~2s; motion spikes quickly to 1.
        if (currentMode == Mode::TURBULENCE)
        {
            int recent = (bufIdx - 1 + BUFFER_SIZE * 2) % BUFFER_SIZE;
            int older = (bufIdx - 1 - MOTION_LOOKBACK + BUFFER_SIZE * 2) % BUFFER_SIZE;
            cv::absdiff(frameBuffer[recent], frameBuffer[older], diffMat);
            cv::cvtColor(diffMat, grayDiff, COLOR_BGR2GRAY);
            cv::resize(grayDiff, motionSmall, motionSmall.size(), 0, 0, INTER_LINEAR);
            cv::GaussianBlur(motionSmall, motionSmall, cv::Size(MOTION_BLUR_SIZE, MOTION_BLUR_SIZE), 0);
            cv::resize(motionSmall, grayDiff, grayDiff.size(), 0, 0, INTER_LINEAR);
            cv::threshold(grayDiff, grayDiff, 20, 255, THRESH_TOZERO); // suppress camera noise
            grayDiff.convertTo(turbScratch, CV_32F, 0.5 / 255.0);
            turbulenceMap *= turbDecay;
            cv::add(turbulenceMap, turbScratch, turbulenceMap);
            cv::min(turbulenceMap, 1.0f, turbulenceMap);
            turbFrame++;
        }

        // Advance rainbow hue — used by rainbowghost and tunnelghost modes.
        if (currentMode == Mode::RAINBOWGHOST || currentMode == Mode::TUNNELGHOST ||
            currentMode == Mode::CHROMAGHOSTECHO || currentMode == Mode::PRISMATICGHOST)
            rainbowHue = fmod(rainbowHue + rainbowSpeed / 60.0f, 360.0f);
        // Advance ring backdrop phase — used by ghost modes and ring warp.
        if (currentMode == Mode::RAINBOWGHOST || currentMode == Mode::TIMEGHOST ||
            currentMode == Mode::TUNNELGHOST || currentMode == Mode::TUNNELTIMEGHOST ||
            currentMode == Mode::PRISMATICGHOST || currentMode == Mode::FLOWWARP)
            ringOffset = fmodf(ringOffset + RING_SPEED / 60.0f, RING_SPACING);

        // Update flow ripple buffer — advect, decay, inject — used by flowripple mode.
        // 1. Build per-pixel backward-warp maps from flowMap.
        // 2. remap advects existing color content forward (in the flow direction).
        // 3. Decay: multiply by P_rippleDecay.value (~1 second lifetime at 60fps).
        // 4. Inject: where flow is strong, add a fresh saturated directional color additively.
        if (currentMode == Mode::FLOWRIPPLE)
        {
            const float PI2 = 2.0f * 3.14159265f;
            const float invSens = 1.0f / P_flowSens.value;
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
            rippleTmp *= P_rippleDecay.value;

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
        else if (key == 'z')
        {
            currentMode = Mode::MOTION;
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'x')
        {
            currentMode = Mode::CHROMA;
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'j')
        {
            currentMode = Mode::MCHROMA;
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 't')
        {
            if (currentMode == Mode::PRISMATIC || currentMode == Mode::PRISMATICGHOST)
                currentMode = (currentMode == Mode::PRISMATIC) ? Mode::PRISMATICGHOST : Mode::PRISMATIC;
            else
                currentMode = lastT;
            lastT = currentMode;
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'y')
        {
            if (currentMode == Mode::FLOWRIPPLE || currentMode == Mode::FLOWHUE)
                currentMode = (currentMode == Mode::FLOWRIPPLE) ? Mode::FLOWHUE : Mode::FLOWRIPPLE;
            else
                currentMode = lastY;
            lastY = currentMode;
            if (currentMode == Mode::FLOWRIPPLE) rippleBuffer.setTo(0);
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'i')
        {
            currentMode = Mode::TURBULENCE;
            turbulenceMap.setTo(0);
            turbFrame = 0;
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'u')
        {
            currentMode = Mode::DATAMOSH;
            datamoshAccum.setTo(0); // fresh slate each entry
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'c')
        {
            if (currentMode == Mode::GHOSTECHO || currentMode == Mode::CHROMAGHOSTECHO)
                currentMode = (currentMode == Mode::GHOSTECHO) ? Mode::CHROMAGHOSTECHO : Mode::GHOSTECHO;
            else
                currentMode = lastC;
            lastC = currentMode;
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'h')
        {
            if (currentMode == Mode::TIMEGHOST || currentMode == Mode::TUNNELTIMEGHOST)
                currentMode = (currentMode == Mode::TIMEGHOST) ? Mode::TUNNELTIMEGHOST : Mode::TIMEGHOST;
            else
                currentMode = lastH;
            lastH = currentMode;
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'g')
        {
            if (currentMode == Mode::RAINBOWGHOST || currentMode == Mode::TUNNELGHOST)
                currentMode = (currentMode == Mode::RAINBOWGHOST) ? Mode::TUNNELGHOST : Mode::RAINBOWGHOST;
            else
                currentMode = lastG;
            lastG = currentMode;
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'v')
        {
            currentMode = Mode::FLOWWARP;
            cout << "Mode: " << getModeName() << endl;
        }
        else if (key == 'k')
        {
            if (currentMode == Mode::WAVEWARP || currentMode == Mode::CHROMAWAVE)
                currentMode = (currentMode == Mode::WAVEWARP) ? Mode::CHROMAWAVE : Mode::WAVEWARP;
            else
                currentMode = lastK;
            lastK = currentMode;
            if (currentMode == Mode::CHROMAWAVE) {
                waveAr.setTo(0); waveBr.setTo(0);
                waveAg.setTo(0); waveBg.setTo(0);
                waveAb.setTo(0); waveBb.setTo(0);
            } else {
                waveA.setTo(0); waveB.setTo(0);
            }
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
            // Returns the Param to reset for the current mode, or nullptr for speed reset.
            ModeParam* p = nullptr;
            if      (currentMode == Mode::MOTION)                                     p = &P_motionDepth;
            else if (currentMode == Mode::CHROMA)                                     p = &P_chromaOffset;
            else if (currentMode == Mode::FLOWHUE)                                    p = &P_flowSens;
            else if (currentMode == Mode::MCHROMA)                                    p = &P_chromaSpread;
            else if (currentMode == Mode::PRISMATIC)                                  p = &P_echoSpacing;
            else if (currentMode == Mode::DATAMOSH)                                   p = &P_datamosh;
            else if (currentMode == Mode::FLOWRIPPLE)                                 p = &P_rippleDecay;
            else if (currentMode == Mode::TURBULENCE)                                 p = &P_turbShift;
            else if (currentMode == Mode::GHOSTECHO || currentMode == Mode::CHROMAGHOSTECHO) p = &P_ghostSpace;
            else if (currentMode == Mode::TIMEGHOST  || currentMode == Mode::RAINBOWGHOST)   p = &P_tghostSpace;
            else if (currentMode == Mode::TUNNELGHOST || currentMode == Mode::TUNNELTIMEGHOST) p = &P_tunnelScale;
            else if (currentMode == Mode::PRISMATICGHOST)                             p = &P_glowBoost;
            else if (currentMode == Mode::FLOWWARP)                                   p = &P_flowWarp;
            else if (currentMode == Mode::WAVEWARP)                                   p = &P_waveRefract;
            else if (currentMode == Mode::CHROMAWAVE)                                 p = &P_chromaWave;
            if (p) { p->reset(); overlayText = p->display(); }
            else   { updateSpeed.store(1); overlayText = "Speed: 1"; }
            cout << overlayText << endl;
            lastOverlayTime = steady_clock::now();
            overlayActive = true;
        }
        else if (key == 0)
        { // Up arrow (Mac)
            ModeParam* p = nullptr;
            if      (currentMode == Mode::MOTION)                                     p = &P_motionDepth;
            else if (currentMode == Mode::CHROMA)                                     p = &P_chromaOffset;
            else if (currentMode == Mode::FLOWHUE)                                    p = &P_flowSens;
            else if (currentMode == Mode::MCHROMA)                                    p = &P_chromaSpread;
            else if (currentMode == Mode::PRISMATIC)                                  p = &P_echoSpacing;
            else if (currentMode == Mode::DATAMOSH)                                   p = &P_datamosh;
            else if (currentMode == Mode::FLOWRIPPLE)                                 p = &P_rippleDecay;
            else if (currentMode == Mode::TURBULENCE)                                 p = &P_turbShift;
            else if (currentMode == Mode::GHOSTECHO || currentMode == Mode::CHROMAGHOSTECHO) p = &P_ghostSpace;
            else if (currentMode == Mode::TIMEGHOST  || currentMode == Mode::RAINBOWGHOST)   p = &P_tghostSpace;
            else if (currentMode == Mode::TUNNELGHOST || currentMode == Mode::TUNNELTIMEGHOST) p = &P_tunnelScale;
            else if (currentMode == Mode::PRISMATICGHOST)                             p = &P_glowBoost;
            else if (currentMode == Mode::FLOWWARP)                                   p = &P_flowWarp;
            else if (currentMode == Mode::WAVEWARP)                                   p = &P_waveRefract;
            else if (currentMode == Mode::CHROMAWAVE)                                 p = &P_chromaWave;
            if (p) { p->up(); overlayText = p->display(); }
            else   { updateSpeed.store(min(BUFFER_SIZE, updateSpeed.load() + 1));
                     overlayText = "Speed: " + to_string(updateSpeed.load()); }
            cout << overlayText << endl;
            lastOverlayTime = steady_clock::now();
            overlayActive = true;
        }
        else if (key == 1)
        { // Down arrow (Mac)
            ModeParam* p = nullptr;
            if      (currentMode == Mode::MOTION)                                     p = &P_motionDepth;
            else if (currentMode == Mode::CHROMA)                                     p = &P_chromaOffset;
            else if (currentMode == Mode::FLOWHUE)                                    p = &P_flowSens;
            else if (currentMode == Mode::MCHROMA)                                    p = &P_chromaSpread;
            else if (currentMode == Mode::PRISMATIC)                                  p = &P_echoSpacing;
            else if (currentMode == Mode::DATAMOSH)                                   p = &P_datamosh;
            else if (currentMode == Mode::FLOWRIPPLE)                                 p = &P_rippleDecay;
            else if (currentMode == Mode::TURBULENCE)                                 p = &P_turbShift;
            else if (currentMode == Mode::GHOSTECHO || currentMode == Mode::CHROMAGHOSTECHO) p = &P_ghostSpace;
            else if (currentMode == Mode::TIMEGHOST  || currentMode == Mode::RAINBOWGHOST)   p = &P_tghostSpace;
            else if (currentMode == Mode::TUNNELGHOST || currentMode == Mode::TUNNELTIMEGHOST) p = &P_tunnelScale;
            else if (currentMode == Mode::PRISMATICGHOST)                             p = &P_glowBoost;
            else if (currentMode == Mode::FLOWWARP)                                   p = &P_flowWarp;
            else if (currentMode == Mode::WAVEWARP)                                   p = &P_waveRefract;
            else if (currentMode == Mode::CHROMAWAVE)                                 p = &P_chromaWave;
            if (p) { p->down(); overlayText = p->display(); }
            else   { updateSpeed.store(max(1, updateSpeed.load() - 1));
                     overlayText = "Speed: " + to_string(updateSpeed.load()); }
            cout << overlayText << endl;
            lastOverlayTime = steady_clock::now();
            overlayActive = true;
        }
    }

    running = false;
    captureThread.join();
    prepThread.join();
    segThread.join();

    cap.release();
    destroyAllWindows();
    cout << "\nProgram ended" << endl;
    return 0;
}
