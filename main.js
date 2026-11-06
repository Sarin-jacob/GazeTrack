// main.js - Camera, calibration flow and UI. The gaze maths lives in gaze-model.js.

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
import { RIGHT_IRIS, LEFT_IRIS, extractFrame, fitModel, predict, isBlink, OneEuroFilter } from "./gaze-model.js";
const { FaceLandmarker, FilesetResolver } = vision;

// --- DOM Elements ---
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const gazeDot = document.getElementById('gaze-dot');
const calOverlay = document.getElementById('calibration-overlay');
const calPoint = document.getElementById('cal-point');
const calStatus = document.getElementById('cal-status');
const statusText = document.getElementById('status-text');
const toggleVideoBtn = document.getElementById('toggle-video-btn');
const startCalBtn = document.getElementById('start-cal-btn');
const clearCalBtn = document.getElementById('clear-cal-btn');
const testBtn = document.getElementById('test-btn');

// --- App State ---
let state = {
    faceLandmarker: null,
    modelLoaded: false,
    isCameraOn: false,
    isCalibrating: false,
    animationFrameId: null,
    videoStream: null,
    model: null,
    onFrame: null, // Set during calibration to receive each new frame.
    lastVideoTime: -1,
    filters: { x: new OneEuroFilter(), y: new OneEuroFilter() }
};

// --- Constants ---
const CALIBRATION_POINT_COUNT = 15;
const SETTLE_DURATION_MS = 300; // The eyes need a moment to land on the point.
const HOLD_DURATION_MS = 1300;
const TRANSITION_DURATION_MS = 700;
const READ_DURATION_MS = 1800; // Time to read a head-movement prompt before samples are taken.
const HEAD_MOVE_DURATION_MS = 4500;
const MIN_SAMPLES = 200;
const MIN_HEAD_MOTION = { tanYaw: 0.03, tx: 0.7 }; // Standard deviations; below both, the head barely moved.
const OPEN_EYE_RATIO = 0.7; // Frames this far under a point's typical eye openness are blinks.

// The model can only learn what head movement does if it sees some. Each step moves the head
// while the eyes stay on a fixed dot, so the gaze target is known while head and eyes both change.
const HEAD_MOVES = [
    { point: { x: 0.5, y: 0.5 }, text: 'Eyes on the dot. Slowly turn your head left and right' },
    { point: { x: 0.1, y: 0.1 }, text: 'Eyes on the dot. Slowly nod up and down' },
    { point: { x: 0.9, y: 0.1 }, text: 'Eyes on the dot. Slide your head left and right' },
    { point: { x: 0.9, y: 0.9 }, text: 'Eyes on the dot. Lean in, then back' },
    { point: { x: 0.1, y: 0.9 }, text: 'Eyes on the dot. Sit up tall, then slouch' }
];

// Accuracy test: look at known points with the head held in different places. The points are
// off the calibration grid, so this measures the model on data it was not fitted to.
const TEST_POSE_DURATION_MS = 3500;
const TEST_SETTLE_DURATION_MS = 500;
const TEST_HOLD_DURATION_MS = 1000;
const TEST_POINTS = [{ x: 0.3, y: 0.3 }, { x: 0.7, y: 0.3 }, { x: 0.7, y: 0.7 }, { x: 0.3, y: 0.7 }];
const TEST_POSES = [
    { name: 'still', text: 'Sit normally' },
    { name: 'left', text: 'Move your head about 10 cm to your left and stay there' },
    { name: 'right', text: 'Move your head about 10 cm to your right and stay there' },
    { name: 'closer', text: 'Lean about 15 cm closer and stay there' },
    { name: 'further', text: 'Lean about 15 cm back and stay there' },
    { name: 'turned', text: 'Turn your head a little to one side and keep it there' }
];

/**
 * Generates a grid of calibration points dynamically based on the viewport's aspect ratio,
 * including points near the edges.
 * @param {number} totalPoints - The approximate number of points you want in the grid.
 * @param {number} margin - The padding from the edge (e.g., 0.05 for 5%).
 * @returns {Array<Object>} An array of {x, y} point objects.
 */
function generateCalibrationPoints(totalPoints = 12, margin = 0.05) {
    const points = [];
    const aspectRatio = window.innerWidth / window.innerHeight;

    // Calculate the ideal number of points for each axis
    let ny = Math.max(2, Math.round(Math.sqrt(totalPoints / aspectRatio)));
    let nx = Math.max(2, Math.round(ny * aspectRatio));

    // The total range to generate points within, e.g., 0.9 for a 0.05 margin (1.0 - 2 * 0.05)
    const span = 1 - 2 * margin;

    nx= Math.max(3, nx);
    ny= Math.max(3, ny);
    for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
            // The formula i / (n - 1) creates evenly spaced points from 0 to 1.
            // We then scale by our span and shift by the margin.
            const x = margin + (i / (nx - 1)) * span;
            const y = margin + (j / (ny - 1)) * span;

            points.push({ x, y });
        }
    }

    console.log(`Generated a ${nx}x${ny} edge-to-edge grid (${points.length} points) for aspect ratio ${aspectRatio.toFixed(2)}.`);
    return points;
}

// --- Main Initialization ---
async function initialize() {
    await setupFaceLandmarker();
    loadCalibrationModel();
    addEventListeners();
    updateUI();
}

// --- MediaPipe and Camera Setup ---
async function setupFaceLandmarker() {
    updateStatus('Loading model...');
    try {
        const filesetResolver = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm');
        state.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' },
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: true,
            runningMode: 'VIDEO',
            numFaces: 1
        });
        state.modelLoaded = true;
        updateStatus('Model loaded.');
    } catch (error) {
        console.error("Error loading FaceLandmarker model:", error);
        updateStatus('Error loading model.');
    }
}

async function toggleCamera() {
    if (state.isCameraOn) {
        stopCamera();
    } else {
        await startCamera();
    }
    updateUI();
}

async function startCamera() {
    try {
        const constraints = {
            video: {
                facingMode: 'user',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };
        state.videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = state.videoStream;

        video.onloadedmetadata = () => {
            video.play();

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const stage = document.getElementById('stage');
            stage.style.aspectRatio = video.videoWidth / video.videoHeight;
            state.isCameraOn = true;
            updateUI();
            updateStatus('Camera on.');
            predictLoop();

            console.log(`Camera started with resolution: ${video.videoWidth}x${video.videoHeight}`);
        };

    } catch (error) {
        console.error("Error accessing camera:", error);
        updateStatus('Camera access denied.');
    }
}

function stopCamera() {
    if (state.videoStream) {
        state.videoStream.getTracks().forEach(track => track.stop());
    }
    video.srcObject = null;
    state.isCameraOn = false;
    cancelAnimationFrame(state.animationFrameId);
    updateStatus('Camera off.');
    updateUI();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// --- Prediction Loop ---
// The only place that runs detection, so calibration and tracking never process a frame twice.
async function predictLoop() {
    if (!state.isCameraOn || !state.modelLoaded) return;

    try {
        if (video.currentTime !== state.lastVideoTime) {
            state.lastVideoTime = video.currentTime;
            processFrame(state.faceLandmarker.detectForVideo(video, performance.now()));
        }
    } catch (error) {
        console.error("Prediction error:", error);
    }

    state.animationFrameId = requestAnimationFrame(predictLoop);
}

function processFrame(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (results.faceLandmarks.length === 0 || !results.facialTransformationMatrixes?.length) return;

    const landmarks = results.faceLandmarks[0];
    const irisData = getIrisData(landmarks);
    drawIrisPoints(irisData.leftCenter, irisData.rightCenter);

    const matrix = results.facialTransformationMatrixes[0].data;
    const frame = extractFrame(landmarks, matrix, video.videoWidth, video.videoHeight);

    if (state.onFrame) {
        state.onFrame(frame);
    } else if (state.model && !state.isCalibrating) {
        predictGaze(frame);
    }
}

function getIrisData(landmarks) {
    const avg = (points) => points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
    const rpts = RIGHT_IRIS.map(i => landmarks[i]);
    const lpts = LEFT_IRIS.map(i => landmarks[i]);
    const rightCenter = { x: avg(rpts).x / rpts.length, y: avg(rpts).y / rpts.length };
    const leftCenter = { x: avg(lpts).x / lpts.length, y: avg(lpts).y / lpts.length };
    return { rightCenter, leftCenter };
}

function drawIrisPoints(lc, rc) {
    ctx.fillStyle = 'rgba(100, 255, 100, 0.8)';
    ctx.beginPath();
    ctx.arc(lc.x * canvas.width, lc.y * canvas.height, 5, 0, 2 * Math.PI);
    ctx.fill();

    ctx.fillStyle = 'rgba(255, 100, 100, 0.8)';
    ctx.beginPath();
    ctx.arc(rc.x * canvas.width, rc.y * canvas.height, 5, 0, 2 * Math.PI);
    ctx.fill();
}

// --- Calibration Logic ---
async function enterOverlay() {
    state.isCalibrating = true;
    updateUI();

    try {
        await document.documentElement.requestFullscreen();
    } catch (err) {
        console.warn("Could not enter fullscreen.");
    }

    calOverlay.style.display = 'flex';
}

function exitOverlay() {
    if (document.fullscreenElement) {
        document.exitFullscreen();
    }
    calOverlay.style.display = 'none';
    state.isCalibrating = false;
    updateUI();
}

async function startCalibration() {
    await enterOverlay();
    let lastPoint = { x: 0.5, y: 0.5 };
    placeCalPoint(lastPoint.x, lastPoint.y);
    calStatus.textContent = "Sit comfortably. Keep your eyes on the dot.";
    await wait(2000);

    // Generated here, not at load, so the grid matches the fullscreen aspect ratio.
    const points = generateCalibrationPoints(CALIBRATION_POINT_COUNT);
    const gridData = [];

    for (let i = 0; i < points.length; i++) {
        const point = points[i];
        calStatus.textContent = "Follow the point";
        await animatePointTransition(lastPoint, point, TRANSITION_DURATION_MS);
        calStatus.textContent = `Hold your gaze (${i + 1}/${points.length})`;
        await wait(SETTLE_DURATION_MS);
        gridData.push(...await collectSamples(point, HOLD_DURATION_MS));
        lastPoint = point;
    }

    const headData = [];
    for (const { point, text } of HEAD_MOVES) {
        calStatus.textContent = text;
        await animatePointTransition(lastPoint, point, TRANSITION_DURATION_MS);
        await wait(READ_DURATION_MS);
        headData.push(...await collectSamples(point, HEAD_MOVE_DURATION_MS));
        lastPoint = point;
    }

    const allData = [...gridData, ...headData];
    if (allData.length >= MIN_SAMPLES) {
        state.model = fitModel(allData);
        state.model.fullscreen = Boolean(document.fullscreenElement);
        resetFilters();
        saveCalibrationModel();
        window.gazeDebug.calibration = allData;
        updateStatus(describeCalibration(state.model, headData));
    } else {
        updateStatus(`Calibration failed: not enough data. (Collected ${allData.length})`);
    }

    exitOverlay();
}

// --- Accuracy Test ---
async function startAccuracyTest() {
    await enterOverlay();
    calPoint.classList.add('test');
    let lastPoint = { x: 0.5, y: 0.5 };
    placeCalPoint(lastPoint.x, lastPoint.y);

    const results = [];
    for (const pose of TEST_POSES) {
        calStatus.textContent = pose.text;
        await wait(TEST_POSE_DURATION_MS);
        calStatus.textContent = 'Look at the dot';

        const samples = [];
        for (const point of TEST_POINTS) {
            await animatePointTransition(lastPoint, point, TRANSITION_DURATION_MS);
            await wait(TEST_SETTLE_DURATION_MS);
            samples.push(...await collectSamples(point, TEST_HOLD_DURATION_MS));
            lastPoint = point;
        }
        results.push({ pose: pose.name, error: medianError(samples), samples });
    }

    calPoint.classList.remove('test');
    window.gazeDebug.accuracyTest = results;
    updateStatus(`Median error, % of screen: ${results.map(r => `${r.pose} ${(r.error * 100).toFixed(1)}`).join(', ')}`);
    exitOverlay();
}

function medianError(samples) {
    const errors = samples.map(({ frame, target }) => {
        const p = predict(state.model, frame);
        return Math.hypot(p.x - target.x, p.y - target.y);
    }).sort((a, b) => a - b);
    return errors[Math.floor(errors.length / 2)] ?? NaN;
}

function describeCalibration(model, headData) {
    const std = (key) => {
        const values = headData.map(d => d.frame[key]);
        const mean = values.reduce((a, b) => a + b, 0) / (values.length || 1);
        return Math.sqrt(values.reduce((a, v) => a + (v - mean) ** 2, 0) / (values.length || 1));
    };
    const fit = `Calibrated. Typical fit error ${(model.fitError * 100).toFixed(1)}% of the screen.`;
    const headMoved = std('tanYaw') >= MIN_HEAD_MOTION.tanYaw || std('tx') >= MIN_HEAD_MOTION.tx;
    return headMoved ? fit : `${fit} Your head barely moved in the movement steps, so recalibrate if the dot drifts when you move.`;
}

/** Collects frames for a fixed time, leaving out blinks. */
async function collectSamples(point, duration) {
    const frames = [];
    state.onFrame = (frame) => frames.push(frame);
    calPoint.classList.add('collecting');
    await wait(duration);
    calPoint.classList.remove('collecting');
    state.onFrame = null;

    const ears = frames.map(f => f.ear).sort((a, b) => a - b);
    const minEar = ears[Math.floor(ears.length / 2)] * OPEN_EYE_RATIO;
    return frames.filter(f => f.ear >= minEar).map(frame => ({ frame, target: point }));
}

function wait(ms) {
    return new Promise(r => setTimeout(r, ms));
}

function placeCalPoint(normX, normY) {
    calPoint.style.left = `${normX * 100}%`;
    calPoint.style.top = `${normY * 100}%`;
}

async function animatePointTransition(from, to, duration) {
    const start = performance.now();

    return new Promise(resolve => {
        function frame() {
            const elapsed = performance.now() - start;
            const progress = Math.min(elapsed / duration, 1);
            const easedProgress = progress * (2 - progress);
            const currentX = from.x + (to.x - from.x) * easedProgress;
            const currentY = from.y + (to.y - from.y) * easedProgress;
            placeCalPoint(currentX, currentY);
            if (progress < 1) {
                requestAnimationFrame(frame);
            } else {
                resolve();
            }
        }
        requestAnimationFrame(frame);
    });
}

// --- Gaze Prediction ---
function predictGaze(frame) {
    if (isBlink(frame, state.model)) return; // Hold the dot; the iris landmarks jump during a blink.

    const raw = predict(state.model, frame);
    const now = performance.now() / 1000;
    const { left, top } = toViewport({
        x: state.filters.x.filter(raw.x, now),
        y: state.filters.y.filter(raw.y, now)
    });
    gazeDot.style.left = `${left}px`;
    gazeDot.style.top = `${top}px`;
}

function resetFilters() {
    state.filters.x.reset();
    state.filters.y.reset();
}

/**
 * Converts a gaze point to page coordinates. Calibration usually runs fullscreen, so its 0-1 range
 * spans the monitor, while tracking usually runs in a window that may be smaller or offset.
 */
function toViewport({ x, y }) {
    const clamp = (v, max) => Math.max(0, Math.min(max, v));
    if (!state.model.fullscreen) {
        return { left: clamp(x * window.innerWidth, window.innerWidth), top: clamp(y * window.innerHeight, window.innerHeight) };
    }
    const originX = window.screenX - (screen.availLeft ?? 0) + (window.outerWidth - window.innerWidth) / 2;
    const originY = window.screenY - (screen.availTop ?? 0) + window.outerHeight - window.innerHeight;
    return {
        left: clamp(x * screen.width - originX, window.innerWidth),
        top: clamp(y * screen.height - originY, window.innerHeight)
    };
}

// --- Local Storage ---
const MODEL_STORAGE_KEY = 'gaze_head_model_v4';

function saveCalibrationModel() {
    localStorage.setItem(MODEL_STORAGE_KEY, JSON.stringify(state.model));
}

function loadCalibrationModel() {
    try {
        const data = JSON.parse(localStorage.getItem(MODEL_STORAGE_KEY));
        if (data && data.wx && data.wy) {
            state.model = data;
            updateStatus('Loaded saved calibration.');
            return true;
        }
    } catch (e) { /* Ignore */ }
    updateStatus('No calibration found.');
    return false;
}

function clearCalibrationModel() {
    localStorage.removeItem(MODEL_STORAGE_KEY);
    state.model = null;
    updateStatus('Calibration data cleared.');
    updateUI();
}

// --- UI and Event Handlers ---
function addEventListeners() {
    toggleVideoBtn.addEventListener('click', toggleCamera);
    startCalBtn.addEventListener('click', startCalibration);
    clearCalBtn.addEventListener('click', clearCalibrationModel);
    testBtn.addEventListener('click', startAccuracyTest);
}

function updateStatus(text) {
    statusText.textContent = `Status: ${text}`;
}

function updateUI() {
    toggleVideoBtn.textContent = state.isCameraOn ? 'Stop Video' : 'Start Video';
    startCalBtn.disabled = !state.isCameraOn || state.isCalibrating;
    testBtn.disabled = !state.isCameraOn || state.isCalibrating || !state.model;

    gazeDot.style.display = (state.model && state.isCameraOn && !state.isCalibrating) ? 'block' : 'none';
}

// --- Start the App ---
window.gazeDebug = {}; // Numbers only, never images: last calibration and test samples, for analysis.
initialize();
