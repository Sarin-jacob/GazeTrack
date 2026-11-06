// main.js - Camera, calibration flow and UI. The gaze maths lives in gaze-model.js.

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
import { RIGHT_IRIS, LEFT_IRIS, extractFrame, fitModel, fitHeadModel, predict, isBlink, Pointer } from "./gaze-model.js";
import { isCaptureAvailable, startCapture, stopCapture, logEvent, logFrame } from "./capture.js";
const { FaceLandmarker, FilesetResolver } = vision;

// --- DOM Elements ---
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const gazeDot = document.getElementById('gaze-dot');
const calOverlay = document.getElementById('calibration-overlay');
const calIntro = document.getElementById('cal-intro');
const calIntroList = document.getElementById('cal-intro-list');
const calBeginBtn = document.getElementById('cal-begin');
const calPoint = document.getElementById('cal-point');
const calStatus = document.getElementById('cal-status');
const headGuide = document.getElementById('head-guide');
const headMarker = document.getElementById('head-marker');
const recBadge = document.getElementById('rec-badge');
const statusText = document.getElementById('status-text');
const toggleVideoBtn = document.getElementById('toggle-video-btn');
const startCalBtn = document.getElementById('start-cal-btn');
const clearCalBtn = document.getElementById('clear-cal-btn');
const testBtn = document.getElementById('test-btn');
const pointerModeSelect = document.getElementById('pointer-mode');

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
    pointer: new Pointer(),
    canCapture: false
};

// --- Constants ---
const CALIBRATION_POINT_COUNT = 15;
const SETTLE_DURATION_MS = 300; // The eyes need a moment to land on the point.
const HOLD_DURATION_MS = 1300;
const TRANSITION_DURATION_MS = 700;
const NOSE_SETTLE_DURATION_MS = 900; // Heads land slower than eyes.
const NOSE_HOLD_DURATION_MS = 1500;
const HEAD_STEP_TIMEOUT_MS = 12000;
const TEST_STEP_TIMEOUT_MS = 8000;
const MIN_SAMPLES = 200;
const MIN_NOSE_SAMPLES = 40;
const OPEN_EYE_RATIO = 0.7; // Frames this far under a point's typical eye openness are blinks.
const GUIDE_RADIUS_PX = 90;
const PIP_REACH = 0.45; // How close the marker must get to a ring, in guide radii.

const CALIBRATION_INTRO = [
    'Turn your sound on. A voice tells you what to do, so you never need to look away to read.',
    'First, follow the yellow dot with your eyes.',
    'Then point your nose at the dot.',
    'Last, keep your eyes on the dot and move your head. A small white marker follows your head. Steer it onto the rings around the dot.'
];
const TEST_INTRO = [
    'Turn your sound on. A voice tells you what to do.',
    'Move your head until the white marker touches the ring, then hold still.',
    'Then look at the blue dot wherever it goes, with your head staying where it is.'
];
const HEAD_TEST_INTRO = [
    'Turn your sound on.',
    'Point your nose at the blue dot wherever it goes.'
];

const NOSE_POINTS = [[0.5, 0.5], [0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]].map(([x, y]) => ({ x, y }));

// How each kind of head movement drives the marker. Values are in guide radii, mirrored so the
// marker goes the way you move. `v` is up.
const SIDES = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const HEAD_AXES = {
    turn: (f) => [-f.tanYaw / 0.25, f.tanPitch / 0.16],
    slide: (f) => [-f.tx / 6, f.ty / 4],
    lean: (f) => [0, -f.dist / 8]
};

// The model can only learn what head movement does if it sees some. Each step moves the head
// while the eyes stay on a fixed dot, so the gaze target is known while head and eyes both change.
const HEAD_STEPS = [
    { point: { x: 0.5, y: 0.5 }, axis: 'turn', pips: SIDES, speech: 'Eyes on the dot. Turn your head to touch each ring.', label: 'Turn your head' },
    { point: { x: 0.15, y: 0.15 }, axis: 'slide', pips: SIDES, speech: 'Eyes on the dot. Now slide your head without turning it.', label: 'Slide your head' },
    { point: { x: 0.85, y: 0.15 }, axis: 'turn', pips: SIDES, speech: 'Turn your head again.', label: 'Turn your head' },
    { point: { x: 0.85, y: 0.85 }, axis: 'lean', pips: [[0, -1], [0, 1]], speech: 'Lean in, then lean back.', label: 'Lean in and back' },
    { point: { x: 0.15, y: 0.85 }, axis: 'slide', pips: SIDES, speech: 'Last one. Slide your head again.', label: 'Slide your head' }
];

// Accuracy test: look at known points with the head held in different places. The points are
// off the calibration grid, so this measures the model on data it was not fitted to.
const TEST_SETTLE_DURATION_MS = 500;
const TEST_HOLD_DURATION_MS = 1000;
const TEST_POINTS = [{ x: 0.3, y: 0.3 }, { x: 0.7, y: 0.3 }, { x: 0.7, y: 0.7 }, { x: 0.3, y: 0.7 }];
const TEST_POSES = [
    { name: 'still', speech: 'Sit normally and follow the dot.' },
    { name: 'left', axis: 'slide', pips: [[-1, 0]], speech: 'Slide your head to the left, to the ring.' },
    { name: 'right', axis: 'slide', pips: [[1, 0]], speech: 'Slide your head to the right, to the ring.' },
    { name: 'closer', axis: 'lean', pips: [[0, 1]], speech: 'Lean in to the ring.' },
    { name: 'further', axis: 'lean', pips: [[0, -1]], speech: 'Lean back to the ring.' },
    { name: 'turned', axis: 'turn', pips: [[1, 0]], speech: 'Turn your head to the ring.' }
];
const HEAD_TEST_POSES = [{ name: 'nose', speech: 'Point your nose at the dot.' }];

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
    state.canCapture = await isCaptureAvailable();
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

    // Hold the pointer through blinks; the iris landmarks jump while the eye closes.
    if (state.model && !isBlink(frame, state.model)) {
        frame.pointer = state.pointer.update(state.model, frame, performance.now() / 1000, pointerModeSelect.value);
        if (!state.isCalibrating) placeGazeDot(frame.pointer);
    }

    logFrame(frame);
    state.onFrame?.(frame);
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

// --- Overlay ---
// Everything to read goes on the intro screen, before any samples are taken. After that,
// instructions are spoken, because reading means looking away from the dot being measured.
async function enterOverlay(introLines, label) {
    state.isCalibrating = true;
    updateUI();

    try {
        await document.documentElement.requestFullscreen();
    } catch (err) {
        console.warn("Could not enter fullscreen.");
    }

    calOverlay.style.display = 'flex';
    calPoint.style.display = 'none';
    calStatus.style.display = 'none';
    calIntroList.replaceChildren(...introLines.map(line => {
        const item = document.createElement('li');
        item.textContent = line;
        return item;
    }));
    calIntro.style.display = 'block';
    calBeginBtn.focus();
    await new Promise(resolve => calBeginBtn.addEventListener('click', resolve, { once: true }));
    calIntro.style.display = 'none';
    calPoint.style.display = 'block';
    calStatus.style.display = 'block';

    if (state.canCapture) {
        startCapture(state.videoStream, label);
        recBadge.style.display = 'block';
    }
}

function exitOverlay() {
    if (document.fullscreenElement) {
        document.exitFullscreen();
    }
    recBadge.style.display = 'none';
    calOverlay.style.display = 'none';
    state.isCalibrating = false;
    updateUI();
}

/** Speaks an instruction, and shows a short label beside the dot. Resolves when it has finished. */
function say(speech, label = '') {
    calStatus.textContent = label;
    logEvent('say', { speech });
    if (!('speechSynthesis' in window)) return wait(1500);

    return new Promise(resolve => {
        const fallback = setTimeout(resolve, 1500 + speech.length * 90); // Some systems never fire onend.
        const utterance = new SpeechSynthesisUtterance(speech);
        utterance.onend = utterance.onerror = () => {
            clearTimeout(fallback);
            resolve();
        };
        speechSynthesis.cancel();
        speechSynthesis.speak(utterance);
    });
}

// --- Calibration Logic ---
async function startCalibration() {
    await enterOverlay(CALIBRATION_INTRO, 'calibration');
    let lastPoint = { x: 0.5, y: 0.5 };
    placeCalPoint(lastPoint.x, lastPoint.y);
    await say('Follow the dot with your eyes.', 'Follow the dot');

    // Generated here, not at load, so the grid matches the fullscreen aspect ratio.
    const points = generateCalibrationPoints(CALIBRATION_POINT_COUNT);
    const gridData = [];
    for (let i = 0; i < points.length; i++) {
        const point = points[i];
        calStatus.textContent = `${i + 1} / ${points.length}`;
        await animatePointTransition(lastPoint, point, TRANSITION_DURATION_MS);
        await wait(SETTLE_DURATION_MS);
        gridData.push(...await collectSamples(point, HOLD_DURATION_MS, 'grid'));
        lastPoint = point;
    }

    await say('Now point your nose at the dot.', 'Point your nose');
    const noseData = [];
    for (const point of NOSE_POINTS) {
        await animatePointTransition(lastPoint, point, TRANSITION_DURATION_MS);
        await wait(NOSE_SETTLE_DURATION_MS);
        noseData.push(...await collectSamples(point, NOSE_HOLD_DURATION_MS, 'nose'));
        lastPoint = point;
    }

    const rest = restPose(gridData);
    const headData = [];
    let stepsFinished = 0;
    for (const step of HEAD_STEPS) {
        await animatePointTransition(lastPoint, step.point, TRANSITION_DURATION_MS);
        await say(step.speech, step.label);
        const { samples, finished } = await steerHead(step, rest, HEAD_STEP_TIMEOUT_MS);
        headData.push(...samples);
        if (finished) stepsFinished++;
        lastPoint = step.point;
    }

    const allData = [...gridData, ...noseData, ...headData];
    if (allData.length >= MIN_SAMPLES) {
        state.model = fitModel(allData);
        state.model.head = noseData.length >= MIN_NOSE_SAMPLES ? fitHeadModel(noseData) : null;
        state.model.fullscreen = Boolean(document.fullscreenElement);
        state.model.rest = rest;
        state.pointer.reset();
        saveCalibrationModel();
        updateStatus(`Calibrated. Typical fit error ${(state.model.fitError * 100).toFixed(1)}% of the screen. Head steps finished: ${stepsFinished}/${HEAD_STEPS.length}.`);
    } else {
        updateStatus(`Calibration failed: not enough data. (Collected ${allData.length})`);
    }

    await say('All done.');
    await finishCapture({ model: state.model, stepsFinished });
    exitOverlay();
}

/** Stops any recording and adds where it went to the status line. */
async function finishCapture(summary) {
    const id = await stopCapture(summary);
    if (id) statusText.textContent += ` Saved capture ${id}.`;
}

/** The natural seated head position: the median of each value over the given samples. */
function restPose(samples) {
    const median = (key) => {
        const values = samples.map(s => s.frame[key]).sort((a, b) => a - b);
        return values[Math.floor(values.length / 2)];
    };
    return { tanYaw: median('tanYaw'), tanPitch: median('tanPitch'), tx: median('tx'), ty: median('ty'), dist: median('dist') };
}

/**
 * Shows the marker and rings around the dot and runs until every ring is touched, or time runs
 * out. The rings are close enough to the dot that the eyes can stay put, and they are centred on
 * the rest pose, not the head's position right now, so every ring stays reachable.
 */
async function steerHead({ point, axis, pips }, rest, timeout) {
    const read = HEAD_AXES[axis];
    const origin = read(rest);
    const rings = pips.map(([u, v]) => {
        const ring = document.createElement('div');
        ring.className = 'head-pip';
        ring.style.left = `calc(50% + ${u * GUIDE_RADIUS_PX}px)`;
        ring.style.top = `calc(50% + ${-v * GUIDE_RADIUS_PX}px)`;
        headGuide.appendChild(ring);
        return { u, v, ring, hit: false };
    });
    headGuide.style.display = 'block';

    const isDone = (frame) => {
        const [rawU, rawV] = read(frame);
        const u = rawU - origin[0];
        const v = rawV - origin[1];
        headMarker.style.left = `calc(50% + ${Math.max(-1.3, Math.min(1.3, u)) * GUIDE_RADIUS_PX}px)`;
        headMarker.style.top = `calc(50% + ${-Math.max(-1.3, Math.min(1.3, v)) * GUIDE_RADIUS_PX}px)`;
        for (const r of rings) {
            if (!r.hit && Math.hypot(u - r.u, v - r.v) < PIP_REACH) {
                r.hit = true;
                r.ring.classList.add('hit');
            }
        }
        return rings.every(r => r.hit);
    };

    const samples = await collectSamples(point, timeout, `head-${axis}`, isDone);
    rings.forEach(r => r.ring.remove());
    headGuide.style.display = 'none';
    return { samples, finished: rings.every(r => r.hit) };
}

/** Collects frames until the time is up or isDone says so, leaving out blinks. */
async function collectSamples(point, duration, phase, isDone = () => false) {
    const frames = [];
    let done = false;
    state.onFrame = (frame) => {
        frames.push(frame);
        done = isDone(frame);
    };
    logEvent('collect-start', { point, phase });
    calPoint.classList.add('collecting');

    const endTime = performance.now() + duration;
    while (!done && performance.now() < endTime) await wait(30);

    calPoint.classList.remove('collecting');
    logEvent('collect-end', { point, phase });
    state.onFrame = null;

    const ears = frames.map(f => f.ear).sort((a, b) => a - b);
    const minEar = ears[Math.floor(ears.length / 2)] * OPEN_EYE_RATIO;
    return frames.filter(f => f.ear >= minEar).map(frame => ({ frame, target: point }));
}

function wait(ms) {
    return new Promise(r => setTimeout(r, ms));
}

function placeCalPoint(normX, normY) {
    for (const el of [calPoint, headGuide]) {
        el.style.left = `${normX * 100}%`;
        el.style.top = `${normY * 100}%`;
    }

    // The label sits just past the head guide, on whichever side has room.
    const labelHalfWidth = 150;
    const x = Math.max(labelHalfWidth, Math.min(window.innerWidth - labelHalfWidth, normX * window.innerWidth));
    const offset = GUIDE_RADIUS_PX + 40;
    calStatus.style.left = `${x}px`;
    calStatus.style.top = `${normY * window.innerHeight + (normY > 0.7 ? -offset : offset)}px`;
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

// --- Accuracy Test ---
async function startAccuracyTest() {
    const mode = pointerModeSelect.value;
    const headOnly = mode === 'head';
    await enterOverlay(headOnly ? HEAD_TEST_INTRO : TEST_INTRO, `test-${mode}`);
    calPoint.classList.add('test');
    let lastPoint = { x: 0.5, y: 0.5 };
    placeCalPoint(lastPoint.x, lastPoint.y);

    const results = [];
    for (const pose of headOnly ? HEAD_TEST_POSES : TEST_POSES) {
        await animatePointTransition(lastPoint, { x: 0.5, y: 0.5 }, TRANSITION_DURATION_MS);
        lastPoint = { x: 0.5, y: 0.5 };
        await say(pose.speech, pose.name);
        if (pose.axis) {
            await steerHead({ point: lastPoint, ...pose }, state.model.rest, TEST_STEP_TIMEOUT_MS);
            await say('Hold that, and follow the dot.', 'Hold still');
        }

        const samples = [];
        for (const point of TEST_POINTS) {
            await animatePointTransition(lastPoint, point, TRANSITION_DURATION_MS);
            await wait(TEST_SETTLE_DURATION_MS);
            samples.push(...await collectSamples(point, TEST_HOLD_DURATION_MS, `test-${pose.name}`));
            lastPoint = point;
        }
        results.push({ pose: pose.name, error: medianError(samples) });
    }

    calPoint.classList.remove('test');
    updateStatus(`Median error (${pointerModeSelect.selectedOptions[0].text}), % of screen: ${results.map(r => `${r.pose} ${(r.error * 100).toFixed(1)}`).join(', ')}.`);
    await say('All done.');
    await finishCapture({ model: state.model, mode, results });
    exitOverlay();
}

function medianError(samples) {
    const errors = samples.map(({ frame, target }) => {
        const p = frame.pointer ?? predict(state.model, frame);
        return Math.hypot(p.x - target.x, p.y - target.y);
    }).sort((a, b) => a - b);
    return errors[Math.floor(errors.length / 2)] ?? NaN;
}

// --- Gaze Prediction ---
function placeGazeDot(point) {
    const { left, top } = toViewport(point);
    gazeDot.style.left = `${left}px`;
    gazeDot.style.top = `${top}px`;
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
const MODEL_STORAGE_KEY = 'gaze_head_model_v5';

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
    pointerModeSelect.addEventListener('change', () => state.pointer.reset());
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
initialize();
