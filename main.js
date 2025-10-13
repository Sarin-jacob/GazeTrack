// main_slow.js - A more accurate, but slower, version.

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
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

// --- App State ---
let state = {
    faceLandmarker: null,
    modelLoaded: false,
    isCameraOn: false,
    isCalibrating: false,
    animationFrameId: null,
    videoStream: null,
    coeffs: { x: null, y: null },
    lastPrediction: { x: 0.5, y: 0.5 }
};

// --- Constants ---
const RIGHT_IRIS = [474, 475, 476, 477];
const LEFT_IRIS = [469, 470, 471, 472];

// --- MODIFICATIONS FOR ACCURACY ---
const SMOOTHING_ALPHA = 0.45; // Reduced smoothing for faster response to raw prediction.
const CALIBRATION_POINTS = [ // Increased from 9 to 17 points for better screen coverage
    // Center
    { x: 0.5, y: 0.5 },
    // Top Row
    { x: 0.05, y: 0.1 }, { x: 0.5, y: 0.1 }, { x: 0.95, y: 0.1 },
    // Mid Row
    { x: 0.1, y: 0.5 }, { x: 0.9, y: 0.5 },
    // Bottom Row
    { x: 0.05, y: 0.9 }, { x: 0.5, y: 0.9 }, { x: 0.95, y: 0.9 },
    // Corners
    { x: 0.05, y: 0.05 }, { x: 0.95, y: 0.05 }, { x: 0.05, y: 0.95 }, { x: 0.95, y: 0.95 },
    // Inner Points
    { x: 0.3, y: 0.3 }, { x: 0.7, y: 0.3 }, { x: 0.3, y: 0.7 }, { x: 0.7, y: 0.7 },
];
const HOLD_DURATION_MS = 2500; // Increased hold time for more data samples per point.
const TRANSITION_DURATION_MS = 1000;
// --- END MODIFICATIONS ---


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
async function predictLoop() {
    if (!state.isCameraOn || !state.modelLoaded) return;

    try {
        const results = await state.faceLandmarker.detectForVideo(video, performance.now());
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[0];
            const irisData = getIrisData(landmarks);
            drawIrisPoints(irisData.leftCenter, irisData.rightCenter);

            if (state.coeffs.x && state.coeffs.y && !state.isCalibrating) {
                predictGaze(irisData);
            }
        }
    } catch (error) {
        console.error("Prediction error:", error);
    }

    state.animationFrameId = requestAnimationFrame(predictLoop);
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

/**
 * Removes statistical outliers from an array of calibration samples.
 * This is done for each calibration point's data individually.
 * @param {Array} samples - The array of samples collected for a single point.
 * @returns {Array} - The filtered array of samples with outliers removed.
 */
function removeOutliers(samples) {
    // We need a reasonable number of samples to identify outliers
    if (samples.length < 15) {
        return samples;
    }

    // 1. Calculate the center iris position (cx, cy) for each sample
    const points = samples.map(s => {
        const cx = (s.iris.rightCenter.x + s.iris.leftCenter.x) / 2;
        const cy = (s.iris.rightCenter.y + s.iris.leftCenter.y) / 2;
        return { x: cx, y: cy };
    });

    // 2. Find the median center point. The median is more robust to outliers than the mean.
    const sortedX = points.map(p => p.x).sort((a, b) => a - b);
    const sortedY = points.map(p => p.y).sort((a, b) => a - b);
    const mid = Math.floor(sortedX.length / 2);
    const medianX = sortedX.length % 2 === 0 ? (sortedX[mid - 1] + sortedX[mid]) / 2 : sortedX[mid];
    const medianY = sortedY.length % 2 === 0 ? (sortedY[mid - 1] + sortedY[mid]) / 2 : sortedY[mid];

    // 3. Calculate the Euclidean distance of each point from the median center
    const distances = points.map(p => Math.sqrt(Math.pow(p.x - medianX, 2) + Math.pow(p.y - medianY, 2)));

    // 4. Calculate the median of these distances (a robust measure of spread)
    const sortedDistances = [...distances].sort((a, b) => a - b);
    const medianDistance = sortedDistances.length % 2 === 0 ? (sortedDistances[mid - 1] + sortedDistances[mid]) / 2 : sortedDistances[mid];

    // 5. Define a threshold. A common heuristic is that points with a distance 
    // more than 3 times the median distance are outliers.
    // We add a tiny epsilon to prevent a threshold of 0 if all points are identical.
    const threshold = medianDistance * 3.0 + 1e-6;

    // 6. Filter the original samples, keeping only the ones within the threshold
    const filteredSamples = samples.filter((_, i) => distances[i] < threshold);

    console.log(`Outlier removal: Kept ${filteredSamples.length} of ${samples.length} samples.`);

    return filteredSamples;
}

// --- Calibration Logic ---
async function startCalibration() {
    state.isCalibrating = true;
    updateUI();
    
    try {
        await document.documentElement.requestFullscreen();
    } catch (err) {
        console.warn("Could not enter fullscreen.");
    }

    calOverlay.style.display = 'flex';
    let collectedData = [];
    
    let lastPoint = { x: 0.5, y: 0.5 };
    placeCalPoint(lastPoint.x, lastPoint.y);
    calStatus.textContent = "Get ready...";
    await new Promise(r => setTimeout(r, 2000));

    for (let i = 0; i < CALIBRATION_POINTS.length; i++) {
        const point = CALIBRATION_POINTS[i];
        await animatePointTransition(lastPoint, point, TRANSITION_DURATION_MS);
        // const samples = await collectDataAtPoint(point, i + 1);
        // collectedData.push(...samples);
        //remove outliers here
        const rawSamples = await collectDataAtPoint(point, i + 1);
        const filteredSamples = removeOutliers(rawSamples);
        collectedData.push(...filteredSamples);
        lastPoint = point;
    }
    
    if (collectedData.length > 50) { // Increased threshold due to more points/time
        const features = collectedData.map(d => getFeatures(d.iris));
        const targetsX = collectedData.map(d => d.target.x);
        const targetsY = collectedData.map(d => d.target.y);
        state.coeffs.x = fitLS(features, targetsX);
        state.coeffs.y = fitLS(features, targetsY);
        saveCalibrationModel();
        updateStatus('Calibration successful!');
    } else {
        updateStatus(`Calibration failed: not enough data. (Collected ${collectedData.length})`);
    }
    
    if (document.fullscreenElement) {
        document.exitFullscreen();
    }
    calOverlay.style.display = 'none';
    state.isCalibrating = false;
    updateUI();
}

async function collectDataAtPoint(point, pointNumber) {
    calStatus.textContent = `Hold your gaze (${pointNumber}/${CALIBRATION_POINTS.length})`;
    await new Promise(r => setTimeout(r, 200));

    calPoint.classList.add('collecting');
    let samples = [];
    const endTime = performance.now() + HOLD_DURATION_MS;
    
    while (performance.now() < endTime) {
        const results = await state.faceLandmarker.detectForVideo(video, performance.now());
        if (results.faceLandmarks.length > 0) {
            samples.push({
                iris: getIrisData(results.faceLandmarks[0]),
                target: point
            });
        }
        await new Promise(r => setTimeout(r, 1000 / 30));
    }
    
    calPoint.classList.remove('collecting');
    return samples;
}

function placeCalPoint(normX, normY) {
    calPoint.style.left = `${normX * 100}%`;
    calPoint.style.top = `${normY * 100}%`;
}

async function animatePointTransition(from, to, duration) {
    const start = performance.now();
    calStatus.textContent = "Follow the point";

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
function predictGaze(irisData) {
    const f = getFeatures(irisData);
    let px = Math.max(0, Math.min(1, dot(f, state.coeffs.x)));
    let py = Math.max(0, Math.min(1, dot(f, state.coeffs.y)));

    state.lastPrediction.x = state.lastPrediction.x * (1 - SMOOTHING_ALPHA) + px * SMOOTHING_ALPHA;
    state.lastPrediction.y = state.lastPrediction.y * (1 - SMOOTHING_ALPHA) + py * SMOOTHING_ALPHA;
    
    const stage = document.getElementById('stage');
    gazeDot.style.left = `${state.lastPrediction.x * window.innerWidth}px`;
    gazeDot.style.top = `${state.lastPrediction.y * window.innerHeight}px`;
}

// --- Mathematical Model (Polynomial Regression) ---
// --- MODIFIED FOR ACCURACY: Upgraded to a 3rd order (cubic) polynomial model ---
function getFeatures({ rightCenter, leftCenter }) {
    const cx = (rightCenter.x + leftCenter.x) / 2;
    const cy = (rightCenter.y + leftCenter.y) / 2;
    // Added cubic terms for a more complex model
    return [
        1, cx, cy, 
        cx * cx, cy * cy, cx * cy,
        cx * cx * cx, cy * cy * cy, cx * cx * cy, cx * cy * cy
    ];
}
function dot(a, b) { return a.reduce((sum, val, i) => sum + val * b[i], 0); }
function transpose(A) { return A[0].map((_, i) => A.map(row => row[i])); }
function matMul(A, B) { const m = A.length, p = A[0].length, n = B[0].length, C = Array.from({ length: m }, () => Array(n).fill(0)); for (let i = 0; i < m; i++) for (let k = 0; k < p; k++) { const v = A[i][k]; for (let j = 0; j < n; j++) C[i][j] += v * B[k][j]; } return C; }
function solveLinear(Aorig, borig) { const n = Aorig.length, M = Aorig.map(r => r.slice()), b = borig.slice(); for (let i = 0; i < n; i++) { let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r; [M[i], M[p]] = [M[p], M[i]]; [b[i], b[p]] = [b[p], b[i]]; let d = M[i][i]; if (Math.abs(d) < 1e-12) M[i][i] += 1e-8; const inv = 1 / M[i][i]; for (let j = i; j < n; j++) M[i][j] *= inv; b[i] *= inv; for (let r = 0; r < n; r++) if (r !== i) { const f = M[r][i]; for (let c = i; c < n; c++) M[r][c] -= f * M[i][c]; b[r] -= f * b[i]; } } return b; }
function fitLS(Arows, targets) { const A = Arows, At = transpose(A), AtA = matMul(At, A), Aty = matMul(At, targets.map(v => [v])), rhs = Aty.map(r => r[0]); return solveLinear(AtA, rhs); }

// --- Local Storage ---
// --- MODIFIED to avoid conflict with the old model ---
const MODEL_STORAGE_KEY = 'gaze_poly_model_v3_slow';

function saveCalibrationModel() {
    localStorage.setItem(MODEL_STORAGE_KEY, JSON.stringify(state.coeffs));
}

function loadCalibrationModel() {
    try {
        const data = JSON.parse(localStorage.getItem(MODEL_STORAGE_KEY));
        if (data && data.x && data.y) {
            state.coeffs = data;
            updateStatus('Loaded saved calibration.');
            return true;
        }
    } catch (e) { /* Ignore */ }
    updateStatus('No calibration found.');
    return false;
}

function clearCalibrationModel() {
    localStorage.removeItem(MODEL_STORAGE_KEY);
    state.coeffs = { x: null, y: null };
    updateStatus('Calibration data cleared.');
    updateUI();
}

// --- UI and Event Handlers ---
function addEventListeners() {
    toggleVideoBtn.addEventListener('click', toggleCamera);
    startCalBtn.addEventListener('click', startCalibration);
    clearCalBtn.addEventListener('click', clearCalibrationModel);
}

function updateStatus(text) {
    statusText.textContent = `Status: ${text}`;
}

function updateUI() {
    toggleVideoBtn.textContent = state.isCameraOn ? 'Stop Video' : 'Start Video';
    startCalBtn.disabled = !state.isCameraOn || state.isCalibrating;
    
    const hasModel = state.coeffs.x && state.coeffs.y;
    gazeDot.style.display = (hasModel && state.isCameraOn && !state.isCalibrating) ? 'block' : 'none';
}

// --- Start the App ---
initialize();