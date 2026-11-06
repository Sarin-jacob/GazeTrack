// gaze-model.js - Head-pose-aware gaze model. No DOM here, so it also runs under Node (see sim/).
//
// The screen point you look at depends on two things: where the eye points inside the head,
// and where the head is. So each frame is reduced to:
//   ex, ey            iris offset from the eye centre, in the head's own frame, in eye-widths.
//                     Moving, scaling or rolling the head in the image does not change these.
//   tanYaw, tanPitch  where the head is facing.
//   tx, ty, dist      where the head is, in cm from the camera.
// A ridge regression maps those to screen coordinates. Its head terms can only be learned if
// the head moved during calibration, which is why calibration has a head-movement phase.

export const RIGHT_IRIS = [469, 470, 471, 472];
export const LEFT_IRIS = [474, 475, 476, 477];
const RIGHT_EYE = { outer: 33, inner: 133, top: 159, bottom: 145 };
const LEFT_EYE = { outer: 263, inner: 362, top: 386, bottom: 374 };

const RIDGE_LAMBDA = 1e-3;
const TRIM_PASSES = 2;
const TRIM_MAD_FACTOR = 3.5; // Samples further than this many MADs from the fit are dropped.
const BLINK_EAR_RATIO = 0.7;
const DEFAULT_BLINK_EAR = 0.12;

// Landmarks are normalized per axis, so convert to pixels first or the geometry is stretched by
// the aspect ratio. Flip y and z so the axes match the pose matrix (x right, y up, z to camera).
function toCamera(p, width, height) {
    return [p.x * width, -p.y * height, -p.z * width];
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);

function centroid(landmarks, indices) {
    const c = { x: 0, y: 0, z: 0 };
    for (const i of indices) {
        c.x += landmarks[i].x / indices.length;
        c.y += landmarks[i].y / indices.length;
        c.z += landmarks[i].z / indices.length;
    }
    return c;
}

function eyeInHead(landmarks, irisIndices, eye, axes, width, height) {
    const outer = toCamera(landmarks[eye.outer], width, height);
    const inner = toCamera(landmarks[eye.inner], width, height);
    const iris = toCamera(centroid(landmarks, irisIndices), width, height);
    const top = toCamera(landmarks[eye.top], width, height);
    const bottom = toCamera(landmarks[eye.bottom], width, height);

    const eyeWidth = norm(sub(outer, inner));
    const offset = sub(iris, mid(outer, inner));
    return {
        x: (offset[0] * axes.right[0] + offset[1] * axes.right[1] + offset[2] * axes.right[2]) / eyeWidth,
        y: (offset[0] * axes.up[0] + offset[1] * axes.up[1] + offset[2] * axes.up[2]) / eyeWidth,
        openness: norm(sub(top, bottom)) / eyeWidth
    };
}

/**
 * Reduces one MediaPipe result to the values the model uses.
 * @param {Array} landmarks - Normalized face landmarks.
 * @param {ArrayLike<number>} matrix - Column-major 4x4 facial transformation matrix (cm).
 * @param {number} width - Video width in pixels.
 * @param {number} height - Video height in pixels.
 */
export function extractFrame(landmarks, matrix, width, height) {
    const axes = {
        right: [matrix[0], matrix[1], matrix[2]],
        up: [matrix[4], matrix[5], matrix[6]]
    };
    const forward = [matrix[8], matrix[9], matrix[10]];

    const right = eyeInHead(landmarks, RIGHT_IRIS, RIGHT_EYE, axes, width, height);
    const left = eyeInHead(landmarks, LEFT_IRIS, LEFT_EYE, axes, width, height);

    return {
        // The eyes are mirror images, so `x` means "towards the nose" on one and "away" on the
        // other only if measured per eye. Both are in the head's frame here, so averaging is safe.
        ex: (right.x + left.x) / 2,
        ey: (right.y + left.y) / 2,
        tanYaw: forward[0] / forward[2],
        tanPitch: forward[1] / forward[2],
        tx: matrix[12],
        ty: matrix[13],
        dist: -matrix[14],
        ear: (right.openness + left.openness) / 2
    };
}

function rawFeatures({ ex, ey, tanYaw, tanPitch, tx, ty, dist }) {
    // The same eye rotation sweeps further across the screen from further away, and the head
    // terms scale the same way. That is what the products with `dist` capture.
    return [
        ex, ey, ex * ex, ey * ey, ex * ey,
        tanYaw, tanPitch, tx, ty, dist,
        dist * ex, dist * ey, dist * tanYaw, dist * tanPitch
    ];
}

function standardize(row, mean, std) {
    return row.map((v, i) => (v - mean[i]) / std[i]);
}

function transpose(A) { return A[0].map((_, i) => A.map(row => row[i])); }
function matMul(A, B) { const m = A.length, p = A[0].length, n = B[0].length, C = Array.from({ length: m }, () => Array(n).fill(0)); for (let i = 0; i < m; i++) for (let k = 0; k < p; k++) { const v = A[i][k]; for (let j = 0; j < n; j++) C[i][j] += v * B[k][j]; } return C; }
function solveLinear(Aorig, borig) { const n = Aorig.length, M = Aorig.map(r => r.slice()), b = borig.slice(); for (let i = 0; i < n; i++) { let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r; [M[i], M[p]] = [M[p], M[i]]; [b[i], b[p]] = [b[p], b[i]]; let d = M[i][i]; if (Math.abs(d) < 1e-12) M[i][i] += 1e-8; const inv = 1 / M[i][i]; for (let j = i; j < n; j++) M[i][j] *= inv; b[i] *= inv; for (let r = 0; r < n; r++) if (r !== i) { const f = M[r][i]; for (let c = i; c < n; c++) M[r][c] -= f * M[i][c]; b[r] -= f * b[i]; } } return b; }

function fitRidge(rows, targets) {
    const At = transpose(rows);
    const AtA = matMul(At, rows);
    for (let i = 0; i < AtA.length; i++) AtA[i][i] += RIDGE_LAMBDA * rows.length;
    const rhs = matMul(At, targets.map(v => [v])).map(r => r[0]);
    return solveLinear(AtA, rhs);
}

function median(values) {
    const s = [...values].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function fitOnce(samples) {
    const raw = samples.map(s => rawFeatures(s.frame));
    const cols = transpose(raw);
    const mean = cols.map(c => c.reduce((a, b) => a + b, 0) / c.length);
    const std = cols.map((c, i) => Math.sqrt(c.reduce((a, v) => a + (v - mean[i]) ** 2, 0) / c.length) || 1);
    const rows = raw.map(r => standardize(r, mean, std));

    const xs = samples.map(s => s.target.x);
    const ys = samples.map(s => s.target.y);
    const biasX = xs.reduce((a, b) => a + b, 0) / xs.length;
    const biasY = ys.reduce((a, b) => a + b, 0) / ys.length;

    return {
        mean, std, biasX, biasY,
        wx: fitRidge(rows, xs.map(v => v - biasX)),
        wy: fitRidge(rows, ys.map(v => v - biasY))
    };
}

/**
 * Fits the model, dropping samples the fit cannot explain (the user glanced away, a bad frame).
 * Trimming on residuals rather than raw iris position keeps the deliberate head movement.
 * @param {Array<{frame: object, target: {x: number, y: number}}>} samples
 * @returns {object} A JSON-serializable model.
 */
export function fitModel(samples) {
    let kept = samples;
    let model = fitOnce(kept);
    const errorOf = (m, s) => {
        const p = predict(m, s.frame);
        return Math.hypot(p.x - s.target.x, p.y - s.target.y);
    };

    for (let pass = 0; pass < TRIM_PASSES; pass++) {
        const errors = kept.map(s => errorOf(model, s));
        const med = median(errors);
        const mad = median(errors.map(e => Math.abs(e - med)));
        const limit = med + TRIM_MAD_FACTOR * mad + 1e-6;
        kept = kept.filter((_, i) => errors[i] < limit);
        model = fitOnce(kept);
    }

    // Open eyes read lowest when looking down, so set the blink line under the low end of the data.
    const ears = kept.map(s => s.frame.ear).sort((a, b) => a - b);
    model.blinkEar = ears[Math.floor(ears.length * 0.05)] * BLINK_EAR_RATIO;
    model.fitError = median(kept.map(s => errorOf(model, s)));
    model.keptSamples = kept.length;
    return model;
}

/** Predicts a gaze point, as a fraction of the calibrated screen. */
export function predict(model, frame) {
    const f = standardize(rawFeatures(frame), model.mean, model.std);
    const dot = (w) => w.reduce((sum, v, i) => sum + v * f[i], 0);
    return { x: model.biasX + dot(model.wx), y: model.biasY + dot(model.wy) };
}

export function isBlink(frame, model) {
    return frame.ear < (model?.blinkEar ?? DEFAULT_BLINK_EAR);
}

// Where the line straight out of the face meets the camera plane, in cm. Moving and turning the
// head are both in here with the right geometry, so a few nose-pointing samples are enough.
function headRay({ tanYaw, tanPitch, tx, ty, dist }) {
    return [1, tx + dist * tanYaw, ty + dist * tanPitch];
}

/**
 * Fits the head pointer: an affine map from the head ray to the screen.
 * @param {Array<{frame: object, target: {x: number, y: number}}>} samples - Nose-pointing samples.
 */
export function fitHeadModel(samples) {
    const rows = samples.map(s => headRay(s.frame));
    const solve = (values) => {
        const At = transpose(rows);
        const AtA = matMul(At, rows);
        for (let i = 0; i < AtA.length; i++) AtA[i][i] += 1e-6;
        return solveLinear(AtA, matMul(At, values.map(v => [v])).map(r => r[0]));
    };
    return { wx: solve(samples.map(s => s.target.x)), wy: solve(samples.map(s => s.target.y)) };
}

export function predictHead(headModel, frame) {
    const f = headRay(frame);
    const dot = (w) => w.reduce((sum, v, i) => sum + v * f[i], 0);
    return { x: dot(headModel.wx), y: dot(headModel.wy) };
}

/**
 * One Euro filter: heavy smoothing while the signal is still, light smoothing while it moves fast.
 * Fixations stay steady and saccades don't lag, which a fixed smoothing factor can't do.
 */
export class OneEuroFilter {
    constructor(minCutoff = 0.8, beta = 2.5, derivativeCutoff = 1) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.derivativeCutoff = derivativeCutoff;
        this.reset();
    }

    reset() {
        this.value = null;
        this.derivative = 0;
        this.lastTime = null;
    }

    static alpha(cutoff, dt) {
        const tau = 1 / (2 * Math.PI * cutoff);
        return 1 / (1 + tau / dt);
    }

    /** @param {number} value @param {number} time - Timestamp in seconds. */
    filter(value, time) {
        if (this.value === null) {
            this.value = value;
            this.lastTime = time;
            return value;
        }
        const dt = Math.max(time - this.lastTime, 1e-3);
        const rawDerivative = (value - this.value) / dt;
        const aD = OneEuroFilter.alpha(this.derivativeCutoff, dt);
        this.derivative += aD * (rawDerivative - this.derivative);

        const cutoff = this.minCutoff + this.beta * Math.abs(this.derivative);
        this.value += OneEuroFilter.alpha(cutoff, dt) * (value - this.value);
        this.lastTime = time;
        return this.value;
    }
}

const WARP_DISTANCE = 0.14; // Fraction of the screen. Further than this, the eyes have gone elsewhere.
const WARP_SECONDS = 0.07;
const DRIFT_SECONDS = 1.5;

/**
 * The on-screen pointer.
 *   eyes: the gaze model alone.
 *   head: the head pointer alone. Steadiest, but you aim with your nose.
 *   both: the head moves the pointer directly, so it is steady and precise, while the gaze model
 *         jumps it when the eyes go somewhere else and slowly removes any head drift.
 */
export class Pointer {
    constructor() {
        this.gaze = { x: new OneEuroFilter(), y: new OneEuroFilter() };
        this.head = { x: new OneEuroFilter(1.5, 1), y: new OneEuroFilter(1.5, 1) };
        this.reset();
    }

    reset() {
        for (const f of [this.gaze.x, this.gaze.y, this.head.x, this.head.y]) f.reset();
        this.position = null;
        this.lastHead = null;
        this.lastTime = null;
    }

    update(model, frame, time, mode) {
        const rawGaze = predict(model, frame);
        const gaze = { x: this.gaze.x.filter(rawGaze.x, time), y: this.gaze.y.filter(rawGaze.y, time) };
        if (!model.head || mode === 'eyes') return (this.position = gaze);

        const rawHead = predictHead(model.head, frame);
        const head = { x: this.head.x.filter(rawHead.x, time), y: this.head.y.filter(rawHead.y, time) };
        if (mode === 'head') return (this.position = head);

        if (!this.lastHead) {
            this.position = gaze;
        } else {
            const dt = Math.max(time - this.lastTime, 1e-3);
            const moved = {
                x: this.position.x + head.x - this.lastHead.x,
                y: this.position.y + head.y - this.lastHead.y
            };
            const offset = Math.hypot(gaze.x - moved.x, gaze.y - moved.y);
            const pull = 1 - Math.exp(-dt / (offset > WARP_DISTANCE ? WARP_SECONDS : DRIFT_SECONDS));
            this.position = {
                x: moved.x + pull * (gaze.x - moved.x),
                y: moved.y + pull * (gaze.y - moved.y)
            };
        }
        this.lastHead = head;
        this.lastTime = time;
        return this.position;
    }
}
