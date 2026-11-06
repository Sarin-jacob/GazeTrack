// gaze-model.js - Head-pose-aware gaze model. No DOM here, so it also runs under Node (see devtools/).
//
// The screen point you look at depends on two things: where the eye points inside the head, and
// where the head is. Two models estimate that in different ways, and their average is the
// prediction. On real recordings they make different mistakes, so the average beats either one.
//   geometric: a few hand-built terms for a ray from the eye to the screen.
//   shape:     every face landmark, in the head's own frame, reduced to its main patterns.
// Both are linear and are fitted during calibration. They can only learn what head movement does
// if the head moved, which is why calibration has head-movement steps.

export const RIGHT_IRIS = [469, 470, 471, 472];
export const LEFT_IRIS = [474, 475, 476, 477];
const RIGHT_EYE = { outer: 33, inner: 133, top: 159, bottom: 145 };
const LEFT_EYE = { outer: 263, inner: 362, top: 386, bottom: 374 };
const LANDMARK_COUNT = 478;

const GEOMETRIC_RIDGE = 30;
const SHAPE_RIDGE = 10;
const SHAPE_COMPONENTS = 40;
const SUBSPACE_ITERATIONS = 3;
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

// Every landmark in the head's frame, scaled by the distance between the eyes, so moving, turning
// or leaning only changes the head values appended at the end.
function faceShape(landmarks, axes, forward, head, width, height) {
    const outerRight = toCamera(landmarks[RIGHT_EYE.outer], width, height);
    const outerLeft = toCamera(landmarks[LEFT_EYE.outer], width, height);
    const center = mid(
        mid(outerRight, toCamera(landmarks[RIGHT_EYE.inner], width, height)),
        mid(outerLeft, toCamera(landmarks[LEFT_EYE.inner], width, height))
    );
    const scale = 1 / norm(sub(outerRight, outerLeft));

    const shape = new Float32Array(LANDMARK_COUNT * 3 + head.length);
    for (let i = 0; i < LANDMARK_COUNT; i++) {
        const p = sub(toCamera(landmarks[i], width, height), center);
        shape[i * 3] = (p[0] * axes.right[0] + p[1] * axes.right[1] + p[2] * axes.right[2]) * scale;
        shape[i * 3 + 1] = (p[0] * axes.up[0] + p[1] * axes.up[1] + p[2] * axes.up[2]) * scale;
        shape[i * 3 + 2] = (p[0] * forward[0] + p[1] * forward[1] + p[2] * forward[2]) * scale;
    }
    shape.set(head, LANDMARK_COUNT * 3);
    return shape;
}

/**
 * Reduces one MediaPipe result to the values the models use.
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

    const tanYaw = forward[0] / forward[2];
    const tanPitch = forward[1] / forward[2];
    const roll = Math.atan2(matrix[1], matrix[0]);
    const tx = matrix[12];
    const ty = matrix[13];
    const dist = -matrix[14];

    return {
        ex: (right.x + left.x) / 2,
        ey: (right.y + left.y) / 2,
        tanYaw, tanPitch, roll, tx, ty, dist,
        ear: (right.openness + left.openness) / 2,
        shape: faceShape(landmarks, axes, forward, [tanYaw, tanPitch, roll, tx, ty, dist], width, height)
    };
}

// The same eye rotation sweeps further across the screen from further away, so the eye terms are
// scaled by distance. Turning and sliding the head are tied together in one ray to the screen.
function geometricFeatures({ ex, ey, tanYaw, tanPitch, tx, ty, dist }) {
    return [
        tx + dist * tanYaw, ty + dist * tanPitch, dist,
        dist * ex, dist * ey, dist * ex * ex, dist * ey * ey, dist * ex * ey
    ];
}

const shapeFeatures = (frame) => frame.shape;

function solveLinear(Aorig, borig) { const n = Aorig.length, M = Aorig.map(r => r.slice()), b = borig.slice(); for (let i = 0; i < n; i++) { let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r; [M[i], M[p]] = [M[p], M[i]]; [b[i], b[p]] = [b[p], b[i]]; let d = M[i][i]; if (Math.abs(d) < 1e-12) M[i][i] += 1e-8; const inv = 1 / M[i][i]; for (let j = i; j < n; j++) M[i][j] *= inv; b[i] *= inv; for (let r = 0; r < n; r++) if (r !== i) { const f = M[r][i]; for (let c = i; c < n; c++) M[r][c] -= f * M[i][c]; b[r] -= f * b[i]; } } return b; }

function median(values) {
    const s = [...values].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * Finds an orthonormal basis of the main directions of variation in Z (n rows of d values).
 * A basis is enough, without the individual components, because ridge regression gives the same
 * predictions in any rotation of it. Seeded, so a calibration always gives the same result.
 */
function mainSubspace(Z, n, d, k) {
    let seed = 1;
    const Q = new Float64Array(d * k);
    for (let i = 0; i < Q.length; i++) {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        Q[i] = seed / 2 ** 32 - 0.5;
    }

    const Y = new Float64Array(n * k);
    for (let pass = 0; pass < SUBSPACE_ITERATIONS; pass++) {
        Y.fill(0);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < d; j++) {
                const z = Z[i * d + j];
                for (let c = 0; c < k; c++) Y[i * k + c] += z * Q[j * k + c];
            }
        }
        Q.fill(0);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < d; j++) {
                const z = Z[i * d + j];
                for (let c = 0; c < k; c++) Q[j * k + c] += z * Y[i * k + c];
            }
        }
        for (let c = 0; c < k; c++) { // Gram-Schmidt.
            for (let e = 0; e < c; e++) {
                let overlap = 0;
                for (let j = 0; j < d; j++) overlap += Q[j * k + c] * Q[j * k + e];
                for (let j = 0; j < d; j++) Q[j * k + c] -= overlap * Q[j * k + e];
            }
            let length = 0;
            for (let j = 0; j < d; j++) length += Q[j * k + c] ** 2;
            length = Math.sqrt(length) || 1;
            for (let j = 0; j < d; j++) Q[j * k + c] /= length;
        }
    }
    return Q;
}

/**
 * Standardizes the features, optionally reduces them to their main patterns, and fits ridge
 * regression. It is all linear, so the result folds down to one weight per raw feature.
 */
function fitLinear(rows, targets, ridge, components = 0) {
    const n = rows.length;
    const d = rows[0].length;
    const mean = new Float64Array(d);
    const std = new Float64Array(d);
    for (const row of rows) for (let j = 0; j < d; j++) mean[j] += row[j] / n;
    for (const row of rows) for (let j = 0; j < d; j++) std[j] += (row[j] - mean[j]) ** 2 / n;
    for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j]) || 1;

    const Z = new Float64Array(n * d);
    rows.forEach((row, i) => {
        for (let j = 0; j < d; j++) Z[i * d + j] = (row[j] - mean[j]) / std[j];
    });

    const k = components || d;
    const Q = components ? mainSubspace(Z, n, d, k) : null;
    const scores = Q ? new Float64Array(n * k) : Z;
    if (Q) {
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < d; j++) {
                const z = Z[i * d + j];
                for (let c = 0; c < k; c++) scores[i * k + c] += z * Q[j * k + c];
            }
        }
    }

    const biasX = targets.reduce((a, t) => a + t.x, 0) / n;
    const biasY = targets.reduce((a, t) => a + t.y, 0) / n;
    const A = Array.from({ length: k }, () => new Array(k).fill(0));
    const rhsX = new Array(k).fill(0);
    const rhsY = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
        for (let a = 0; a < k; a++) {
            const v = scores[i * k + a];
            rhsX[a] += v * (targets[i].x - biasX);
            rhsY[a] += v * (targets[i].y - biasY);
            for (let b = a; b < k; b++) A[a][b] += v * scores[i * k + b];
        }
    }
    for (let a = 0; a < k; a++) {
        A[a][a] += ridge;
        for (let b = 0; b < a; b++) A[a][b] = A[b][a];
    }

    const fold = (w, bias) => {
        const raw = new Array(d);
        let offset = bias;
        for (let j = 0; j < d; j++) {
            let v = 0;
            if (Q) for (let c = 0; c < k; c++) v += Q[j * k + c] * w[c];
            else v = w[j];
            raw[j] = v / std[j];
            offset -= raw[j] * mean[j];
        }
        return { w: raw, b: offset };
    };
    return { x: fold(solveLinear(A, rhsX), biasX), y: fold(solveLinear(A, rhsY), biasY) };
}

function applyLinear(model, features) {
    let x = model.x.b;
    let y = model.y.b;
    for (let j = 0; j < features.length; j++) {
        x += model.x.w[j] * features[j];
        y += model.y.w[j] * features[j];
    }
    return { x, y };
}

/** Fits, then refits without samples it cannot explain (a glance away, a bad frame). */
function fitTrimmed(samples, featuresOf, ridge, components) {
    const rows = samples.map(s => featuresOf(s.frame));
    let kept = samples.map((_, i) => i);
    const fit = () => fitLinear(kept.map(i => rows[i]), kept.map(i => samples[i].target), ridge, components);

    let model = fit();
    for (let pass = 0; pass < TRIM_PASSES; pass++) {
        const errors = kept.map(i => {
            const p = applyLinear(model, rows[i]);
            return Math.hypot(p.x - samples[i].target.x, p.y - samples[i].target.y);
        });
        const med = median(errors);
        const mad = median(errors.map(e => Math.abs(e - med)));
        const limit = med + TRIM_MAD_FACTOR * mad + 1e-6;
        kept = kept.filter((_, i) => errors[i] < limit);
        model = fit();
    }
    return model;
}

/**
 * Fits both models.
 * @param {Array<{frame: object, target: {x: number, y: number}}>} samples
 * @returns {object} A JSON-serializable model.
 */
export function fitModel(samples) {
    const model = {
        geometric: fitTrimmed(samples, geometricFeatures, GEOMETRIC_RIDGE, 0),
        shape: fitTrimmed(samples, shapeFeatures, SHAPE_RIDGE, SHAPE_COMPONENTS)
    };

    // Open eyes read lowest when looking down, so set the blink line under the low end of the data.
    const ears = samples.map(s => s.frame.ear).sort((a, b) => a - b);
    model.blinkEar = ears[Math.floor(ears.length * 0.05)] * BLINK_EAR_RATIO;
    model.fitError = median(samples.map(s => {
        const p = predict(model, s.frame);
        return Math.hypot(p.x - s.target.x, p.y - s.target.y);
    }));
    return model;
}

/** Predicts a gaze point, as a fraction of the calibrated screen. */
export function predict(model, frame) {
    const a = applyLinear(model.geometric, geometricFeatures(frame));
    const b = applyLinear(model.shape, shapeFeatures(frame));
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
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
    const solve = (axis) => {
        const A = [0, 1, 2].map(a => [0, 1, 2].map(b => rows.reduce((sum, r) => sum + r[a] * r[b], 0) + (a === b ? 1e-6 : 0)));
        return solveLinear(A, [0, 1, 2].map(a => rows.reduce((sum, r, i) => sum + r[a] * samples[i].target[axis], 0)));
    };
    return { wx: solve('x'), wy: solve('y') };
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
