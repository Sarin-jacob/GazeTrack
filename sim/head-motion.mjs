// Compares the old iris-only model with gaze-model.js on a simulated head and eyes.
// Run with: node sim/head-motion.mjs
//
// This checks the geometry and the fit, nothing more. Real accuracy depends on how noisy
// MediaPipe's landmarks are on a real webcam, which the noise values below only approximate.

import { extractFrame, fitModel, fitHeadModel, predict, Pointer, RIGHT_IRIS, LEFT_IRIS } from '../gaze-model.js';

// Seeded so runs are comparable.
let seed = 12345;
const rand = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
const gauss = (s = 1) => s * Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
const uniform = (a) => (rand() * 2 - 1) * a;

const W = 1280, H = 720;
const FOCAL = (H / 2) / Math.tan((63 / 2) * Math.PI / 180); // MediaPipe assumes a 63 degree vertical FOV.
const SCREEN = { w: 34, h: 19 }; // cm, top edge just below the camera.
const EYEBALL_R = 1.2;
const NOISE = { irisPx: 0.5, cornerPx: 0.3, poseDeg: 0.4, poseCm: 0.15, zScale: 0.7 };

function rotation(yaw, pitch, roll) {
    const [cy, sy, cp, sp, cr, sr] = [Math.cos(yaw), Math.sin(yaw), Math.cos(pitch), Math.sin(pitch), Math.cos(roll), Math.sin(roll)];
    const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
    const Rx = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]];
    const Rz = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]];
    const mul = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
    return mul(Ry, mul(Rx, Rz));
}
const apply = (R, v) => R.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

/** Renders landmarks and the pose matrix for a head pose looking at a screen point. */
function render(head, target) {
    const R = rotation(head.yaw, head.pitch, head.roll);
    const t = [head.x, head.y, -head.z];
    const world = (p) => add(apply(R, p), t);
    // Camera looks down -z. Landmark z mimics MediaPipe: relative depth, in image-width units.
    const project = (p, noise) => ({
        x: (W / 2 + FOCAL * p[0] / -p[2] + gauss(noise)) / W,
        y: (H / 2 - FOCAL * p[1] / -p[2] + gauss(noise)) / H,
        z: -(p[2] + head.z) * (FOCAL / head.z) * NOISE.zScale / W
    });

    const screenPoint = [(target.x - 0.5) * SCREEN.w, -1 - target.y * SCREEN.h, 0];
    const lm = [];
    const eye = (side, ids, irisIds) => {
        const c = [side * 3.1, 3.2, 3.5]; // Eyeball centre in the head frame, cm.
        const corners = [[side * 4.6, 3.1, 4.3], [side * 1.7, 3.1, 4.4]]; // outer, inner
        ids.outerInner.forEach((id, i) => { lm[id] = project(world(corners[i]), NOISE.cornerPx); });
        const eyeWorld = world(c);
        const d = [screenPoint[0] - eyeWorld[0], screenPoint[1] - eyeWorld[1], screenPoint[2] - eyeWorld[2]];
        const len = Math.hypot(...d);
        const iris = add(eyeWorld, d.map(v => EYEBALL_R * v / len));
        const irisNoise = [gauss(NOISE.irisPx), gauss(NOISE.irisPx)];
        const p = project(iris, 0);
        irisIds.forEach(id => { lm[id] = { x: p.x + irisNoise[0] / W, y: p.y + irisNoise[1] / H, z: p.z }; });
        lm[ids.topBottom[0]] = project(world([side * 3.1, 3.7, 4.5]), NOISE.cornerPx);
        lm[ids.topBottom[1]] = project(world([side * 3.1, 2.7, 4.5]), NOISE.cornerPx);
    };
    eye(-1, { outerInner: [33, 133], topBottom: [159, 145] }, RIGHT_IRIS);
    eye(1, { outerInner: [263, 362], topBottom: [386, 374] }, LEFT_IRIS);

    const n = NOISE.poseDeg * Math.PI / 180;
    const Rn = rotation(head.yaw + gauss(n), head.pitch + gauss(n), head.roll + gauss(n));
    const matrix = [
        Rn[0][0], Rn[1][0], Rn[2][0], 0,
        Rn[0][1], Rn[1][1], Rn[2][1], 0,
        Rn[0][2], Rn[1][2], Rn[2][2], 0,
        t[0] + gauss(NOISE.poseCm), t[1] + gauss(NOISE.poseCm), t[2] + gauss(NOISE.poseCm * 2), 1
    ];
    return { lm, matrix };
}

const REST = { x: 0, y: -3, z: 55, yaw: 0, pitch: -0.12, roll: 0 };
const sway = (s = 1) => ({
    x: REST.x + gauss(0.3 * s), y: REST.y + gauss(0.3 * s), z: REST.z + gauss(0.4 * s),
    yaw: gauss(0.01 * s), pitch: REST.pitch + gauss(0.01 * s), roll: gauss(0.01 * s)
});

// Old model, as it was in main.js: cubic polynomial of the raw iris position.
function oldFeatures(lm) {
    const cx = RIGHT_IRIS.concat(LEFT_IRIS).reduce((s, i) => s + lm[i].x, 0) / 8;
    const cy = RIGHT_IRIS.concat(LEFT_IRIS).reduce((s, i) => s + lm[i].y, 0) / 8;
    return [1, cx, cy, cx * cx, cy * cy, cx * cy, cx ** 3, cy ** 3, cx * cx * cy, cx * cy * cy];
}
function fitOld(samples) {
    const n = 10, A = Array.from({ length: n }, () => Array(n).fill(0)), bx = Array(n).fill(0), by = Array(n).fill(0);
    for (const { f, target } of samples) for (let i = 0; i < n; i++) {
        bx[i] += f[i] * target.x; by[i] += f[i] * target.y;
        for (let j = 0; j < n; j++) A[i][j] += f[i] * f[j];
    }
    const solve = (Ao, bo) => {
        const M = Ao.map((r, i) => [...r, bo[i]]);
        for (let i = 0; i < n; i++) {
            let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
            [M[i], M[p]] = [M[p], M[i]];
            if (Math.abs(M[i][i]) < 1e-14) M[i][i] = 1e-14;
            for (let r = 0; r < n; r++) if (r !== i) { const k = M[r][i] / M[i][i]; for (let c = i; c <= n; c++) M[r][c] -= k * M[i][c]; }
        }
        return M.map((r, i) => r[n] / r[i]);
    };
    return { x: solve(A, bx), y: solve(A, by) };
}
const clamp = (v) => Math.max(0, Math.min(1, v));
const dotp = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

// Calibration grid: same 5x3 layout generateCalibrationPoints gives at 16:9.
const grid = [];
for (let j = 0; j < 3; j++) for (let i = 0; i < 5; i++) grid.push({ x: 0.05 + i * 0.225, y: 0.05 + j * 0.45 });

// Head phase: one movement per anchor, a sine sweep while the eyes stay on the dot.
const MOVES = [
    { point: { x: 0.5, y: 0.5 }, key: 'yaw', amp: 0.26 },
    { point: { x: 0.1, y: 0.1 }, key: 'pitch', amp: 0.18 },
    { point: { x: 0.9, y: 0.1 }, key: 'x', amp: 7 },
    { point: { x: 0.9, y: 0.9 }, key: 'z', amp: 10 },
    { point: { x: 0.1, y: 0.9 }, key: 'y', amp: 5 }
];

// Real sway is slow, and heads drift towards what the eyes look at. Both make head motion hard
// to tell from gaze, so per-frame random sway would flatter the fit. Hold an offset per point,
// add the head-follows-target bias, and keep only a little per-frame jitter.
function posedFor(target) {
    const base = sway();
    base.yaw += 0.1 * (target.x - 0.5);
    base.pitch -= 0.06 * (target.y - 0.5);
    base.x += 0.8 * (target.x - 0.5);
    return base;
}
const jitter = (head) => {
    const j = sway(0.15);
    return {
        x: head.x + j.x - REST.x, y: head.y + j.y - REST.y, z: head.z + j.z - REST.z,
        yaw: head.yaw + j.yaw, pitch: head.pitch + j.pitch - REST.pitch, roll: head.roll + j.roll
    };
};

// `readTop` reproduces reading prompts at the top of the screen: for part of each head step the
// eyes are on the text while the sample is still labelled with the dot.
function collect(withHeadPhase, readTop = 0) {
    const samples = [];
    for (const target of grid) {
        const base = posedFor(target);
        for (let i = 0; i < 36; i++) samples.push({ head: jitter(base), target, gaze: target });
    }
    if (withHeadPhase) for (const m of MOVES) {
        const base = posedFor(m.point);
        for (let i = 0; i < 135; i++) {
            const head = jitter(base);
            head[m.key] += m.amp * Math.sin(i / 135 * 4 * Math.PI);
            samples.push({ head, target: m.point, gaze: i / 135 < readTop ? { x: 0.5, y: 0.2 } : m.point });
        }
    }
    return samples;
}

// Nose-pointing step: the face aims at each point, and the eyes are there too.
function aimAt(target) {
    const head = jitter(REST);
    const sx = (target.x - 0.5) * SCREEN.w, sy = -1 - target.y * SCREEN.h;
    head.yaw = Math.atan((sx - head.x) / head.z) + gauss(0.012);
    head.pitch = -Math.atan((sy - head.y) / head.z) + gauss(0.012);
    return head;
}
const NOSE_POINTS = [[0.5, 0.5], [0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]].map(([x, y]) => ({ x, y }));
function collectNose() {
    const samples = [];
    for (const target of NOSE_POINTS) for (let i = 0; i < 45; i++) samples.push({ head: aimAt(target), target, gaze: target });
    return samples;
}

const toFrame = (r) => extractFrame(r.lm, r.matrix, W, H);

const labelled = (samples) => samples.map(s => ({ frame: toFrame(render(s.head, s.gaze)), target: s.target }));
const oldModel = fitOld(collect(false).map(s => ({ f: oldFeatures(render(s.head, s.gaze).lm), target: s.target })));
const eyeOnly = fitModel(labelled(collect(false)));
const misread = fitModel(labelled(collect(true, 0.4)));
const noseSamples = labelled(collectNose());
const newModel = fitModel([...labelled(collect(true)), ...noseSamples]);
newModel.head = fitHeadModel(noseSamples);

const scenarios = {
    'still head': () => sway(0.3),
    'slide +/-3 cm': () => ({ ...sway(), x: uniform(3), y: REST.y + uniform(3) }),
    'slide +/-8 cm': () => ({ ...sway(), x: uniform(8), y: REST.y + uniform(5) }),
    'lean +/-10 cm': () => ({ ...sway(), z: REST.z + uniform(10) }),
    'turn +/-10 deg': () => ({ ...sway(), yaw: uniform(0.175), pitch: REST.pitch + uniform(0.12) }),
    'roll +/-10 deg': () => ({ ...sway(), roll: uniform(0.175) }),
    'all at once': () => ({
        x: uniform(6), y: REST.y + uniform(4), z: REST.z + uniform(8),
        yaw: uniform(0.17), pitch: REST.pitch + uniform(0.12), roll: uniform(0.09)
    })
};

const medianOf = (v) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)];
const pct = (v) => `${(v * 100).toFixed(1).padStart(5)}%`;

console.log('Median error as a fraction of the screen, single frames, no smoothing\n');
console.log('scenario           old     no head step  read prompts   new');
for (const [name, gen] of Object.entries(scenarios)) {
    const errs = [[], [], [], []];
    for (let i = 0; i < 1500; i++) {
        const target = { x: rand(), y: rand() };
        const r = render(gen(), target);
        const f = oldFeatures(r.lm);
        const frame = toFrame(r);
        const o = { x: clamp(dotp(f, oldModel.x)), y: clamp(dotp(f, oldModel.y)) };
        [o, predict(eyeOnly, frame), predict(misread, frame), predict(newModel, frame)].forEach((p, k) => {
            errs[k].push(Math.hypot(clamp(p.x) - target.x, clamp(p.y) - target.y));
        });
    }
    console.log(`${name.padEnd(16)} ${errs.map(e => pct(medianOf(e))).join('        ')}`);
}

// Pointer behaviour over time: 30 fps, a new target every 2 s, head drifting slowly.
console.log('\nPointer over time: error while fixating, frame-to-frame jitter, time to reach a new target\n');
for (const mode of ['eyes', 'head', 'both']) {
    const pointer = new Pointer();
    const errors = [], steps = [], arrive = [];
    let target = { x: 0.5, y: 0.5 }, last = null, arrived = null;
    for (let n = 0; n < 30 * 120; n++) {
        const t = n / 30, phase = n % 60;
        if (phase === 0) { target = { x: 0.1 + rand() * 0.8, y: 0.1 + rand() * 0.8 }; arrived = null; }
        const head = mode === 'head' ? aimAt(target) : jitter(posedFor(target));
        if (mode !== 'head') { head.x += 2 * Math.sin(t / 3); head.yaw += 0.04 * Math.sin(t / 5); }
        const p = pointer.update(newModel, toFrame(render(head, target)), t, mode);
        const e = Math.hypot(p.x - target.x, p.y - target.y);
        if (arrived === null && e < 0.12) { arrived = phase / 30; arrive.push(arrived); }
        if (phase >= 30) {
            errors.push(e);
            if (last) steps.push(Math.hypot(p.x - last.x, p.y - last.y));
        }
        last = { ...p };
    }
    const mean = (v) => v.reduce((a, b) => a + b, 0) / (v.length || 1);
    console.log(`${mode.padEnd(5)} error ${pct(medianOf(errors))}   jitter ${(mean(steps) * 100).toFixed(2)}%   arrive ${mean(arrive).toFixed(2)} s (${arrive.length}/60 targets)`);
}
