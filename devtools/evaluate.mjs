// Scores gaze-model.js on real recordings: fits on each calibration in captures/, then tests on its
// held-out grid dots and on the test runs recorded after it. Run with: node devtools/evaluate.mjs

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GRID_WEIGHT, extractFrame, fitModel, predict } from '../gaze-model.js';

const VIDEO_WIDTH = 1280;
const VIDEO_HEIGHT = 720;
const ROW_LENGTH = 1 + 16 + 478 * 3; // Time, pose matrix, landmarks. See capture.js.
const OPEN_EYE_RATIO = 0.7;
const captures = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'captures');

const median = (v) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)];

function loadFrames(folder) {
    const buffer = fs.readFileSync(path.join(folder, 'landmarks.bin'));
    const data = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
    const frames = [];
    for (let start = 0; start < data.length; start += ROW_LENGTH) {
        const landmarks = [];
        for (let i = 0; i < 478; i++) {
            const at = start + 17 + i * 3;
            landmarks.push({ x: data[at], y: data[at + 1], z: data[at + 2] });
        }
        const frame = extractFrame(landmarks, data.subarray(start + 1, start + 17), VIDEO_WIDTH, VIDEO_HEIGHT);
        frame.t = data[start];
        frames.push(frame);
    }
    return frames;
}

/** Splits a session into its collection windows, leaving out blinks as the app does. */
function segments(folder) {
    const { events } = JSON.parse(fs.readFileSync(path.join(folder, 'session.json')));
    const frames = loadFrames(folder);
    const starts = events.filter(e => e.type === 'collect-start');
    const ends = events.filter(e => e.type === 'collect-end');
    return starts.map((start, i) => {
        const inWindow = frames.filter(f => f.t >= start.t && f.t <= ends[i].t);
        const minEar = median(inWindow.map(f => f.ear)) * OPEN_EYE_RATIO;
        return { phase: start.phase, point: start.point, frames: inWindow.filter(f => f.ear >= minEar) };
    }).filter(s => s.frames.length > 0);
}

const names = fs.readdirSync(captures).filter(n => fs.existsSync(path.join(captures, n, 'landmarks.bin'))).sort();
const calibrations = names.filter(n => n.includes('calibration'));

const toSamples = (segs) => segs.flatMap(s => s.frames.map(frame => ({ frame, target: s.point, weight: s.phase === 'grid' ? GRID_WEIGHT : 1 })));
const dotError = (model, { point, frames }) => {
    const predictions = frames.map(f => predict(model, f));
    return Math.hypot(median(predictions.map(p => p.x)) - point.x, median(predictions.map(p => p.y)) - point.y);
};
const percent = (v) => (v * 100).toFixed(1);

const heldOutErrors = [];
const testErrors = [];
calibrations.forEach((calibration, i) => {
    const calSegments = segments(path.join(captures, calibration));

    // Half the grid dots are kept out of the fit, to score natural viewing on unseen dots.
    const held = calSegments.filter(s => s.phase === 'grid').filter((_, n) => n % 2 === 1);
    const partial = fitModel(toSamples(calSegments.filter(s => !held.includes(s))));
    const errors = held.map(s => dotError(partial, s));
    heldOutErrors.push(...errors);
    console.log(`${calibration}: held-out grid dots ${percent(median(errors))}`);

    const model = fitModel(toSamples(calSegments));
    const next = calibrations[i + 1] ?? '~';
    for (const name of names.filter(n => n.includes('test') && n > calibration && n < next)) {
        const byPose = {};
        for (const segment of segments(path.join(captures, name))) {
            if (!segment.phase.startsWith('test-')) continue;
            const error = dotError(model, segment);
            (byPose[segment.phase.slice(5)] ??= []).push(error);
            testErrors.push(error);
        }
        console.log(`  ${name}: ${Object.entries(byPose).map(([pose, e]) => `${pose} ${percent(median(e))}`).join(', ')}`);
    }
});
const worst = [...testErrors].sort((a, b) => a - b)[Math.floor(testErrors.length * 0.9)];
console.log(`Median error, % of the screen: held-out grid dots ${percent(median(heldOutErrors))}, test dots ${percent(median(testErrors))}, worst tenth of test dots ${percent(worst)} or more`);
