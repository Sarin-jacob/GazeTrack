// Scores gaze-model.js on real recordings: fits on the newest calibration in captures/ and tests
// on the runs recorded after it. Run with: node devtools/evaluate.mjs

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { extractFrame, fitModel, fitHeadModel, predict } from '../gaze-model.js';

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
const calibration = names.filter(n => n.includes('calibration')).at(-1);
const calSegments = segments(path.join(captures, calibration));
const samples = calSegments.flatMap(s => s.frames.map(frame => ({ frame, target: s.point })));

const started = performance.now();
const model = fitModel(samples);
console.log(`Fitted on ${calibration}: ${samples.length} samples in ${((performance.now() - started) / 1000).toFixed(1)} s, stored size ${(JSON.stringify(model).length / 1024).toFixed(0)} KB`);
model.head = fitHeadModel(calSegments.filter(s => s.phase === 'nose').flatMap(s => s.frames.map(frame => ({ frame, target: s.point }))));

const allErrors = [];
for (const name of names.filter(n => n.includes('test') && n > calibration)) {
    const byPose = {};
    for (const { phase, point, frames } of segments(path.join(captures, name))) {
        if (!phase.startsWith('test-')) continue;
        const predictions = frames.map(f => predict(model, f));
        const error = Math.hypot(median(predictions.map(p => p.x)) - point.x, median(predictions.map(p => p.y)) - point.y);
        (byPose[phase.slice(5)] ??= []).push(error);
        allErrors.push(error);
    }
    const line = Object.entries(byPose).map(([pose, e]) => `${pose} ${(median(e) * 100).toFixed(1)}`).join(', ');
    console.log(`${name}: ${line}`);
}
console.log(`Median error across all test dots: ${(median(allErrors) * 100).toFixed(1)}% of the screen`);
