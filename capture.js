// capture.js - Development only: records a session for devtools/capture_server.py.
// Saves the camera video and a timeline, so a session can be replayed through the model later.
// Does nothing unless the page is on localhost with that server running.

const isLocal = ['localhost', '127.0.0.1'].includes(location.hostname);
let session = null;

export async function isCaptureAvailable() {
    if (!isLocal || !window.MediaRecorder) return false;
    try {
        return (await fetch('/capture/ping')).ok;
    } catch {
        return false;
    }
}

export function startCapture(stream, label) {
    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm', videoBitsPerSecond: 6_000_000 });
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    session = { id: `${stamp}-${label}`, recorder, chunks: [], start: performance.now(), events: [], frames: [], landmarks: [] };
    recorder.ondataavailable = (e) => session?.chunks.push(e.data);
    recorder.onstart = () => { session.start = performance.now(); };
    recorder.start(1000);
}

/** Logs a timeline event, such as a target appearing. Times are ms from the start of the video. */
export function logEvent(type, data = {}) {
    if (session) session.events.push({ t: performance.now() - session.start, type, ...data });
}

export function logFrame({ shape, ...frame }, landmarks, matrix) {
    if (!session) return;
    const t = performance.now() - session.start;
    session.frames.push({ t, ...frame });

    // Raw landmarks go in a binary file, so devtools/evaluate.mjs can re-run any future model.
    const row = new Float32Array(1 + 16 + landmarks.length * 3);
    row[0] = t;
    row.set(matrix, 1);
    landmarks.forEach((p, i) => row.set([p.x, p.y, p.z], 17 + i * 3));
    session.landmarks.push(row);
}

/** Stops recording and uploads the session. Returns its id, or null if nothing was recording. */
export async function stopCapture(summary) {
    if (!session) return null;
    const { id, recorder, chunks, events, frames, landmarks } = session;
    const stopped = new Promise(resolve => { recorder.onstop = resolve; });
    recorder.stop();
    await stopped;
    session = null;

    const post = (name, body) => fetch(`/capture/${id}/${name}`, { method: 'POST', body });
    await post('video.webm', new Blob(chunks, { type: 'video/webm' }));
    await post('landmarks.bin', new Blob(landmarks));
    await post('session.json', JSON.stringify({
        id,
        screen: { width: screen.width, height: screen.height },
        viewport: { width: window.innerWidth, height: window.innerHeight },
        fullscreen: Boolean(document.fullscreenElement),
        events,
        frames,
        ...summary
    }));
    return id;
}
