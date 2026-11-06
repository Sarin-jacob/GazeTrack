"""Local development server: serves the app and saves capture sessions to captures/.

Run from anywhere with: python devtools/capture_server.py

Captures hold webcam video of your face, so this binds to this machine only and captures/ is
gitignored. The app only talks to this server on localhost, so the Pages site never records.
"""

import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAPTURES = ROOT / "captures"
HOST, PORT = "127.0.0.1", 8765
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
UPLOAD_PATH = re.compile(r"^/capture/([\w.-]+)/(session\.json|video\.webm|landmarks\.bin)$")


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        upload = UPLOAD_PATH.match(self.path)
        if self.path == "/capture/ping":
            self.send_response(204)
            self.end_headers()
        elif self.path == "/capture/list":
            names = sorted(p.name for p in CAPTURES.iterdir()) if CAPTURES.exists() else []
            body = json.dumps(names).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif upload:
            self.path = f"/captures/{upload[1]}/{upload[2]}"
            super().do_GET()
        elif self.path.startswith("/captures"):
            self.send_error(404)
        else:
            super().do_GET()

    def do_POST(self):
        match = UPLOAD_PATH.match(self.path)
        length = int(self.headers.get("Content-Length", 0))
        if not match or length > MAX_UPLOAD_BYTES:
            self.send_error(400)
            return

        session, name = match.groups()
        folder = CAPTURES / session
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / name, "wb") as out:
            remaining = length
            while remaining:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)

        print(f"Saved {folder / name} ({length / 1e6:.1f} MB)", flush=True)
        self.send_response(204)
        self.end_headers()


if __name__ == "__main__":
    handler = partial(Handler, directory=str(ROOT))
    print(f"Serving on http://localhost:{PORT}, captures go to {CAPTURES}", flush=True)
    ThreadingHTTPServer((HOST, PORT), handler).serve_forever()
