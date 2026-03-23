#!/usr/bin/env python3
"""Simple host-side MJPEG webcam streamer for macOS -> devcontainer workflows.

Run on host OS:
  python3 scripts/host_webcam_stream.py --port 8080

Then inside the devcontainer, set:
  export CAMERA_SOURCE=http://host.docker.internal:8080/video.mjpg
"""

import argparse
import sys
import threading
import time
from http import server

try:
    import cv2
except ModuleNotFoundError:
    print(
        "Missing dependency: cv2 (OpenCV) is not installed for this host Python.\n"
        "Install it on host with one of these commands:\n"
        "  python3 -m pip install opencv-python\n"
        "  conda install -c conda-forge opencv\n"
        "Then run this script again.",
        file=sys.stderr,
    )
    raise SystemExit(1)


class FrameStore:
    """Thread-safe latest-frame storage."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None

    def set_frame(self, jpeg_bytes):
        with self._lock:
            self._jpeg = jpeg_bytes

    def get_frame(self):
        with self._lock:
            return self._jpeg


FRAME_STORE = FrameStore()


def parse_camera_index(raw_value):
    """Parse camera index argument.

    Returns:
      int camera index, or None when auto-probing should be used.
    """
    value = str(raw_value).strip().lower()
    if value == 'auto':
        return None

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid --camera-index '{raw_value}'. Use an integer or 'auto'.") from exc


def open_webcam(camera_index, width, height, fps):
    """Open webcam with backend and index fallback (friendly for macOS)."""
    backends = []
    if hasattr(cv2, 'CAP_AVFOUNDATION'):
        backends.append(('CAP_AVFOUNDATION', cv2.CAP_AVFOUNDATION))
    backends.append(('CAP_ANY', cv2.CAP_ANY))

    candidate_indices = [camera_index] if camera_index is not None else [0, 1, 2, 3]

    for index in candidate_indices:
        for backend_name, backend_value in backends:
            cap = cv2.VideoCapture(index, backend_value)

            if width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps > 0:
                cap.set(cv2.CAP_PROP_FPS, fps)

            if cap.isOpened():
                # Read one frame to ensure the camera is actually delivering data.
                ok, _ = cap.read()
                if ok:
                    return cap, index, backend_name

            cap.release()

    raise RuntimeError(
        "Could not open a webcam from candidate indices "
        f"{candidate_indices}.\n"
        "Try one of:\n"
        "  1) Grant camera permission to your terminal app (and VS Code if used) in\n"
        "     System Settings -> Privacy & Security -> Camera\n"
        "  2) Close apps currently using the camera (Zoom, Meet, Teams, etc.)\n"
        "  3) Retry with explicit index, e.g. --camera-index 1\n"
        "  4) Retry with --camera-index auto"
    )


class WebcamRequestHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(
                b"""<html><body>
                <h3>Host Webcam Stream</h3>
                <img src='/video.mjpg' style='max-width: 100%;' />
                </body></html>"""
            )
            return

        if self.path != '/video.mjpg':
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header('Age', '0')
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        while True:
            frame = FRAME_STORE.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            try:
                part_header = (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    + f'Content-Length: {len(frame)}\r\n\r\n'.encode('ascii')
                )
                self.wfile.write(part_header)
                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
                self.wfile.flush()
                time.sleep(0.005)
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, fmt, *args):
        # Keep server logs quiet by default.
        return


def webcam_capture_loop(cap):
    if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        # Mirror preview orientation to match user expectation.
        frame = cv2.flip(frame, 1)

        success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not success:
            continue

        FRAME_STORE.set_frame(encoded.tobytes())


def main():
    parser = argparse.ArgumentParser(description='Host webcam MJPEG streamer')
    parser.add_argument('--host', default='0.0.0.0', help='Bind host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Bind port (default: 8080)')
    parser.add_argument('--camera-index', default='auto', help="Webcam index or 'auto' (default: auto)")
    parser.add_argument('--width', type=int, default=640, help='Capture width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Capture height (default: 480)')
    parser.add_argument('--fps', type=int, default=20, help='Capture FPS target (default: 20)')
    args = parser.parse_args()

    camera_index = parse_camera_index(args.camera_index)
    cap, selected_index, backend_name = open_webcam(camera_index, args.width, args.height, args.fps)

    print(
        f'Webcam opened: index={selected_index}, backend={backend_name}, '
        f'width={args.width}, height={args.height}, fps={args.fps}'
    )

    capture_thread = threading.Thread(
        target=webcam_capture_loop,
        args=(cap,),
        daemon=True,
    )
    capture_thread.start()

    stream_url = f'http://{args.host}:{args.port}/video.mjpg'
    print('Host webcam stream is running')
    print(f'Local preview: {stream_url}')
    print(f'Devcontainer CAMERA_SOURCE: http://host.docker.internal:{args.port}/video.mjpg')

    httpd = server.ThreadingHTTPServer((args.host, args.port), WebcamRequestHandler)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        cap.release()


if __name__ == '__main__':
    main()
