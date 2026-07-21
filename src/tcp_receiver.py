import json
import os
import socket


def resolve_tcp_host():
    return os.getenv("TCP_HOST", "127.0.0.1").strip() or "127.0.0.1"


def resolve_tcp_port():
    raw_port = os.getenv("TCP_PORT", "8765").strip()
    try:
        return int(raw_port)
    except ValueError:
        return 8765


def main():
    host = resolve_tcp_host()
    port = resolve_tcp_port()

    print(f"Connecting to TCP sender at {host}:{port} ...", flush=True)

    with socket.create_connection((host, port)) as sock:
        with sock.makefile("r", encoding="utf-8") as stream:
            print("Connected. Waiting for JSON hand-state data...", flush=True)
            for line in stream:
                line = line.strip()
                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Invalid JSON: {line}", flush=True)
                    continue

                print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
