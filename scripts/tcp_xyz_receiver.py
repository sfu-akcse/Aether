#!/usr/bin/env python3
import argparse
import json
import socket


def main():
    parser = argparse.ArgumentParser(description="Receive XYZ points over TCP and print them.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to")
    parser.add_argument("--port", type=int, default=5000, help="TCP port to listen on")
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"Listening on {args.host}:{args.port}")

        while True:
            connection, address = server.accept()
            print(f"Connected by {address[0]}:{address[1]}")
            with connection:
                file_obj = connection.makefile("r")
                for line in file_obj:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        point = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Received invalid line: {line}")
                        continue

                    print(
                        "Received x={x:.2f} y={y:.2f} z={z:.2f}".format(
                            x=point.get("x", 0.0),
                            y=point.get("y", 0.0),
                            z=point.get("z", 0.0),
                        )
                    )


if __name__ == "__main__":
    main()
