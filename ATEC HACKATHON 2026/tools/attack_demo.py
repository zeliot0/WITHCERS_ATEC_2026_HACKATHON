import argparse
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.security import MissionCrypto, derive_mission_key, NONCE_LEN, TAG_LEN


def build_bitflip(crypto):
    wire = bytearray(crypto.encrypt(b"HZ:rock x=12.4 y=-3.1 conf=0.91"))
    wire[-1] ^= 0xFF
    return bytes(wire)


def build_forge(length):
    return os.urandom(max(length, NONCE_LEN + TAG_LEN))


def main():
    parser = argparse.ArgumentParser(description="External attacker for the C.R.A.T.E.R link")
    parser.add_argument("--mode", choices=["bitflip", "forge"], default="bitflip",
                        help="bitflip: valid ciphertext with one byte flipped; forge: random bytes, no key")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CRATER_RECEIVER_PORT", "47001")))
    args = parser.parse_args()

    crypto = MissionCrypto(derive_mission_key())
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Attacking {args.host}:{args.port} with {args.count} '{args.mode}' packet(s)")
    for i in range(args.count):
        payload = build_bitflip(crypto) if args.mode == "bitflip" else build_forge(64)
        sock.sendto(payload, (args.host, args.port))
        print(f"  [{i + 1}/{args.count}] sent {len(payload)} bytes")
        if i + 1 < args.count:
            time.sleep(args.interval)
    sock.close()
    print("Done. Check the Security View: the packets should appear as REJECTED.")


if __name__ == "__main__":
    main()
