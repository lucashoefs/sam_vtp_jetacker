#!/usr/bin/env python3
"""Hiwonder RRC protocol driver for the mini STM32 controller (STM32F407VET6).

Protocol confirmed empirically against a JetAcker chassis board
(USB 1a86:55d4) by validating 666 captured frames:

    frame = 0xAA 0x55 | func(u8) | len(u8) | data[len] | crc8
    crc8  = CRC-8/MAXIM (poly 0x31 reflected, init 0x00) over func+len+data

Known payloads:
    func 3  MOTOR  [0x01, n] + n x (id_u8 zero-based, float32_le rev/s)
    func 7  IMU    24B = 6 x float32_le: ax, ay, az (g), gx, gy, gz

Still being determined: func 5 BUS_SERVO. Use `probe-servo` to find it.

CLI:
    python3 rrc.py imu                      # decode the IMU stream
    python3 rrc.py probe-servo              # find the bus servo format (SAFE)
    python3 rrc.py servo --id 1 --pos 500   # once the format is known
    python3 rrc.py motor --id 2 --rps 0.5   # WHEELS OFF THE GROUND
    python3 rrc.py stop
"""

from __future__ import annotations

import argparse
import struct
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial missing:  sudo apt install python3-serial")


# --------------------------------------------------------------------------
# CRC-8/MAXIM  (poly 0x31 reflected -> 0x8C, init 0x00, no final xor)
# --------------------------------------------------------------------------
def _build_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8C if crc & 1 else crc >> 1
        table.append(crc)
    return table


_CRC_TABLE = _build_table()


def crc8_maxim(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc = _CRC_TABLE[crc ^ byte]
    return crc


class PacketFunc:
    SYS = 0
    LED = 1
    BUZZER = 2
    MOTOR = 3
    PWM_SERVO = 4
    BUS_SERVO = 5
    KEY = 6
    IMU = 7
    GAMEPAD = 8
    SBUS = 9
    OLED = 10
    RGB = 11


class RRCBoard:
    """Minimal driver. Opens the port, frames packets, parses telemetry."""

    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 1000000):
        self.port = serial.Serial(port, baud, timeout=0.1)
        self._buf = bytearray()

    # -- writing ----------------------------------------------------------
    def write_packet(self, func: int, data: bytes) -> bytes:
        body = bytes([func, len(data)]) + bytes(data)
        pkt = b"\xAA\x55" + body + bytes([crc8_maxim(body)])
        self.port.write(pkt)
        self.port.flush()
        return pkt

    def set_motor_speed(self, speeds) -> bytes:
        """speeds: iterable of (motor_id_1based, rev_per_sec).

        IDs are 1-based here to match Hiwonder's own code; the wire format
        is zero-based, so 1 is subtracted on encode.
        """
        speeds = list(speeds)
        data = bytearray([0x01, len(speeds)])
        for motor_id, rps in speeds:
            data.extend(struct.pack("<Bf", int(motor_id) - 1, float(rps)))
        return self.write_packet(PacketFunc.MOTOR, bytes(data))

    def stop_motors(self) -> None:
        self.set_motor_speed([(i, 0.0) for i in range(1, 5)])


    def set_bus_servo_position(self, duration_s, positions):
        """CONFIRMED payload for func 5.

            [0x01, duration_u16_le, count, (id_u8, position_u16_le) x count]

        Mirrors the SDK signature bus_servo_set_position(duration, [[id, pos]]).
        One duration for all servos: these servos interpolate internally, so a
        shared duration makes multi-servo moves start and finish together.

        positions: iterable of (servo_id, position 0..1000)
        """
        positions = list(positions)
        dur = int(float(duration_s) * 1000.0)
        data = bytearray([0x01])
        data.extend(struct.pack("<H", dur))
        data.append(len(positions))
        for servo_id, pos in positions:
            data.extend(struct.pack("<BH", int(servo_id), int(pos)))
        return self.write_packet(PacketFunc.BUS_SERVO, bytes(data))

    def read_bus_servo(self, subcmd: int, servo_id: int, timeout: float = 0.3):
        """Register read. Reply payload is [id, subcmd, status, value...]."""
        self.write_packet(PacketFunc.BUS_SERVO, bytes([subcmd, servo_id]))
        deadline = time.time() + timeout
        while time.time() < deadline:
            for func, data in self.read_frames(0.05):
                if (func == PacketFunc.BUS_SERVO and len(data) >= 3
                        and data[0] == servo_id and data[1] == subcmd):
                    if data[2] != 0:
                        return None
                    tail = data[3:]
                    if len(tail) == 1:
                        return tail[0]
                    if len(tail) == 2:
                        return struct.unpack("<H", tail)[0]
                    return tail
        return None

    def read_servo_position(self, servo_id: int):
        return self.read_bus_servo(5, servo_id)

    def set_led(self, on_s=0.2, off_s=0.2, repeat=3, led_id=1) -> bytes:
        payload = struct.pack(
            "<BHHH", led_id, int(on_s * 1000), int(off_s * 1000), repeat
        )
        return self.write_packet(PacketFunc.LED, payload)

    def set_buzzer(self, freq=2400, on_s=0.1, off_s=0.1, repeat=1) -> bytes:
        payload = struct.pack(
            "<HHHH", freq, int(on_s * 1000), int(off_s * 1000), repeat
        )
        return self.write_packet(PacketFunc.BUZZER, payload)

    # -- reading ----------------------------------------------------------
    def read_frames(self, timeout_s: float = 0.05):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            chunk = self.port.read(256)
            if chunk:
                self._buf.extend(chunk)
        while True:
            idx = self._buf.find(b"\xAA\x55")
            if idx < 0 or len(self._buf) < idx + 4:
                return
            func = self._buf[idx + 2]
            length = self._buf[idx + 3]
            end = idx + 5 + length
            if len(self._buf) < end:
                del self._buf[:idx]
                return
            data = bytes(self._buf[idx + 4 : idx + 4 + length])
            crc = self._buf[idx + 4 + length]
            del self._buf[:end]
            if crc8_maxim(bytes([func, length]) + data) == crc:
                yield func, data

    def read_imu(self, timeout_s: float = 0.5):
        """Return (ax, ay, az, gx, gy, gz) or None. Accel is in g."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            for func, data in self.read_frames(0.05):
                if func == PacketFunc.IMU and len(data) == 24:
                    return struct.unpack("<6f", data)
        return None

    def close(self) -> None:
        try:
            self.stop_motors()
        except Exception:
            pass
        self.port.close()


# --------------------------------------------------------------------------
# Bus servo format discovery
#
# The payload layout for func 5 is not published. Hiwonder's higher-level
# ServosPosition message carries a duration plus (id, position) pairs, and
# set_motor_speed establishes a [subcommand, count] prefix convention, so
# these are the plausible encodings. Each is tried in turn; watch the wheels.
# --------------------------------------------------------------------------
def servo_payloads(servo_id: int, position: int, duration_ms: int = 500):
    """Yield (description, payload) candidates for a servo position command."""
    sid, pos, dur = int(servo_id), int(position), int(duration_ms)

    yield ("A: [0x01, dur16, n, (id,pos16)]",
           bytes([0x01]) + struct.pack("<H", dur) + bytes([1])
           + struct.pack("<BH", sid, pos))

    yield ("B: [dur16, n, (id,pos16)]",
           struct.pack("<H", dur) + bytes([1]) + struct.pack("<BH", sid, pos))

    yield ("C: [0x01, n, (id,pos16)]",
           bytes([0x01, 1]) + struct.pack("<BH", sid, pos))

    yield ("D: [n, (id,pos16)]",
           bytes([1]) + struct.pack("<BH", sid, pos))

    yield ("E: [0x01, id, pos16, dur16]",
           bytes([0x01, sid]) + struct.pack("<HH", pos, dur))

    yield ("F: [id, pos16, dur16]",
           bytes([sid]) + struct.pack("<HH", pos, dur))

    yield ("G: [0x01, n, (id,pos16,dur16)]",
           bytes([0x01, 1]) + struct.pack("<BHH", sid, pos, dur))

    yield ("H: [0x01, dur16, n, (id,pos16)] dur first as u8",
           bytes([0x01, dur & 0xFF, 1]) + struct.pack("<BH", sid, pos))


def probe_servo(board: RRCBoard, servo_ids, dwell: float) -> None:
    """One candidate at a time, with a prompt between each.

    The previous version fired everything back-to-back, which made it
    impossible to tell which packet caused observed motion. Now each
    candidate is sent alone and you confirm before moving on.
    """
    print("BUS SERVO FORMAT PROBE  (interactive)")
    print("=" * 60)
    print("One command is sent at a time. After each, say whether anything")
    print("moved. Drive motors are never commanded.\n")
    print("SAFETY: if a servo ends up straining against a stop, answer 'q'")
    print("and power-cycle the chassis board before continuing.\n")

    hits = []
    for sid in servo_ids:
        for label, _ in servo_payloads(sid, 500):
            letter = label[0]
            print(f"\n--- servo id {sid}, format {letter}")
            print(f"    {label}")

            # centre first, so any motion from the next step is meaningful
            payload = dict(
                (l[0], pl) for l, pl in servo_payloads(sid, 500)
            )[letter]
            board.write_packet(PacketFunc.BUS_SERVO, payload)
            time.sleep(dwell)

            # then a clear, small deflection
            payload = dict(
                (l[0], pl) for l, pl in servo_payloads(sid, 420)
            )[letter]
            pkt = board.write_packet(PacketFunc.BUS_SERVO, payload)
            print(f"    sent: {pkt.hex(' ')}")
            time.sleep(dwell)

            ans = input("    moved? [f]ront wheels / [c]amera / [n]o / [q]uit: ")
            ans = ans.strip().lower()

            if ans.startswith("q"):
                print("\naborting - power-cycle the board if anything is straining")
                break
            if ans.startswith("f"):
                hits.append((sid, letter, "front wheels", label))
                print("    ^ STEERING SERVO")
            elif ans.startswith("c"):
                hits.append((sid, letter, "camera mount", label))
                print("    ^ pan-tilt servo")

            # return to centre before the next candidate
            payload = dict(
                (l[0], pl) for l, pl in servo_payloads(sid, 500)
            )[letter]
            board.write_packet(PacketFunc.BUS_SERVO, payload)
            time.sleep(dwell)
        else:
            continue
        break

    print("\n" + "=" * 60)
    if hits:
        print("RESULTS")
        for sid, letter, what, label in hits:
            print(f"  servo id {sid}, format {letter} -> {what}")
            print(f"      {label}")
    else:
        print("nothing identified")


def centre_all(board: RRCBoard, servo_ids, letter: str) -> None:
    """Send position 500 to every listed servo using one payload format."""
    for sid in servo_ids:
        payload = dict(
            (l[0], pl) for l, pl in servo_payloads(sid, 500)
        )[letter.upper()]
        board.write_packet(PacketFunc.BUS_SERVO, payload)
        print(f"  centred servo {sid} (format {letter})")
        time.sleep(0.3)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("imu")
    sub.add_parser("led")
    sub.add_parser("stop")

    p = sub.add_parser("probe-servo")
    p.add_argument("--ids", default="1,2,3,4")
    p.add_argument("--dwell", type=float, default=0.6)

    p = sub.add_parser("centre")
    p.add_argument("--ids", default="1,2,3,4")
    p.add_argument("--format", default="A")

    p = sub.add_parser("servo")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--pos", type=int, required=True)
    p.add_argument("--format", default="A", help="candidate letter from probe")

    p = sub.add_parser("motor")
    p.add_argument("--id", type=int, required=True, help="1-based motor id")
    p.add_argument("--rps", type=float, required=True)
    p.add_argument("--seconds", type=float, default=2.0)

    args = ap.parse_args()
    board = RRCBoard(args.port)

    try:
        if args.cmd == "imu":
            print("ax, ay, az (g)      gx, gy, gz")
            for _ in range(40):
                vals = board.read_imu()
                if vals:
                    print("  " + "  ".join(f"{v:8.3f}" for v in vals))
                time.sleep(0.05)

        elif args.cmd == "led":
            pkt = board.set_led()
            print(f"sent: {pkt.hex(' ')}")
            print("the LED on the STM32 board should blink 3 times")
            time.sleep(2.0)

        elif args.cmd == "stop":
            board.stop_motors()
            print("all motors zeroed")

        elif args.cmd == "probe-servo":
            ids = [int(x) for x in args.ids.split(",")]
            probe_servo(board, ids, args.dwell)

        elif args.cmd == "centre":
            centre_all(board, [int(x) for x in args.ids.split(",")], args.format)

        elif args.cmd == "servo":
            payload = dict(
                (l[0], p) for l, p in servo_payloads(args.id, args.pos)
            )[args.format.upper()]
            pkt = board.write_packet(PacketFunc.BUS_SERVO, payload)
            print(f"sent: {pkt.hex(' ')}")

        elif args.cmd == "motor":
            print("!! WHEELS OFF THE GROUND !!")
            input("press Enter to confirm... ")
            board.set_motor_speed([(args.id, args.rps)])
            time.sleep(args.seconds)
            board.stop_motors()
            print("stopped")

    finally:
        board.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
