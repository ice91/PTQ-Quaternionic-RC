"""Launch the isolated P1G-A pilot. Does not import canonical or legacy source."""
import sys
from pathlib import Path

import hashlib, json
CANDIDATE = Path(__file__).resolve().parents[1]
BUNDLE = CANDIDATE.parent
protocol = json.loads((BUNDLE / "FIT_PROTOCOL.json").read_text())
driver = CANDIDATE / "src/ptquat/scientific_pilot_driver.py"
digest = hashlib.sha256(driver.read_bytes()).hexdigest()
if digest != protocol["driver_sha256"]:
    raise SystemExit("driver hash mismatch: " + digest)
locked = hashlib.sha256((BUNDLE / "FIT_PROTOCOL.json").read_bytes()).hexdigest()
if locked != (BUNDLE / "FIT_PROTOCOL_SHA256.txt").read_text().strip():
    raise SystemExit("protocol hash mismatch")
sys.path.insert(0, str(CANDIDATE / "src"))
from ptquat.scientific_pilot_driver import run

if __name__ == "__main__":
    run(BUNDLE)
