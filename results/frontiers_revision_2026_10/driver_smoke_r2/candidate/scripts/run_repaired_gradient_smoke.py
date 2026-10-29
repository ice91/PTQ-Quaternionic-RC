"""ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE. New jac=True CLI."""
from pathlib import Path
import sys,argparse,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);args=p.parse_args();B=Path(args.bundle).resolve()
 if hashlib.sha256((B/'SMOKE_PROTOCOL.json').read_bytes()).hexdigest()!='aefe3ebc8f0500ae3fd79eb39305da7e3025cc2b237d9df7dac0c7be3db4f6dc':raise RuntimeError('Frozen smoke protocol mismatch')
 gate=json.loads((B/'DRIVER_UNIT_GATE.json').read_text());assert gate['status']=='PASS'
 for name,digest in json.loads((B/'TESTED_CODE_HASHES.json').read_text()).items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
 for name,digest in json.loads((B/'VERIFIED_INPUT_HASHES.json').read_text()).items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
 from ptquat.repaired_smoke_driver import run,LABEL
 print(LABEL,flush=True);run(B)
if __name__=='__main__':main()
