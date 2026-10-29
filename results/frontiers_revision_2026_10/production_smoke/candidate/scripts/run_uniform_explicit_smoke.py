"""ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE. Actual isolated CLI and driver."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ptquat.data import load_tidy_sparc
from ptquat.bounded_driver import run_bounded_smoke

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args();B=Path(a.bundle).resolve()
    gate=json.loads((B/'parity_v1/PARITY_GATE.json').read_text());assert gate['status']=='PASS'
    hashes=json.loads((B/'CANDIDATE_TESTED_HASHES.json').read_text())
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    protocol=json.loads((B/'SMOKE_PROTOCOL.json').read_text());assert hashlib.sha256((B/'SMOKE_PROTOCOL.json').read_bytes()).hexdigest()=='3860858cfc991489dc5d5ee878236efa5591f43e0f11b74cb280b318beff76cc'
    inputs=json.loads((B/'INPUT_MANIFEST.json').read_text())
    for r in [inputs['data'],inputs['P1E_reference']]+inputs['theta']:
        assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
    gd=load_tidy_sparc(inputs['data']['path']);gs=[gd[k] for k in sorted(gd)]
    assert len(gs)==91 and sum(len(g.v_obs) for g in gs)==1782
    vectors={Path(r['path']).parent.name.replace('_','-'):np.load(r['path']) for r in inputs['theta']}
    run_bounded_smoke(gs,vectors,protocol,B/'engineering_smoke_v1')
if __name__=='__main__':main()
