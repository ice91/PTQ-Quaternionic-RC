"""Read-only source authority validation, never Git mutation."""
from pathlib import Path
import json,hashlib,subprocess
B=Path(__file__).resolve().parent;I=B.parents[2];P=I.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True)
def main():
 before=json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text());missing=[p for p in before if not Path(p).is_file()];changed=[p for p,h in before.items() if Path(p).is_file() and sha(Path(p))!=h]
 tested=json.loads((B/'CANDIDATE_TESTED_HASHES.json').read_text());candidate_changed=[p for p,h in tested.items() if sha(Path(p))!=h]
 record=dict(files=len(before),changed=changed,missing=missing,candidate_tested_hash_mismatches=candidate_changed,status='PASS' if not (changed or missing or candidate_changed) else 'FAIL')
 (B/'PRESERVATION_VALIDATION.json').write_text(json.dumps(record,indent=2)+'\n');assert record['status']=='PASS'
 print(json.dumps(record,indent=2))
if __name__=='__main__':main()
