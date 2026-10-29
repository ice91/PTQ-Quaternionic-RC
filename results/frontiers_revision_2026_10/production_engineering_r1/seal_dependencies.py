from pathlib import Path
import sys,json,ast,hashlib
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'candidate/src'))
from ptquat.dependency_gradient import source_hashes
p=B/'dependency_v1/VERIFIED_DEPENDENCIES.json';r=json.loads(p.read_text());old=json.loads((B/'DEPENDENCY_PRE_REVIEW_LOGIC_HASHES.json').read_text());tree=ast.parse((B/'candidate/src/ptquat/dependency_gradient.py').read_text());new={n.name:hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in old};assert old==new
for name,digest in r['source_hashes'].items():assert source_hashes()[name]==digest,name
r['source_hashes']=source_hashes();r['verified_versions']=['REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1'];r['seal_provenance']='read-only original full728 receipt retained; addedgradient modulehash andB0-onlyverificationguard afterindependentreview; ledger/step/sample AST unchanged'
q=B/'dependency_v1/SEALED_VERIFIED_DEPENDENCIES.json';assert not q.exists();q.write_text(json.dumps(r,indent=2)+'\n');print('SEALED',r['local_coordinates'],r['max_unaffected'])
