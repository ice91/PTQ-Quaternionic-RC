"""Read-only diff/call-graph collection. No optimizer."""
from pathlib import Path
import ast,difflib,json,hashlib
B=Path(__file__).resolve().parent;I=B.parents[2]
patch=[];changes=[];calls=[]
for p in sorted((B/'candidate').rglob('*.py')):
    rel=p.relative_to(B/'candidate');orig=I/rel;old=orig.read_text() if orig.is_file() else '';new=p.read_text()
    if old!=new:
        patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+str(rel) if orig.is_file() else '/dev/null',tofile='b/'+str(rel)))
        changes.append(dict(file=str(rel),original_sha256=hashlib.sha256(orig.read_bytes()).hexdigest() if orig.is_file() else None,candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    for node in ast.walk(ast.parse(new)):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            for call in ast.walk(node):
                if isinstance(call,ast.Call):
                    target=ast.unparse(call.func)
                    if any(s in target for s in ['likelihood','loglike','log_marginal','covariance','minimize','catalog_velocity','integrate','callback']):
                        calls.append(dict(file=str(rel),function=node.name,line=call.lineno,target=target))
(B/'integration.diff').write_text(''.join(patch))
(B/'INTEGRATION_FILES.json').write_text(json.dumps(changes,indent=2)+'\n');(B/'CALL_GRAPH_STATIC.json').write_text(json.dumps(calls,indent=2)+'\n')
# Record the legacy original scripts that are intentionally not candidate entrypoints.
legacy=[]
for p in sorted((I/'scripts').glob('*.py')):
    s=p.read_text()
    if any(x in s for x in ['optimize_mle','fit_global','log_likelihood','evaluate_loglike']):legacy.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),disposition='UNCHANGED_HISTORICAL_NOT_PRIMARY_ENTRYPOINT; candidate supports only explicit version-specific smoke CLI'))
(B/'LEGACY_ENTRYPOINT_INVENTORY.json').write_text(json.dumps(legacy,indent=2)+'\n')
print('CHANGED',len(changes),'STATIC CALL EDGES',len(calls),'LEGACY ENTRYPOINTS',len(legacy))
