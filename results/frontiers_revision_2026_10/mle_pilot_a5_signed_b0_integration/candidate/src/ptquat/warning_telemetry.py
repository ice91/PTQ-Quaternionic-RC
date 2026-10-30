"""Bounded warning telemetry: counts+one example/key, never a list/event."""
import warnings,hashlib,time,re
from contextvars import ContextVar
from contextlib import contextmanager
import numpy as np
ACTIVE=ContextVar('ptquat_warning_collector',default=None)
class WarningCollector:
    def __init__(self,max_keys=256):self.records={};self.max_keys=max_keys;self.exceptions=[]
    def capture(self,message,category,filename,lineno,model,version,point_hash):
        normalized=re.sub(r'\s+',' ',str(message)).strip()
        key=(category.__name__,normalized,str(filename),int(lineno),model,version)
        if key not in self.records:
            if len(self.records)>=self.max_keys:raise RuntimeError('Warning telemetry distinct-key limit; fail closed')
            self.records[key]=dict(category=category.__name__,normalized_message=normalized,source_file=str(filename),line=int(lineno),model=model,likelihood_version=version,count=0,first_occurrence_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),representative_parameter_hash=point_hash,numerical_validity_status='FINITE_OUTPUT_AUDIT_REQUIRED',classification='UNRESOLVED_WARNING_ROOT_CAUSE')
        self.records[key]['count']+=1
    def as_dict(self):return dict(total_events=sum(r['count'] for r in self.records.values()),distinct_keys=len(self.records),records=list(self.records.values()),exceptions=list(self.exceptions),storage='bounded one representative/key, exact counts')
@contextmanager
def collect_warnings(collector):
    token=ACTIVE.set(collector)
    try:yield collector
    finally:ACTIVE.reset(token)
class WarningContext:
    def __init__(self,model,version,p,gi):
        self.collector=ACTIVE.get();self.model=model;self.version=version
        # Hash every decoded input, not merely galaxy-local values, for forensic provenance.
        h=hashlib.sha256()
        for k in sorted(p):
            v=p[k];h.update(k.encode());h.update(str(v).encode() if isinstance(v,str) or v is None else np.asarray(v).tobytes())
        h.update(str(gi).encode());self.hash=h.hexdigest()
    def __enter__(self):
        if self.collector is None:return self
        import threading
        if threading.current_thread() is not threading.main_thread():raise RuntimeError("Warning aggregation requires serial Python main thread")
        self.scope=warnings.catch_warnings();self.scope.__enter__();warnings.simplefilter('always')
        def handler(message,category,filename,lineno,file=None,line=None):
            self.collector.capture(message,category,filename,lineno,self.model,self.version,self.hash)
        warnings.showwarning=handler
        return self
    def __exit__(self,kind,value,tb):
        if self.collector is None:return False
        if kind is not None:self.collector.exceptions.append(dict(model=self.model,likelihood_version=self.version,parameter_hash=self.hash,error=repr(value),classification='NUMERICAL_CALL_FAILED' if issubclass(kind,FloatingPointError) else 'EXCEPTION_PROPAGATED'))
        if kind is None:
            for r in self.collector.records.values():
                if r["model"]==self.model and r["likelihood_version"]==self.version:r["numerical_validity_status"]="FINITE_OUTPUT_ONLY_NOT_A_HARMLESSNESS_PROOF"
        self.scope.__exit__(kind,value,tb)
        return False
