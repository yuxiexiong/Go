# Parser for KataGo .bin.gz / .txt.gz model files, mirroring cpp/neuralnet/desc.cpp (versions 3..14, conv-only).
import gzip, struct, sys
import numpy as np

class Reader:
    def __init__(self, data, binary):
        self.d = data; self.p = 0; self.binary = binary
    def tok(self):
        d = self.d; p = self.p; n = len(d)
        while p < n and d[p:p+1].isspace(): p += 1
        s = p
        while p < n and not d[p:p+1].isspace(): p += 1
        self.p = p
        return d[s:p].decode()
    def int(self): return int(self.tok())
    def float(self): return float(self.tok())
    def floats(self, n):
        if self.binary:
            k = 0
            while self.d[self.p:self.p+1] != b'@':
                self.p += 1; k += 1
                assert k < 100
            assert self.d[self.p:self.p+5] == b'@BIN@', self.d[self.p:self.p+10]
            self.p += 5
            a = np.frombuffer(self.d, dtype='<f4', count=n, offset=self.p).copy()
            self.p += 4*n
            return a
        else:
            return np.array([self.float() for _ in range(n)], dtype=np.float32)

def conv(r):
    name=r.tok(); y=r.int(); x=r.int(); ic=r.int(); oc=r.int(); dy=r.int(); dx=r.int()
    w = r.floats(y*x*ic*oc).reshape(y,x,ic,oc).transpose(3,2,0,1).copy()  # oc,ic,y,x
    return dict(type='conv',name=name,w=w,dil=(dy,dx))
def bn(r):
    name=r.tok(); c=r.int(); eps=r.float(); hs=r.int(); hb=r.int()
    mean=r.floats(c); var=r.floats(c)
    scale=r.floats(c) if hs else None
    bias=r.floats(c) if hb else None
    return dict(type='bn',name=name,c=c,eps=eps,mean=mean,var=var,scale=scale,bias=bias)
def actv(r, v):
    name=r.tok(); kind = r.tok() if v>=11 else 'ACTIVATION_RELU'
    return dict(type='act',name=name,kind=kind)
def matmul(r):
    name=r.tok(); ic=r.int(); oc=r.int()
    w=r.floats(ic*oc).reshape(ic,oc).T.copy()  # torch Linear weight: oc,ic
    return dict(type='matmul',name=name,w=w)
def matbias(r):
    name=r.tok(); c=r.int(); return dict(type='matbias',name=name,b=r.floats(c))

def parse(path, header_only=False, raw=None):
    if raw is None:
        raw = open(path,'rb').read()
        if path.endswith('.gz'): raw = gzip.decompress(raw)
    binary = '.bin' in path
    r = Reader(raw, binary)
    m = {}
    m['name']=r.tok(); v=m['version']=r.int()
    m['numInputChannels']=r.int(); m['numInputGlobalChannels']=r.int()
    if v>=13:
        m['postprocess']=[r.float() for _ in range(7)]
    if v>=15:
        raise NotImplementedError('v15+ not handled by this parser (meta encoder etc.)')
    t = m['trunk'] = {}
    t['name']=r.tok(); t['numBlocks']=r.int(); t['trunkC']=r.int(); t['midC']=r.int(); t['regularC']=r.int(); t['dilatedC']=r.int(); t['gpoolC']=r.int()
    if header_only: return m
    t['initialConv']=conv(r); t['initialMatMul']=matmul(r)
    blocks=[]
    for i in range(t['numBlocks']):
        kind=r.tok()
        if kind=='ordinary_block':
            b=dict(kind=kind,name=r.tok(),preBN=bn(r),preAct=actv(r,v),regularConv=conv(r),midBN=bn(r),midAct=actv(r,v),finalConv=conv(r))
        elif kind=='gpool_block':
            b=dict(kind=kind,name=r.tok(),preBN=bn(r),preAct=actv(r,v),regularConv=conv(r),gpoolConv=conv(r),gpoolBN=bn(r),gpoolAct=actv(r,v),
                   gpoolToBiasMul=matmul(r),midBN=bn(r),midAct=actv(r,v),finalConv=conv(r))
        else:
            raise NotImplementedError(kind)
        blocks.append(b)
    t['blocks']=blocks
    t['trunkTipBN']=bn(r); t['trunkTipAct']=actv(r,v)
    p = m['policy'] = {}
    p['name']=r.tok()
    p['policyOutChannels'] = 1 if v<12 else 2
    for k in ['p1Conv','g1Conv']: p[k]=conv(r)
    p['g1BN']=bn(r); p['g1Act']=actv(r,v); p['gpoolToBiasMul']=matmul(r); p['p1BN']=bn(r); p['p1Act']=actv(r,v); p['p2Conv']=conv(r); p['gpoolToPassMul']=matmul(r)
    h = m['value'] = {}
    h['name']=r.tok()
    h['v1Conv']=conv(r); h['v1BN']=bn(r); h['v1Act']=actv(r,v); h['v2Mul']=matmul(r); h['v2Bias']=matbias(r); h['v2Act']=actv(r,v)
    h['v3Mul']=matmul(r); h['v3Bias']=matbias(r); h['sv3Mul']=matmul(r); h['sv3Bias']=matbias(r); h['vOwnershipConv']=conv(r)
    rest = raw[r.p:].strip()
    m['trailing_bytes'] = len(rest)
    return m

def desc(v):
    ty=v.get('type')
    if ty in ('conv','matmul'): return str(tuple(v['w'].shape))+('' if ty!='conv' or v['dil']==(1,1) else f"dil{v['dil']}")
    if ty=='matbias': return str(tuple(v['b'].shape))
    if ty=='bn': return f"BN(c={v['c']},eps={v['eps']},scale={v['scale'] is not None},bias={v['bias'] is not None})"
    if ty=='act': return v['kind']
    return '?'

def summary(m):
    t=m['trunk']; out=[]
    out.append(f"name={m['name']} version={m['version']} in={m['numInputChannels']} inGlobal={m['numInputGlobalChannels']} postprocess={m.get('postprocess')}")
    out.append(f"trunk blocks={t['numBlocks']} trunkC={t['trunkC']} midC={t['midC']} regularC={t['regularC']} dilatedC={t['dilatedC']} gpoolC={t['gpoolC']}")
    if 'initialConv' not in t: return "\n".join(out)
    out.append(f"initialConv {desc(t['initialConv'])} initialMatMul {desc(t['initialMatMul'])}")
    kinds=''.join('G' if b['kind']=='gpool_block' else 'o' for b in t['blocks'])
    out.append(f"block kinds (o=ordinary,G=gpool): {kinds}")
    seen=set()
    for b in t['blocks']:
        s="; ".join(f"{k}:{desc(v)}" for k,v in b.items() if isinstance(v,dict))
        if s not in seen:
            seen.add(s); out.append(f"{b['kind']} structure: {s}")
    out.append(f"trunkTip: {desc(t['trunkTipBN'])} {t['trunkTipAct']['kind']}")
    for hk in ['policy','value']:
        out.append(hk+": "+"; ".join(f"{k}:{desc(v)}" for k,v in m[hk].items() if isinstance(v,dict)))
    out.append(f"trailing bytes after value head: {m['trailing_bytes']}")
    return "\n".join(out)

if __name__=='__main__':
    print(summary(parse(sys.argv[1])))
