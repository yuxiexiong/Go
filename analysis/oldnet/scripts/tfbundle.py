# Minimal reader for TensorFlow tensor-bundle checkpoints (variables.index + variables.data-00000-of-00001),
# single shard, float32, no compression. No TensorFlow needed.
import sys
import numpy as np

def _varint(b, p):
    r = 0; s = 0
    while True:
        c = b[p]; p += 1
        r |= (c & 0x7F) << s; s += 7
        if c < 0x80: return r, p

def _block_entries(data, off, size):
    blk = data[off:off + size]
    nrest = int.from_bytes(blk[-4:], "little")
    end = len(blk) - 4 - 4 * nrest
    p = 0; key = b""
    while p < end:
        shared, p = _varint(blk, p); nons, p = _varint(blk, p); vl, p = _varint(blk, p)
        key = key[:shared] + blk[p:p + nons]; p += nons
        val = blk[p:p + vl]; p += vl
        yield key, val

def _proto(b):
    p = 0; out = {}
    while p < len(b):
        tag, p = _varint(b, p); f, wt = tag >> 3, tag & 7
        if wt == 0: v, p = _varint(b, p)
        elif wt == 2:
            l, p = _varint(b, p); v = b[p:p + l]; p += l
        elif wt == 5: v = b[p:p + 4]; p += 4
        elif wt == 1: v = b[p:p + 8]; p += 8
        else: raise ValueError(wt)
        out.setdefault(f, []).append(v)
    return out

def read_index(path):
    data = open(path, "rb").read()
    footer = data[-48:]; p = 0
    _, p = _varint(footer, p); _, p = _varint(footer, p)
    ix_off, p = _varint(footer, p); ix_size, p = _varint(footer, p)
    out = {}
    for _, v in _block_entries(data, ix_off, ix_size):
        h_off, q = _varint(v, 0); h_size, q = _varint(v, q)
        for key, val in _block_entries(data, h_off, h_size):
            if key == b"": continue
            e = _proto(val)
            dims = []
            if 2 in e:
                for dim in _proto(e[2][0]).get(2, []):
                    dims.append(_proto(dim).get(1, [0])[0])
            out[key.decode()] = dict(dtype=e.get(1, [0])[0], shape=tuple(dims), shard=e.get(3, [0])[0], offset=e.get(4, [0])[0], size=e.get(5, [0])[0])
    return out

class Bundle:
    def __init__(self, index_path, data_path):
        self.index = read_index(index_path)
        self.data = np.memmap(data_path, dtype=np.uint8, mode="r")
    def get(self, name):
        e = self.index[name]
        assert e["dtype"] == 1 and e["shard"] == 0, e
        a = np.frombuffer(self.data[e["offset"]:e["offset"] + e["size"]].tobytes(), dtype="<f4")
        return a.reshape(e["shape"]).copy()
    def names(self): return list(self.index)

if __name__ == "__main__":
    b = read_index(sys.argv[1])
    tot = 0
    for k, e in b.items():
        tot += e["size"]; print("%-60s %s %s off=%d size=%d" % (k, e["dtype"], e["shape"], e["offset"], e["size"]))
    print("TOTAL", len(b), tot)
