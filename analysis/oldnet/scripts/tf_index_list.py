# Minimal parser of a TensorFlow tensor-bundle "variables.index" (leveldb SSTable of BundleEntryProto).
# Prints variable name, dtype, shape, byte size. No TensorFlow needed.
import sys

def varint(b, p):
    r = 0; s = 0
    while True:
        c = b[p]; p += 1
        r |= (c & 0x7F) << s; s += 7
        if c < 0x80: return r, p

def block_entries(data, off, size):
    blk = data[off:off + size]
    nrest = int.from_bytes(blk[-4:], "little")
    end = len(blk) - 4 - 4 * nrest
    p = 0; key = b""
    while p < end:
        shared, p = varint(blk, p); nons, p = varint(blk, p); vl, p = varint(blk, p)
        key = key[:shared] + blk[p:p + nons]; p += nons
        val = blk[p:p + vl]; p += vl
        yield key, val

def parse_proto(b):
    p = 0; out = {}
    while p < len(b):
        tag, p = varint(b, p); f, wt = tag >> 3, tag & 7
        if wt == 0:
            v, p = varint(b, p)
        elif wt == 2:
            l, p = varint(b, p); v = b[p:p + l]; p += l
        elif wt == 5:
            v = b[p:p + 4]; p += 4
        elif wt == 1:
            v = b[p:p + 8]; p += 8
        else:
            raise ValueError(wt)
        out.setdefault(f, []).append(v)
    return out

def shape_of(b):
    d = parse_proto(b)
    dims = []
    for dim in d.get(2, []):
        dd = parse_proto(dim); dims.append(dd.get(1, [0])[0])
    return tuple(dims)

data = open(sys.argv[1], "rb").read()
footer = data[-48:]
p = 0
mi_off, p = varint(footer, p); mi_size, p = varint(footer, p)
ix_off, p = varint(footer, p); ix_size, p = varint(footer, p)
DT = {1: "float32", 3: "int32", 9: "int64", 7: "string", 10: "bool"}
rows = []
for k, v in block_entries(data, ix_off, ix_size):
    h_off, q = varint(v, 0); h_size, q = varint(v, q)
    for key, val in block_entries(data, h_off, h_size):
        if key == b"":
            continue  # header entry
        e = parse_proto(val)
        dtype = DT.get(e.get(1, [0])[0], e.get(1, [0])[0])
        shp = shape_of(e[2][0]) if 2 in e else ()
        size = e.get(5, [0])[0]
        rows.append((key.decode(), dtype, shp, size))
tot = 0
for r in rows:
    tot += r[3]
    print("%-70s %-8s %-22s %d" % (r[0], r[1], r[2], r[3]))
print("TOTAL vars", len(rows), "bytes", tot)
