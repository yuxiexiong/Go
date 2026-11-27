# Fetch one (small) member of a remote zip via HTTP Range. Usage: URL MEMBER_SUFFIX OUTFILE
import sys, struct, zlib, urllib.request

def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b), "User-Agent": "zip-lister"})
    return urllib.request.urlopen(req, timeout=60).read()

def size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "zip-lister"})
    return int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])

url, suffix, outf = sys.argv[1:4]
n = size(url)
tail = rng(url, max(0, n - 65536), n - 1)
i = tail.rfind(b"PK\x05\x06")
cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
cd = rng(url, cd_off, cd_off + cd_size - 1)
p = 0
while cd[p:p + 4] == b"PK\x01\x02":
    method = struct.unpack("<H", cd[p + 10:p + 12])[0]
    csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
    fl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
    loff = struct.unpack("<I", cd[p + 42:p + 46])[0]
    name = cd[p + 46:p + 46 + fl].decode()
    p += 46 + fl + el + cl
    if name.endswith(suffix):
        assert csize < 50_000_000
        lh = rng(url, loff, loff + 29)
        lfl, lel = struct.unpack("<HH", lh[26:30])
        data = rng(url, loff + 30 + lfl + lel, loff + 30 + lfl + lel + csize - 1)
        raw = data if method == 0 else zlib.decompress(data, -15)
        assert len(raw) == usize
        open(outf, "wb").write(raw)
        print("wrote", outf, name, usize)
        break
