# List a remote zip's central directory using HTTP Range requests (no full download).
import sys, struct, urllib.request

def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b), "User-Agent": "zip-lister"})
    return urllib.request.urlopen(req, timeout=60).read()

def size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "zip-lister"})
    return int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])

for url in sys.argv[1:]:
    n = size(url)
    tail = rng(url, max(0, n - 65536), n - 1)
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    if cd_off == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd = rng(url, cd_off, cd_off + cd_size - 1)
    print("==", url.rsplit("/", 1)[-1], "zip bytes", n)
    p = 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        fl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        name = cd[p + 46:p + 46 + fl].decode(errors="replace")
        print("  %12d %12d  %s" % (usize, csize, name))
        p += 46 + fl + el + cl
