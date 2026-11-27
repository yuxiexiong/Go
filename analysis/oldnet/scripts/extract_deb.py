# Extract data.tar.zst from .deb files (ar archives) into a sysroot. Usage: extract_deb.py SYSROOT DEB...
import sys, io, tarfile
sys.path.insert(0, "/workspace/go-oldnet-conv/pylib")
import zstandard

def ar_members(data):
    assert data[:8] == b"!<arch>\n"
    p = 8
    while p < len(data):
        hdr = data[p:p+60]; name = hdr[:16].decode().strip(); size = int(hdr[48:58].decode().strip())
        yield name, data[p+60:p+60+size]
        p += 60 + size + (size % 2)

root = sys.argv[1]
for deb in sys.argv[2:]:
    for name, body in ar_members(open(deb, "rb").read()):
        if name.startswith("data.tar"):
            assert name.endswith(".zst"), name
            raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(body)).read()
            with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
                tf.extractall(root, filter="tar")
            print("extracted", deb, name)
