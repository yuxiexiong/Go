#!/bin/bash
# Run the official katago v1.18.1 eigen (AppImage-extracted) binary on Debian 10 via a private glibc 2.35 / libstdc++ 12 / libssl3 sysroot.
D=/workspace/go-oldnet-conv/dl
S=$D/sysroot
LD=$S/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2
exec "$LD" --library-path "$S/lib/x86_64-linux-gnu:$S/usr/lib/x86_64-linux-gnu:$D/katago/squashfs-root/usr/lib" "$D/katago/squashfs-root/usr/bin/katago" "$@"
