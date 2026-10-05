#!/bin/sh
# Upstream validation fixtures; manga artwork is downloaded only, not redistributed.
set -eu
revision=c333b5d36e88d539d6b040b4c4cf90ad5ecd4f69
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
dir="$root/tests/fixtures"
mkdir -p "$dir/images"
base="https://raw.githubusercontent.com/kha-white/manga-ocr/$revision"
curl -fLsS --retry 3 "$base/manga_ocr/assets/example.jpg" -o "$dir/example.jpg"
for i in 00 01 02 03 04 05 06 07 08 09 10 11; do
    curl -fLsS --retry 3 "$base/tests/data/images/$i.jpg" -o "$dir/images/$i.jpg"
done

curl -fLsS --retry 3 "https://raw.githubusercontent.com/dmMaze/comic-text-detector/440b978563c71b758e31aaa315d100faba1efa2f/data/doc/AisazuNihaIrarenai-003.jpg" -o "$dir/page.jpg"
