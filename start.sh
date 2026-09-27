#!/bin/sh
# Start the educational chest-film bench on this machine.
# Needs curl and python3. Does not install PyTorch.
# Downloads are skipped when the destination file is already non-empty.
# Not for clinical use.
set -eu

cd "$(dirname "$0")" || exit 1

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

PORT="${PORT:-8080}"
case "$PORT" in
  ''|*[!0-9]*)
    echo "PORT must be a number." >&2
    exit 1
    ;;
esac

RELEASE="https://github.com/srajones/torchxrayvision/releases/download/edu-bench-v1"
ORT_VERSION="1.30.0"
DICOM_VERSION="1.8.21"
CDN="https://cdn.jsdelivr.net/npm"
MODELS="demo/browser/models"
VENDOR="demo/browser/vendor"
PAGE="demo/browser"

if [ ! -f "$MODELS/registry.json" ]; then
  echo "Missing $MODELS/registry.json. Run ./start.sh from a clone of this repo." >&2
  exit 1
fi

mkdir -p "$MODELS" "$VENDOR"

fetch() {
  dest=$1
  url=$2
  if [ -s "$dest" ]; then
    printf 'have %s\n' "$(basename "$dest")"
    return 0
  fi
  printf 'downloading %s\n' "$(basename "$dest")"
  tmp="${dest}.part"
  if ! curl -fL --retry 3 --retry-delay 2 -A "torchxrayvision-edu-bench" -o "$tmp" "$url"; then
    rm -f "$tmp"
    printf 'download failed: %s\n' "$url" >&2
    exit 1
  fi
  mv "$tmp" "$dest"
}

files=$(python3 - "$MODELS/registry.json" <<'PY'
import json
import sys

registry = json.load(open(sys.argv[1], encoding="utf-8"))
for model in registry["models"]:
    name = model["file"]
    if "/" in name or "\\" in name or name.startswith("."):
        raise SystemExit("refusing model filename " + name)
    print(name)
PY
) || exit 1

for name in $files; do
  fetch "$MODELS/$name" "$RELEASE/$name"
done

fetch "$VENDOR/ort.wasm.min.mjs" "$CDN/onnxruntime-web@${ORT_VERSION}/dist/ort.wasm.min.mjs"
fetch "$VENDOR/ort-wasm-simd-threaded.mjs" "$CDN/onnxruntime-web@${ORT_VERSION}/dist/ort-wasm-simd-threaded.mjs"
fetch "$VENDOR/ort-wasm-simd-threaded.wasm" "$CDN/onnxruntime-web@${ORT_VERSION}/dist/ort-wasm-simd-threaded.wasm"
fetch "$VENDOR/dicomParser.min.js" "$CDN/dicom-parser@${DICOM_VERSION}/dist/dicomParser.min.js"

echo "Starting the bench on http://127.0.0.1:${PORT}/"

python3 - "$PORT" "$PAGE" <<'PY' &
import mimetypes
import os
import sys
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from http.server import HTTPServer

port = int(sys.argv[1])
directory = os.path.abspath(sys.argv[2])
os.chdir(directory)

class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class Handler(SimpleHTTPRequestHandler):
    def list_directory(self, path):
        self.send_error(404, "Not found")
        return None

    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".js", ".mjs"):
            return "text/javascript"
        if ext == ".wasm":
            return "application/wasm"
        if ext == ".json":
            return "application/json"
        if ext == ".svg":
            return "image/svg+xml"
        guessed = mimetypes.guess_type(path)[0]
        return guessed or "application/octet-stream"

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.command, self.path.split("?", 1)[0]))

try:
    server = ThreadedServer(("127.0.0.1", port), Handler)
except OSError as error:
    sys.stderr.write("Could not listen on 127.0.0.1:%s (%s). Try PORT=8765 ./start.sh\n" % (port, error))
    raise SystemExit(1)

print("listening", flush=True)
server.serve_forever()
PY
server_pid=$!

cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
stop() {
  cleanup
  exit 0
}
trap stop INT TERM
trap cleanup EXIT

ready=0
i=0
while [ "$i" -lt 50 ]; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "The local server exited before it was ready." >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    ready=1
    break
  fi
  i=$((i + 1))
  sleep 0.2
done

if [ "$ready" -ne 1 ]; then
  echo "The bench did not answer on port ${PORT}." >&2
  exit 1
fi

url="http://127.0.0.1:${PORT}/"
echo "Film bench: $url"
echo "Not for clinical use. Leave this window open. Ctrl+C stops the server."

if command -v open >/dev/null 2>&1; then
  open "$url" || true
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$url" >/dev/null 2>&1 || true
fi

wait "$server_pid"
