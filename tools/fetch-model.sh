#!/usr/bin/env bash
# Downloads the model files from the Hugging Face Hub and splits the ONNX
# weights into parts small enough for the artifact host (15 MB per file).
# Parts use .wasm because the host serves that binary type unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=Xenova/distilbert-base-uncased-finetuned-sst-2-english
DEST=models/$REPO
mkdir -p "$DEST/onnx"
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json; do
  curl -fsSL --retry 3 -o "$DEST/$f" "https://huggingface.co/$REPO/resolve/main/$f"
done
TMP=$(mktemp -d)
curl -fSL --retry 3 -o "$TMP/model_quantized.onnx" "https://huggingface.co/$REPO/resolve/main/onnx/model_quantized.onnx"
rm -f "$DEST"/onnx/model_quantized.part*.wasm
split -b 14000000 -d -a 1 "$TMP/model_quantized.onnx" "$TMP/part"
i=1
sizes=()
for p in "$TMP"/part*; do sizes+=("$(stat -c %s "$p")"); mv "$p" "$DEST/onnx/model_quantized.part$i.wasm"; i=$((i+1)); done
(IFS=,; printf '{"parts":[%s]}\n' "${sizes[*]}") > "$DEST/onnx/model_quantized.parts.json"
sha256sum "$TMP/model_quantized.onnx"
rm -rf "$TMP"
ls -l "$DEST" "$DEST/onnx"
