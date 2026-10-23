# DistilBERT Sentiment Demo

Sentiment analysis that runs entirely in the browser with [Transformers.js](https://github.com/huggingface/transformers.js) 4.3.1 and
[`Xenova/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english) (8-bit, WebAssembly).

- `index.html` – the page: live scoring, token view, scored sample reviews.
- `worker.js` – loads the library and model in a Web Worker and classifies text.
- `ort/` – ONNX Runtime Web WebAssembly files (from `onnxruntime-web` on npm).
- `models/` – the model files. The 67.6 MB ONNX weights are split into five
  parts (`.wasm` extension so strict static hosts serve them unchanged); the
  worker stitches them back together.
- `tools/fetch-model.sh` – re-downloads the model from the Hugging Face Hub and splits it.

## Run locally

Serve the folder with any static server, then open it:

```sh
npx serve .
```
