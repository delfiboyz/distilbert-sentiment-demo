// Runs the model off the main thread so typing never stalls.
// Every file comes from next to this script: the page's host blocks requests
// to huggingface.co, so the model and the ONNX Runtime binary ship with it.
// The library is loaded with import() rather than a static import: a module
// worker's static imports are checked against worker-src, which only allows
// the page's own files, while import() is checked against script-src.
const LIBRARY_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js';
const MODEL_ID = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english';
const MODEL_FILE = 'onnx/model_quantized.onnx';
const RUNTIME_WASM = 'ort/ort-wasm-simd-threaded.wasm';
const RUNTIME_MJS = 'ort/ort-wasm-simd-threaded.mjs';

const here = new URL('./', self.location.href);
const nativeFetch = self.fetch.bind(self);

// The quantized model is split into parts that fit the host's per-file limit.
// The parts carry a .wasm extension only because the host serves that binary
// type byte for byte; they are raw slices of the ONNX file, never run as wasm.
// Stitch them back into one streamed response, read lazily so the metadata
// probe Transformers.js makes first doesn't download anything.
async function partsResponse(url) {
  const res = await nativeFetch(url.replace(/\.onnx$/, '.parts.json'));
  if (!res.ok) throw new Error(`Model manifest returned HTTP ${res.status}`);
  const { parts } = await res.json();
  const total = parts.reduce((a, b) => a + b, 0);
  let index = 0;
  let reader = null;
  const body = new ReadableStream({
    async pull(controller) {
      for (;;) {
        if (!reader) {
          if (index >= parts.length) return controller.close();
          const partUrl = url.replace(/\.onnx$/, `.part${index + 1}.wasm`);
          const res = await nativeFetch(partUrl);
          if (!res.ok) throw new Error(`Model part ${index + 1} returned HTTP ${res.status}`);
          reader = res.body.getReader();
          index += 1;
        }
        const { done, value } = await reader.read();
        if (done) { reader = null; continue; }
        controller.enqueue(value);
        return;
      }
    },
    cancel(reason) { return reader?.cancel(reason); },
  }, { highWaterMark: 0 });
  return new Response(body, {
    status: 200,
    headers: { 'content-length': String(total), 'content-type': 'application/octet-stream' },
  });
}

function configure(env) {
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = 'models/';
  env.useBrowserCache = false;
  env.useWasmCache = false;
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.wasmPaths = {
    mjs: new URL(RUNTIME_MJS, here).href,
    wasm: new URL(RUNTIME_WASM, here).href,
  };
  env.fetch = (input, init) => {
    const url = typeof input === 'string' ? input : input.url;
    if (url.endsWith(MODEL_FILE)) return partsResponse(url);
    return nativeFetch(input, init);
  };
}

async function fetchWithProgress(url, file) {
  const res = await nativeFetch(url);
  if (!res.ok) throw new Error(`${file} returned HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const out = total ? new Uint8Array(total) : null;
  const parts = [];
  let loaded = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (out) out.set(value, loaded); else parts.push(value);
    loaded += value.length;
    self.postMessage({ type: 'progress', file, loaded, total });
  }
  if (out) return out.buffer;
  return new Blob(parts).arrayBuffer();
}

let classifier = null;

async function load() {
  const started = performance.now();
  if (typeof WebAssembly !== 'object') throw new Error('This browser has WebAssembly turned off.');
  const { pipeline, env } = await import(LIBRARY_URL);
  configure(env);
  env.backends.onnx.wasm.wasmBinary = await fetchWithProgress(new URL(RUNTIME_WASM, here).href, 'runtime');
  classifier = await pipeline('sentiment-analysis', MODEL_ID, {
    dtype: 'q8',
    device: 'wasm',
    progress_callback: (p) => {
      if (p.status === 'progress' && p.file) {
        self.postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total });
      }
    },
  });
  // The first run warms up the session; keep it out of the reported timings.
  await classifier('warm up', { top_k: 2 });
  self.postMessage({ type: 'ready', ms: performance.now() - started });
}

async function classify(text) {
  const t0 = performance.now();
  const scores = await classifier(text, { top_k: 2 });
  const ms = performance.now() - t0;
  const byLabel = Object.fromEntries(scores.map((s) => [s.label, s.score]));
  const tokens = classifier.tokenizer.tokenize(text, { add_special_tokens: true });
  return { positive: byLabel.POSITIVE ?? 0, negative: byLabel.NEGATIVE ?? 0, tokens, ms };
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') {
      await load();
    } else if (data.type === 'classify') {
      self.postMessage({ type: 'result', id: data.id, ...(await classify(data.text)) });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: data.id, stage: data.type, message: String(err?.message ?? err) });
  }
};
