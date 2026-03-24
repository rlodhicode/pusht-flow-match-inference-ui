// model.js — ONNX inference for PushT Flow-Match Transformer
// Mirrors the rollout loop in the notebook exactly.

import * as ort from "onnxruntime-web";

// Disable multi-threading to avoid needing SharedArrayBuffer / COOP-COEP.
// Remove this line if your server properly sets the COOP/COEP headers and
// you want faster multi-threaded WASM.
ort.env.wasm.numThreads = 1;

// ─── Config (must match training) ────────────────────────────────────────────
export const OBS_HORIZON = 2;
export const PRED_HORIZON = 16;
export const ACTION_HORIZON = 8;
export const DCT_K = 10;

// These stats come from dataset.stats in the notebook.
// obs space is [0,512] for xy and [0,2π] for angle.
// action space is [0,512] for xy.
const OBS_STATS = {
  min: [0, 0, 0, 0, 0],
  max: [512, 512, 512, 512, 2 * Math.PI],
};
const ACTION_STATS = {
  min: [0, 0],
  max: [512, 512],
};

// ─── Session ─────────────────────────────────────────────────────────────────

let _session = null;

/**
 * Load the ONNX model.
 * @param {string} url — path to tiny_flowmatch.onnx (e.g. '/tiny_flowmatch.onnx')
 */
export async function loadModel(url) {
  _session = await ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
  });
  console.log(
    "[model] Loaded. Inputs:",
    _session.inputNames,
    "Outputs:",
    _session.outputNames,
  );
}

export function isModelLoaded() {
  return _session !== null;
}

// ─── Normalisation ────────────────────────────────────────────────────────────

/** Normalise a single scalar: (v - min) / (max - min) * 2 - 1 */
function normalise(val, min, max) {
  return ((val - min) / (max - min)) * 2 - 1;
}

/** Un-normalise action: (n + 1) / 2 * (max - min) + min */
function unnormalise(n, min, max) {
  return ((n + 1) / 2) * (max - min) + min;
}

/**
 * Build the 6-D observation vector from a raw 5-D observation.
 * Raw obs: [agent_x, agent_y, block_x, block_y, block_angle]
 * Model obs: [norm_ax, norm_ay, norm_bx, norm_by, sin(angle), cos(angle)]
 */
function buildObs6(rawObs5) {
  const [ax, ay, bx, by, angle] = rawObs5;
  return [
    normalise(ax, OBS_STATS.min[0], OBS_STATS.max[0]),
    normalise(ay, OBS_STATS.min[1], OBS_STATS.max[1]),
    normalise(bx, OBS_STATS.min[2], OBS_STATS.max[2]),
    normalise(by, OBS_STATS.min[3], OBS_STATS.max[3]),
    Math.sin(angle),
    Math.cos(angle),
  ];
}

// ─── DCT-III (IDCT) with ortho normalisation ──────────────────────────────────
// Implements scipy.fftpack.idct(coeffs, norm='ortho') for a 1-D array.

function idctOrtho(coeffs) {
  const N = coeffs.length;
  const result = new Float32Array(N);
  const sqrtN = Math.sqrt(N);
  const sqrt2N = Math.sqrt(2 / N);
  for (let n = 0; n < N; n++) {
    let sum = coeffs[0] / sqrtN;
    for (let k = 1; k < N; k++) {
      sum +=
        coeffs[k] * sqrt2N * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N));
    }
    result[n] = sum;
  }
  return result;
}

/**
 * DCT decompress.
 * coeffsKx2: array of shape [DCT_K][2] (Float32Array flat, or 2D array)
 * Returns Float32Array of shape [PRED_HORIZON * 2]
 */
function dctDecompress(coeffsFlat) {
  // Pad from DCT_K to PRED_HORIZON with zeros
  const padded0 = new Float32Array(PRED_HORIZON);
  const padded1 = new Float32Array(PRED_HORIZON);
  for (let k = 0; k < DCT_K; k++) {
    padded0[k] = coeffsFlat[k * 2 + 0];
    padded1[k] = coeffsFlat[k * 2 + 1];
  }

  const traj0 = idctOrtho(padded0);
  const traj1 = idctOrtho(padded1);

  const result = new Float32Array(PRED_HORIZON * 2);
  for (let i = 0; i < PRED_HORIZON; i++) {
    result[i * 2 + 0] = traj0[i];
    result[i * 2 + 1] = traj1[i];
  }
  return result;
}

// ─── Box-Muller normal random ─────────────────────────────────────────────────
function randn() {
  const u1 = 1 - Math.random(); // avoid log(0)
  const u2 = 1 - Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ─── Main inference ───────────────────────────────────────────────────────────

/**
 * Run flow-matching sampling and return an action sequence.
 *
 * @param {Array}  obsHistory  — last OBS_HORIZON raw observations (each: [ax,ay,bx,by,angle])
 * @param {number} flowSteps   — Euler integration steps (100 in paper; 10–20 for speed)
 * @param {function} [onStep]  — optional callback(step, total) for progress
 * @returns {Array} actions — ACTION_HORIZON × [x, y] in physics coords [0,512]
 */
export async function sampleActions(obsHistory, flowSteps = 10, onStep = null) {
  if (!_session) throw new Error("Model not loaded. Call loadModel() first.");

  // ── Build obs tensor [1, OBS_HORIZON, 6] ─────────────────────────────────
  const obsFlat = new Float32Array(OBS_HORIZON * 6);
  for (let i = 0; i < OBS_HORIZON; i++) {
    const obs6 = buildObs6(obsHistory[i]);
    for (let j = 0; j < 6; j++) {
      obsFlat[i * 6 + j] = obs6[j];
    }
  }
  const obsTensor = new ort.Tensor("float32", obsFlat, [1, OBS_HORIZON, 6]);

  // ── Initialise x ~ N(0,I)  shape [1, DCT_K, 2] ───────────────────────────
  const xData = new Float32Array(DCT_K * 2);
  for (let i = 0; i < xData.length; i++) xData[i] = randn();

  // ── Euler flow integration ────────────────────────────────────────────────
  const dt = 1.0 / flowSteps;
  for (let step = 0; step < flowSteps; step++) {
    const tVal = step * dt;

    const xTensor = new ort.Tensor("float32", new Float32Array(xData), [
      1,
      DCT_K,
      2,
    ]);
    const tTensor = new ort.Tensor("float32", new Float32Array([tVal]), [1, 1]);

    const output = await _session.run({
      obs: obsTensor,
      x: xTensor,
      t: tTensor,
    });
    const vField = output.v.data; // Float32Array [1 * DCT_K * 2]

    for (let i = 0; i < DCT_K * 2; i++) {
      xData[i] += vField[i] * dt;
    }

    if (onStep) onStep(step + 1, flowSteps);
  }

  // ── DCT decompress → trajectory [PRED_HORIZON, 2] ────────────────────────
  const trajNorm = dctDecompress(xData); // [PRED_HORIZON * 2], normalised

  // ── Un-normalise ──────────────────────────────────────────────────────────
  const actions = [];
  const start = OBS_HORIZON - 1; // = 1
  const end = start + ACTION_HORIZON; // = 9
  for (let i = start; i < end; i++) {
    const nx = trajNorm[i * 2 + 0];
    const ny = trajNorm[i * 2 + 1];
    actions.push([
      unnormalise(nx, ACTION_STATS.min[0], ACTION_STATS.max[0]),
      unnormalise(ny, ACTION_STATS.min[1], ACTION_STATS.max[1]),
    ]);
  }

  return actions; // Array of ACTION_HORIZON [x,y] pairs in world coords [0,512]
}
