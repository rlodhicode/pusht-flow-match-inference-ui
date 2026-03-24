import { useState, useEffect, useRef, useCallback } from "react";
import SimCanvas from "./SimCanvas";
import { usePhysicsWorker } from "./utils/usePhysicsWorker";
import {
  loadModel,
  sampleActions,
  isModelLoaded,
  OBS_HORIZON,
  loadStats,
} from "./utils/model";

// ─── Constants ────────────────────────────────────────────────────────────────
const MAX_STEPS = 300;
const CANVAS_SIZE = 480;
const FRAME_MS = 100; // ~10 fps to match env metadata
const SUCCESS_THRESHOLD = 0.95;

// ─── Slider input component ──────────────────────────────────────────────────
function SliderField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  unit = "",
  disabled = false,
}) {
  return (
    <div className="field">
      <div className="field-row">
        <span className="field-label">{label}</span>
        <span className="field-value">
          {typeof value === "number" ? value.toFixed(step < 1 ? 1 : 0) : value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="slider"
        disabled={disabled}
      />
    </div>
  );
}

// ─── Stat chip ───────────────────────────────────────────────────────────────
function Stat({ label, value, accent }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className={`stat-value${accent ? " accent" : ""}`}>{value}</span>
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────────────────────
export default function App() {
  const isProd = import.meta.env.PROD;

  // ── Model state ──────────────────────────────────────────────────────────
  const [modelStatus, setModelStatus] = useState("idle");
  const [modelError, setModelError] = useState("");
  const [modelUrl, setModelUrl] = useState("/tiny_flowmatch.onnx");

  // ── Sim config (sliders) ─────────────────────────────────────────────────
  const [agentX, setAgentX] = useState(256);
  const [agentY, setAgentY] = useState(100);
  const [blockX, setBlockX] = useState(256);
  const [blockY, setBlockY] = useState(300);
  const [blockAngleDeg, setBlockAngleDeg] = useState(0);
  const [flowSteps, setFlowSteps] = useState(20);

  // ── Physics worker ────────────────────────────────────────────────────────
  // ready  — Pyodide + pymunk have loaded
  // obs    — [agentX, agentY, blockX, blockY, blockAngle]  (same as notebook)
  // coverage / done — updated after every step()
  // step(tx, ty)  — returns Promise, resolves after pymunk substeps complete
  // reset(state?) — returns Promise, restarts the env
  const {
    ready: workerReady,
    obs: workerObs,
    coverage,
    done: workerDone,
    step: workerStep,
    reset: workerReset,
  } = usePhysicsWorker();

  // ── Sim UI state ─────────────────────────────────────────────────────────
  const [simState, setSimState] = useState(null);
  const [trajectory, setTrajectory] = useState([]);
  const [running, setRunning] = useState(false);
  const [inferring, setInferring] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const [done, setDone] = useState(false);
  const [flowProgress, setFlowProgress] = useState(0);

  const isEditable = !running && stepCount === 0;

  // Refs for the animation loop
  const obsHistoryRef = useRef([]);
  const actionQueueRef = useRef([]);
  const stepCountRef = useRef(0);
  const runningRef = useRef(false);
  const frameTimerRef = useRef(null);
  const inferLockRef = useRef(false);

  // ── Sync workerObs → simState ─────────────────────────────────────────────
  // Whenever the worker sends back a new observation, update the canvas.
  useEffect(() => {
    if (!workerObs) return;
    const [ax, ay, bx, by, angle] = workerObs;
    setSimState((prev) => ({
      agentPos: { x: ax, y: ay },
      blockPos: { x: bx, y: by },
      blockAngle: angle,
      target: prev?.target ?? null,
    }));
  }, [workerObs]);

  // ── Sync slider values → canvas ─────────────────────────────────────────────
  useEffect(() => {
    if (!isEditable) return;
    if (!workerReady) return;

    resetSim();
  }, [agentX, agentY, blockX, blockY, blockAngleDeg]);

  // ── Model loading ─────────────────────────────────────────────────────────
  const handleLoadModel = useCallback(async () => {
    setModelStatus("loading");
    setModelError("");
    try {
      await loadModel(modelUrl);
      setModelStatus("ready");
      await loadStats();
    } catch (e) {
      console.error(e);
      setModelStatus("error");
      setModelError(e.message || String(e));
    }
  }, [modelUrl]);

  useEffect(() => {
    if (!isProd) return;
    if (modelStatus !== "idle") return;

    handleLoadModel();
  }, [isProd, modelStatus, handleLoadModel]);

  // ── Reset simulation ──────────────────────────────────────────────────────
  // Sends the slider values to the worker as a 5-element state array.
  const resetSim = useCallback(async () => {
    const blockAngleRad = (blockAngleDeg * Math.PI) / 180;
    const state = [agentX, agentY, blockX, blockY, blockAngleRad];

    obsHistoryRef.current = [];
    actionQueueRef.current = [];
    stepCountRef.current = 0;

    setStepCount(0);
    setDone(false);
    setTrajectory([]);
    setFlowProgress(0);
    setSimState({
      agentPos: { x: agentX, y: agentY },
      blockPos: { x: blockX, y: blockY },
      blockAngle: blockAngleRad,
      target: null,
    });

    if (workerReady) {
      // reset() resolves once the worker has applied the state and sent back obs
      const result = await workerReset(state);
      // Seed obsHistory with OBS_HORIZON copies of the initial obs
      const initObs = result.obs;
      obsHistoryRef.current = Array(OBS_HORIZON).fill(initObs);
    }
  }, [agentX, agentY, blockX, blockY, blockAngleDeg, workerReady, workerReset]);

  // Once the worker is ready, do the first reset to populate obsHistory
  useEffect(() => {
    if (workerReady) resetSim();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workerReady]);

  // ── Simulation step ───────────────────────────────────────────────────────
  const doSimStep = useCallback(async () => {
    if (!workerReady) return;

    // Re-fill action queue via inference when empty
    if (actionQueueRef.current.length === 0) {
      if (inferLockRef.current) return;
      inferLockRef.current = true;
      setInferring(true);
      setFlowProgress(0);

      try {
        const actions = await sampleActions(
          obsHistoryRef.current,
          flowSteps,
          (s, total) => setFlowProgress(Math.round((s / total) * 100)),
        );
        actionQueueRef.current = actions;
        setTrajectory(actions.map(([x, y]) => [x, y]));
      } catch (e) {
        console.error("[inference]", e);
      } finally {
        inferLockRef.current = false;
        setInferring(false);
        setFlowProgress(100);
      }
    }

    if (actionQueueRef.current.length === 0) return; // inference failed

    // Send next waypoint to pymunk worker
    const [tx, ty] = actionQueueRef.current.shift();
    const result = await workerStep(tx, ty);

    // result = { obs, coverage, done } — same shape as usePhysicsWorker resolves
    const newObs = result.obs;

    // Update rolling obs history (maxlen = OBS_HORIZON)
    obsHistoryRef.current = [
      ...obsHistoryRef.current.slice(-(OBS_HORIZON - 1)),
      newObs,
    ];

    stepCountRef.current += 1;
    setStepCount(stepCountRef.current);

    // Update target crosshair on canvas
    setSimState((prev) => (prev ? { ...prev, target: [tx, ty] } : prev));

    const isDone =
      result.coverage > SUCCESS_THRESHOLD || stepCountRef.current >= MAX_STEPS;

    if (isDone) {
      setDone(true);
      setRunning(false);
      runningRef.current = false;
    }
  }, [workerReady, workerStep, flowSteps]);

  // ── Animation loop ────────────────────────────────────────────────────────
  useEffect(() => {
    if (running && !done) {
      frameTimerRef.current = setInterval(() => {
        if (!runningRef.current) return;
        doSimStep();
      }, FRAME_MS);
    } else {
      clearInterval(frameTimerRef.current);
    }
    return () => clearInterval(frameTimerRef.current);
  }, [running, done, doSimStep]);

  const handleStart = () => {
    if (!isModelLoaded()) return;
    runningRef.current = true;
    setRunning(true);
    setDone(false);
  };

  const handlePause = () => {
    runningRef.current = false;
    setRunning(false);
  };

  const handleReset = () => {
    runningRef.current = false;
    setRunning(false);
    setInferring(false);
    inferLockRef.current = false;
    clearInterval(frameTimerRef.current);
    resetSim();
  };

  const rand = (min, max) => Math.random() * (max - min) + min;

  const handleRandomize = () => {
    if (!isEditable) return;

    setAgentX(rand(50, 460));
    setAgentY(rand(50, 460));
    setBlockX(rand(100, 400));
    setBlockY(rand(100, 400));
    setBlockAngleDeg(rand(0, 360));
  };

  // ── Derived display values ────────────────────────────────────────────────
  // Use coverage from the worker hook (kept live after every step)
  const coveragePct = Math.round((coverage ?? 0) * 100);
  const coverageColor =
    coverage > SUCCESS_THRESHOLD
      ? "#4ade80"
      : coverage > 0.6
        ? "#facc15"
        : "#f87171";

  // Loading label for the header badge
  const workerStatus = !workerReady ? "loading" : "ready";

  return (
    <div className="app">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-left">
          <span className="logo">◈</span>
          <div>
            <h1 className="title">PushT Flow-Match</h1>
            <p className="subtitle">
              DCT-Compressed Transformer · ONNX Inference
            </p>
          </div>
        </div>
        <div className="header-right">
          <div
            className={`badge badge-${workerStatus}`}
            style={{ marginRight: 8 }}
          >
            {workerReady ? "● pymunk ready" : "◌ loading pymunk…"}
          </div>
          <div className={`badge badge-${modelStatus}`}>
            {modelStatus === "idle" && "○ no model"}
            {modelStatus === "loading" && "◌ loading model…"}
            {modelStatus === "ready" && "● model ready"}
            {modelStatus === "error" && "✕ load failed"}
          </div>
        </div>
      </header>

      <main className="main">
        {/* ── Canvas + Stats ───────────────────────────────────────────── */}
        <section className="canvas-section">
          <div className="canvas-wrapper">
            <SimCanvas
              simState={simState}
              canvasSize={CANVAS_SIZE}
              trajectory={trajectory}
            />

            {/* Pyodide boot overlay */}
            {!workerReady && (
              <div className="overlay overlay-loading">
                <span className="overlay-icon">⏳</span>
                <span className="overlay-msg">Loading pymunk via Pyodide…</span>
              </div>
            )}

            {/* Done overlay */}
            {done && (
              <div
                className={`overlay ${coverage > SUCCESS_THRESHOLD ? "overlay-success" : "overlay-fail"}`}
              >
                <span className="overlay-icon">
                  {coverage > SUCCESS_THRESHOLD ? "✓" : "◎"}
                </span>
                <span className="overlay-msg">
                  {coverage > SUCCESS_THRESHOLD
                    ? "Success"
                    : "Time limit reached"}
                </span>
              </div>
            )}

            {/* Inference progress bar */}
            {inferring && (
              <div className="infer-bar">
                <div
                  className="infer-fill"
                  style={{ width: `${flowProgress}%` }}
                />
                <span className="infer-label">inferring… {flowProgress}%</span>
              </div>
            )}
          </div>

          {/* Stats row */}
          <div className="stats-row">
            <Stat label="step" value={`${stepCount} / ${MAX_STEPS}`} />
            <Stat label="coverage" value={`${coveragePct}%`} accent />
            <Stat label="goal" value="256, 256, 45°" />
            <Stat label="flow steps" value={flowSteps} />
          </div>

          {/* Coverage bar */}
          <div className="coverage-bar-wrap">
            <div className="coverage-bar-track">
              <div
                className="coverage-bar-fill"
                style={{ width: `${coveragePct}%`, background: coverageColor }}
              />
              <div
                className="coverage-threshold"
                style={{ left: `${SUCCESS_THRESHOLD * 100}%` }}
              />
            </div>
            <span className="coverage-label" style={{ color: coverageColor }}>
              {coveragePct}%
            </span>
          </div>

          {/* Controls */}
          <div className="controls-row">
            <button
              className="btn btn-primary"
              onClick={handleStart}
              disabled={
                running || modelStatus !== "ready" || !workerReady || done
              }
            >
              ▶ Run
            </button>
            <button className="btn" onClick={handlePause} disabled={!running}>
              ⏸ Pause
            </button>
            <button className="btn" onClick={handleReset}>
              ↺ Reset
            </button>
          </div>
        </section>

        {/* ── Side panel ───────────────────────────────────────────────── */}
        <aside className="sidebar">
          {/* Model loader */}
          <div className="panel">
            <h2 className="panel-title">Model</h2>
            <div className="field">
              <span className="field-label">ONNX path</span>
              <input
                className="text-input"
                value={modelUrl}
                onChange={(e) => setModelUrl(e.target.value)}
                placeholder="/tiny_flowmatch.onnx"
                disabled={isProd}
              />
              <span className="hint">
                {isProd ? (
                  <>Model path is fixed in production builds.</>
                ) : (
                  <>
                    Place <code>tiny_flowmatch.onnx</code> in the{" "}
                    <code>public/</code> folder. Path can be edited in
                    development.
                  </>
                )}
              </span>
            </div>
            <button
              className={`btn btn-full ${modelStatus === "loading" ? "btn-loading" : "btn-primary"}`}
              onClick={handleLoadModel}
              disabled={modelStatus === "loading" || isProd}
            >
              {modelStatus === "loading" ? "◌ Loading WASM…" : "⬇ Load Model"}
            </button>
            {modelStatus === "error" && (
              <p className="error-msg">⚠ {modelError}</p>
            )}
          </div>

          {/* Initial state */}
          <div className="panel">
            <h2 className="panel-title">Initial State</h2>
            <p className="panel-sub">Agent position</p>
            <SliderField
              label="X"
              value={agentX}
              min={50}
              max={460}
              onChange={setAgentX}
              disabled={!isEditable}
            />
            <SliderField
              label="Y"
              value={agentY}
              min={50}
              max={460}
              onChange={setAgentY}
              disabled={!isEditable}
            />
            <p className="panel-sub" style={{ marginTop: 12 }}>
              Block position & angle
            </p>
            <SliderField
              label="X"
              value={blockX}
              min={100}
              max={400}
              onChange={setBlockX}
              disabled={!isEditable}
            />
            <SliderField
              label="Y"
              value={blockY}
              min={100}
              max={400}
              onChange={setBlockY}
              disabled={!isEditable}
            />
            <SliderField
              label="Angle"
              value={blockAngleDeg}
              min={0}
              max={360}
              unit="°"
              onChange={setBlockAngleDeg}
              disabled={!isEditable}
            />
            <button
              className="btn btn-full"
              onClick={handleRandomize}
              disabled={!isEditable}
              style={{ marginTop: 8 }}
            >
              🎲 Randomize
            </button>
          </div>

          {/* Inference settings */}
          <div className="panel">
            <h2 className="panel-title">Inference</h2>
            <SliderField
              label="Flow steps"
              value={flowSteps}
              min={5}
              max={100}
              step={5}
              onChange={setFlowSteps}
            />
            <p className="hint">
              More steps = better quality but slower. Paper uses 100; 10–20 is
              fast for demos.
            </p>
          </div>

          {/* Legend */}
          <div className="panel panel-legend">
            <h2 className="panel-title">Legend</h2>
            <div className="legend">
              <span
                className="legend-swatch"
                style={{
                  background: "rgba(80,220,120,0.45)",
                  border: "1px solid rgba(80,220,120,0.7)",
                }}
              />
              <span>Goal pose</span>
            </div>
            <div className="legend">
              <span
                className="legend-swatch"
                style={{ background: "#6b7f8f" }}
              />
              <span>T-block (current)</span>
            </div>
            <div className="legend">
              <span
                className="legend-swatch"
                style={{ background: "#4169e1", borderRadius: "50%" }}
              />
              <span>Agent</span>
            </div>
            <div className="legend">
              <span
                className="legend-swatch"
                style={{ background: "rgba(255,140,50,0.8)", borderRadius: 2 }}
              />
              <span>Planned trajectory</span>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
