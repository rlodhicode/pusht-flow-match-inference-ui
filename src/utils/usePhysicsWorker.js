/**
 * usePhysicsWorker.js
 *
 * React hook that owns the physicsWorker.js Web Worker lifecycle.
 * Returns { ready, obs, coverage, done, step, reset, initState }.
 *
 * Usage in App.jsx:
 *   const { ready, obs, coverage, done, step, reset } = usePhysicsWorker();
 *
 *   // obs is [agentX, agentY, blockX, blockY, blockAngle]  — same as notebook _get_obs()
 *   // step(targetX, targetY) — call once per control tick
 *   // reset(optionalState)   — restart env
 */

import { useEffect, useRef, useCallback, useState } from "react";

export function usePhysicsWorker(initialState = null) {
  const workerRef = useRef(null);
  const resolversRef = useRef([]); // queue of {resolve, reject} for each message

  const [ready, setReady] = useState(false);
  const [obs, setObs] = useState(null); // [ax, ay, bx, by, angle]
  const [coverage, setCoverage] = useState(0);
  const [done, setDone] = useState(false);

  // ── Spawn worker once ────────────────────────────────────────────────────
  useEffect(() => {
    const worker = new Worker("/physicsWorker.js");
    workerRef.current = worker;

    worker.onmessage = (e) => {
      const msg = e.data;

      if (msg.type === "ready") {
        // Worker booted — send initial state (or let it sit idle)
        setReady(true);
        if (initialState) {
          worker.postMessage({ type: "init", state: initialState });
        } else {
          worker.postMessage({ type: "reset" });
        }
        return;
      }

      if (msg.type === "obs") {
        // Ensure obs is always a plain JS Array (never a JsProxy)
        const obs = Array.isArray(msg.obs) ? msg.obs : Array.from(msg.obs);
        const safeMsg = { ...msg, obs };
        setObs(obs);
        setCoverage(msg.coverage);
        setDone(msg.done);
        // Resolve the pending promise from step()/reset() if any
        const resolver = resolversRef.current.shift();
        if (resolver) resolver.resolve(safeMsg);
        return;
      }

      if (msg.type === "error") {
        console.error("[physicsWorker]", msg.message);
        const resolver = resolversRef.current.shift();
        if (resolver) resolver.reject(new Error(msg.message));
      }
    };

    worker.onerror = (e) => {
      console.error("[physicsWorker] uncaught:", e);
    };

    return () => worker.terminate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Helper: send a message and get a Promise back ────────────────────────
  const sendAndWait = useCallback((msg) => {
    return new Promise((resolve, reject) => {
      resolversRef.current.push({ resolve, reject });
      workerRef.current.postMessage(msg);
    });
  }, []);

  // ── Public API ───────────────────────────────────────────────────────────

  /** Run one control step. Returns Promise<{obs, coverage, done}>. */
  const step = useCallback(
    (targetX, targetY) =>
      sendAndWait({ type: "step", action: [targetX, targetY] }),
    [sendAndWait],
  );

  /** Reset the environment. Pass a 5-element state array or nothing for random. */
  const reset = useCallback(
    (state = null) => sendAndWait({ type: "reset", state }),
    [sendAndWait],
  );

  /** Set an explicit initial state without resetting the full space. */
  const initState = useCallback(
    (state) => sendAndWait({ type: "init", state }),
    [sendAndWait],
  );

  return { ready, obs, coverage, done, step, reset, initState };
}
