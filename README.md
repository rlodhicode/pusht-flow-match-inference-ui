# PushT Flow-Match Inference UI

Browser UI for the PushT DCT-compressed flow-matching transformer, matching the pymunk rollout from the notebook exactly.

## Setup

```bash
npm install
```

Then copy your exported model to the `public/` folder:

```
public/
  tiny_flowmatch.onnx      ← fp32 model (WASM backend)
  tiny_flowmatch_fp16.onnx ← optional fp16 (not used by default)
```

```bash
npm run dev
```

Open `http://localhost:5173`.

## Usage

1. Click **⬇ Load Model** — loads the ONNX model via WASM (first load compiles WASM; takes ~5 s).
2. Adjust **Initial State** sliders (agent XY, block XY, block angle in degrees).
3. Click **↺ Apply & Reset** to apply your initial state.
4. Click **▶ Run** to start the rollout.

The simulation runs at 10 fps (matching the notebook's `control_hz=10`).  
Each inference step runs the flow-matching Euler integration for `flow_steps` iterations (default 20; use 100 for notebook parity, 5–10 for speed).

## Architecture notes

| Component | Detail |
|-----------|--------|
| Model | FlowMatchTransformer, obs_dim=6, dct_k=10, d_model=128, nhead=4, 2 layers |
| Obs | last 2 frames: `[norm_ax, norm_ay, norm_bx, norm_by, sin(θ), cos(θ)]` |
| Action | DCT-10 → IDCT-16 → slice [1:9] → un-normalise → `[x, y]` in `[0,512]` |
| Physics | Euler PD-control agent + impulse-based T-block collision |
| Goal | fixed at `[256, 256, π/4]`; success at 95 % area coverage |

## Troubleshooting

**Model fails to load**: Make sure `tiny_flowmatch.onnx` is in `public/`. The Vite dev server needs the `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers set in `vite.config.js` (already configured) for the WASM backend to work. If you deploy to production, you need those same headers on the HTTP server.

**Inference is slow**: Lower the `Flow steps` slider (5–10 is fine for demos). The model has ~1M params; first call compiles the WASM graph which takes a few seconds.

**Block doesn't move**: Check that the initial agent position is close enough to the block to make contact. The model should guide the agent to push the block toward the goal.
