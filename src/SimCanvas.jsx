/**
 * SimCanvas.jsx
 *
 * Pure renderer — no physics dependency.
 * All coordinates are in pymunk space (y-UP, 0..512).
 * T-shape vertices are the exact local coords from the notebook's add_tee().
 *
 * Props:
 *   simState  — { agentPos:{x,y}, blockPos:{x,y}, blockAngle:number, target:[x,y]|null }
 *   canvasSize — pixel size of the square canvas (default 480)
 *   trajectory — array of [x,y] action waypoints to preview
 */
import { useRef, useEffect } from "react";

const WORLD_SIZE = 512;
const AGENT_RADIUS = 15;

// Exact vertices from notebook add_tee(scale=30, length=4), body-local, y-UP
const T_POLYS_LOCAL = [
  [
    [-60, 0],
    [60, 0],
    [60, 30],
    [-60, 30],
  ], // horizontal bar
  [
    [-15, 30],
    [-15, 120],
    [15, 120],
    [15, 30],
  ], // vertical bar
];

const GOAL_POS = { x: 256, y: 256 };
const GOAL_ANGLE = Math.PI / 4;

function toCanvas(x, y, s) {
  const scale = s / WORLD_SIZE;
  return [x * scale, (WORLD_SIZE - y) * scale];
}

function worldVerts(polyLocal, pos, angle) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return polyLocal.map(([lx, ly]) => ({
    x: pos.x + lx * cos - ly * sin,
    y: pos.y + lx * sin + ly * cos,
  }));
}

function drawT(ctx, pos, angle, fillStyle, s) {
  for (const poly of T_POLYS_LOCAL) {
    const verts = worldVerts(poly, pos, angle);
    ctx.beginPath();
    const [x0, y0] = toCanvas(verts[0].x, verts[0].y, s);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < verts.length; i++) {
      const [xi, yi] = toCanvas(verts[i].x, verts[i].y, s);
      ctx.lineTo(xi, yi);
    }
    ctx.closePath();
    ctx.fillStyle = fillStyle;
    ctx.fill();
  }
}

function strokeT(ctx, pos, angle, strokeStyle, lineWidth, s) {
  for (const poly of T_POLYS_LOCAL) {
    const verts = worldVerts(poly, pos, angle);
    ctx.beginPath();
    const [x0, y0] = toCanvas(verts[0].x, verts[0].y, s);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < verts.length; i++) {
      const [xi, yi] = toCanvas(verts[i].x, verts[i].y, s);
      ctx.lineTo(xi, yi);
    }
    ctx.closePath();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }
}

export default function SimCanvas({
  simState,
  canvasSize = 480,
  trajectory = [],
}) {
  const canvasRef = useRef(null);
  const s = canvasSize;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#0f1117";
    ctx.fillRect(0, 0, s, s);

    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 0.5;
    const gridN = 8,
      step = s / gridN;
    for (let i = 1; i < gridN; i++) {
      ctx.beginPath();
      ctx.moveTo(i * step, 0);
      ctx.lineTo(i * step, s);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * step);
      ctx.lineTo(s, i * step);
      ctx.stroke();
    }

    drawT(ctx, GOAL_POS, GOAL_ANGLE, "rgba(80,220,120,0.25)", s);
    strokeT(ctx, GOAL_POS, GOAL_ANGLE, "rgba(80,220,120,0.60)", 1, s);

    if (trajectory.length > 0) {
      ctx.beginPath();
      const [fx, fy] = toCanvas(trajectory[0][0], trajectory[0][1], s);
      ctx.moveTo(fx, fy);
      for (let i = 1; i < trajectory.length; i++) {
        const [tx, ty] = toCanvas(trajectory[i][0], trajectory[i][1], s);
        ctx.lineTo(tx, ty);
      }
      ctx.strokeStyle = "rgba(255,160,50,0.3)";
      ctx.lineWidth = 1;
      ctx.stroke();
      for (let i = 0; i < trajectory.length; i++) {
        const [tx, ty] = toCanvas(trajectory[i][0], trajectory[i][1], s);
        ctx.beginPath();
        ctx.arc(tx, ty, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,160,50,${(1 - i / trajectory.length) * 0.8})`;
        ctx.fill();
      }
    }

    if (!simState) return;

    drawT(ctx, simState.blockPos, simState.blockAngle, "#6b7f8f", s);
    strokeT(ctx, simState.blockPos, simState.blockAngle, "#9bafbf", 1, s);

    const scale = s / WORLD_SIZE;
    const [ax, ay] = toCanvas(simState.agentPos.x, simState.agentPos.y, s);
    const agentR = AGENT_RADIUS * scale;

    const grd = ctx.createRadialGradient(
      ax,
      ay,
      agentR * 0.3,
      ax,
      ay,
      agentR * 1.8,
    );
    grd.addColorStop(0, "rgba(65,105,225,0.3)");
    grd.addColorStop(1, "rgba(65,105,225,0)");
    ctx.beginPath();
    ctx.arc(ax, ay, agentR * 1.8, 0, Math.PI * 2);
    ctx.fillStyle = grd;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(ax, ay, agentR, 0, Math.PI * 2);
    ctx.fillStyle = "#4169e1";
    ctx.fill();
    ctx.strokeStyle = "#7fa4ff";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    if (simState.target) {
      const [tx, ty] = toCanvas(simState.target[0], simState.target[1], s);
      const cs = 5;
      ctx.strokeStyle = "#ff8c42";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(tx - cs, ty);
      ctx.lineTo(tx + cs, ty);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(tx, ty - cs);
      ctx.lineTo(tx, ty + cs);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(tx, ty, cs * 0.6, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,140,66,0.5)";
      ctx.stroke();
    }

    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, s - 2, s - 2);
  }, [simState, canvasSize, trajectory]);

  return (
    <canvas
      ref={canvasRef}
      width={s}
      height={s}
      style={{ width: "100%", height: "auto", borderRadius: "4px" }}
    />
  );
}
