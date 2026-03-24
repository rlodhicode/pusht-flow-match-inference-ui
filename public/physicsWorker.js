/**
 * physicsWorker.js
 *
 * Runs inside a Web Worker. Loads Pyodide + pymunk, then exposes a simple
 * message-passing API to App.jsx via usePhysicsWorker.js.
 *
 * Messages IN  (from main thread):
 *   { type: 'init',  state: [ax, ay, bx, by, angle] }
 *   { type: 'step',  action: [x, y] }
 *   { type: 'reset', state: [ax, ay, bx, by, angle] }   // optional explicit state
 *
 * Messages OUT (to main thread):
 *   { type: 'ready' }
 *   { type: 'obs',   obs: [ax, ay, bx, by, angle], coverage: float, done: bool }
 *   { type: 'error', message: string }
 */

importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js");

// ─── Python source (verbatim from the notebook, stripped of gym/render deps) ──
const PUSHT_PY = `
import pymunk
from pymunk.vec2d import Vec2d
import math

class PushTPhysics:
    """
    Minimal port of PushTEnv physics — no gym, no rendering, no shapely.
    Matches the notebook _setup() / step() / _get_obs() exactly.
    """

    SIM_HZ      = 100
    CONTROL_HZ  = 10
    K_P         = 100.0
    K_V         = 20.0
    GOAL_POSE   = (256.0, 256.0, math.pi / 4)
    SUCCESS_THR = 0.95

    def __init__(self):
        self.space = None
        self.agent = None
        self.block = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init(self, state):
        """Build the space and set initial state."""
        if hasattr(state, "to_py"):
            state = state.to_py()
        self._setup()
        self._set_state(state)

    def step(self, action_x, action_y):
        """
        Run one control step (SIM_HZ/CONTROL_HZ substeps).
        Returns (obs_list, coverage, done).
        """
        action = Vec2d(action_x, action_y)
        dt = 1.0 / self.SIM_HZ
        n_steps = self.SIM_HZ // self.CONTROL_HZ

        for _ in range(n_steps):
            acceleration = (
                self.K_P * (action - self.agent.position)
                + self.K_V * (Vec2d(0, 0) - self.agent.velocity)
            )
            self.agent.velocity += acceleration * dt
            self.space.step(dt)

        obs      = self._get_obs()
        coverage = self._compute_coverage()
        done     = coverage > self.SUCCESS_THR
        return obs, coverage, done

    def reset(self, state=None):
        """Re-initialise the space with a new (or random) state."""
        import random
        # Convert JsProxy → Python before any checks (JS null becomes Python None)
        if hasattr(state, "to_py"):
            state = state.to_py()
        self._setup()
        if state is None:
            state = [
                random.randint(50, 450),
                random.randint(50, 450),
                random.randint(100, 400),
                random.randint(100, 400),
                random.uniform(-math.pi, math.pi),
            ]
        self._set_state(state)
        obs      = self._get_obs()
        coverage = self._compute_coverage()
        return obs, coverage, False

    # ------------------------------------------------------------------
    # Internal helpers (copied verbatim from notebook)
    # ------------------------------------------------------------------

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0

        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        self.agent = self._add_circle((256, 400), 15)
        self.block = self._add_tee((256, 300), 0)
        self.goal_pose = list(self.GOAL_POSE)
        self.success_threshold = self.SUCCESS_THR

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        return shape

    def _add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        self.space.add(body, shape)
        return body

    def _add_tee(self, position, angle, scale=30):
        mass   = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            ( length * scale / 2, scale),
            ( length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        vertices2 = [
            (-scale / 2,          scale),
            (-scale / 2, length * scale),
            ( scale / 2, length * scale),
            ( scale / 2,          scale),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body     = pymunk.Body(mass, inertia1 + inertia2)
        shape1   = pymunk.Poly(body, vertices1)
        shape2   = pymunk.Poly(body, vertices2)
        body.center_of_gravity = (
            (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        )
        body.position = position
        body.angle    = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def _set_state(self, state):
        # state may arrive as a Pyodide JsProxy — convert to plain Python list
        if hasattr(state, "to_py"):
            state = state.to_py()
        state = list(state)
        pos_agent = [float(state[0]), float(state[1])]
        pos_block = [float(state[2]), float(state[3])]
        rot_block  = float(state[4])
        self.agent.position = pos_agent
        self.block.angle    = rot_block
        self.block.position = pos_block
        self.space.step(1.0 / self.SIM_HZ)

    def _get_obs(self):
        a = self.agent.position
        b = self.block.position
        return [
            float(a.x), float(a.y),
            float(b.x), float(b.y),
            float(self.block.angle % (2 * math.pi)),
        ]

    def _compute_coverage(self):
        """
        Polygon-clipping coverage identical to the notebook's shapely approach,
        but implemented with pure Python so we don't need shapely in Pyodide.
        We use pymunk's own vertex data and a simple shoelace + S-H clipper.
        """
        goal_body = pymunk.Body(1, 1)
        goal_body.position = (self.goal_pose[0], self.goal_pose[1])
        goal_body.angle    = self.goal_pose[2]

        def world_verts(body, shape):
            return [body.local_to_world(v) for v in shape.get_vertices()]

        def poly_area(pts):
            n    = len(pts)
            area = 0.0
            for i in range(n):
                j     = (i + 1) % n
                area += pts[i][0] * pts[j][1]
                area -= pts[j][0] * pts[i][1]
            return abs(area) / 2.0

        def clip_poly(subj, clip):
            """Sutherland-Hodgman."""
            def inside(p, a, b):
                return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0
            def intersect(p1, p2, p3, p4):
                d1 = (p2[0]-p1[0], p2[1]-p1[1])
                d2 = (p4[0]-p3[0], p4[1]-p3[1])
                cr = d1[0]*d2[1] - d1[1]*d2[0]
                if abs(cr) < 1e-10:
                    return p1
                t  = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / cr
                return (p1[0]+t*d1[0], p1[1]+t*d1[1])
            output = list(subj)
            for i in range(len(clip)):
                if not output:
                    return []
                inp  = output
                output = []
                a, b = clip[i], clip[(i+1) % len(clip)]
                for j in range(len(inp)):
                    cur  = inp[j]
                    prev = inp[j-1]
                    if inside(cur, a, b):
                        if not inside(prev, a, b):
                            output.append(intersect(prev, cur, a, b))
                        output.append(cur)
                    elif inside(prev, a, b):
                        output.append(intersect(prev, cur, a, b))
            return output

        shapes = list(self.block.shapes)
        goal_polys  = [
            [(float(v.x), float(v.y)) for v in world_verts(goal_body, s)]
            for s in shapes
        ]
        block_polys = [
            [(float(v.x), float(v.y)) for v in world_verts(self.block, s)]
            for s in shapes
        ]

        inter_area = 0.0
        for gp in goal_polys:
            for bp in block_polys:
                clipped     = clip_poly(gp, bp)
                inter_area += poly_area(clipped)

        goal_area = sum(poly_area(gp) for gp in goal_polys)
        return min(1.0, inter_area / goal_area) if goal_area > 0 else 0.0


# Global singleton — worker keeps one env alive between messages
_env = PushTPhysics()
`;

// ─── Worker state ──────────────────────────────────────────────────────────────
let pyodide = null;
let env = null; // Python PushTPhysics instance

async function init() {
  pyodide = await loadPyodide();

  // Install pymunk from self-hosted wheel in public/ — avoids GitHub CORS block.
  // One-time setup (run from your project root):
  //   curl -L -o public/pymunk-7.2.0-cp312-cp312-pyodide_2024_0_wasm32.whl
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  const PYMUNK_WHEEL = "pymunk-7.2.0-cp312-cp312-pyodide_2024_0_wasm32.whl";
  await micropip.install(`${self.location.origin}/${PYMUNK_WHEEL}`);

  // Define the Python class and grab the singleton
  await pyodide.runPythonAsync(PUSHT_PY);
  env = pyodide.globals.get("_env");

  self.postMessage({ type: "ready" });
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Safely convert a Python list / JsProxy to a plain JS Array of numbers.
 * Works whether the value is already a JS array, a Pyodide JsProxy, or a
 * Python list returned from a tuple unpack.
 */
function toJsArray(val) {
  if (Array.isArray(val)) return val;
  if (val && typeof val.toJs === "function") return val.toJs({ depth: 2 });
  // Pyodide tuple/list returned from result[0] — iterate it
  return Array.from(val);
}

/**
 * Unpack a Python (obs_list, coverage, done) tuple safely.
 * Python tuples crossing the boundary become JsProxy iterables.
 */
function unpackResult(result) {
  // result is a Python tuple — index access works on JsProxy
  const obs = toJsArray(result.get(0));
  const coverage = result.get(1);
  const done = result.get(2);
  return { obs, coverage, done };
}

// ─── Message handler ───────────────────────────────────────────────────────────
self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      env.init(msg.state);
      const obs = toJsArray(env._get_obs());
      const coverage = env._compute_coverage();
      self.postMessage({
        type: "obs",
        obs: Array.from(obs),
        coverage,
        done: false,
      });
    } else if (msg.type === "step") {
      const [action_x, action_y] = msg.action;
      const result = env.step(action_x, action_y);
      const { obs, coverage, done } = unpackResult(result);
      self.postMessage({ type: "obs", obs, coverage, done });
    } else if (msg.type === "reset") {
      const result = env.reset(msg.state ?? null);
      const { obs, coverage, done } = unpackResult(result);
      self.postMessage({ type: "obs", obs, coverage, done: false });
    }
  } catch (err) {
    self.postMessage({ type: "error", message: err.toString() });
  }
};

// Boot immediately
init().catch((err) =>
  self.postMessage({ type: "error", message: err.toString() }),
);
