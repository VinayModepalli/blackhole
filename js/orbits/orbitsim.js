// Orbit lab: massive test particles on timelike geodesics of Kerr,
// restricted to the equatorial plane (which is invariant under the flow).
// Uses the exact same Hamiltonian formulation as the ray tracer, integrated
// with RK4 in double precision. Energy E = -p_t and angular momentum
// L = x p_y - y p_x are exactly conserved quantities of the Kerr metric;
// their numerical drift is displayed as a live accuracy check.

import {
  fAndK, ksRadius, metricDot, lower,
  horizonOuter, isco, ergoEquator, photonOrbit, omegaCirc,
} from "../physics/kerr.js";
import { rk4Step } from "../physics/geodesic.js";

export class OrbitSim {
  constructor(canvas, readoutEl) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.readout = readoutEl;
    this.spin = 0.9;
    this.speed = 1.0;
    this.viewRadius = 13;       // world units shown from center to edge
    this.particles = [];
    this.launch = null;         // drag-to-launch state
    this._bindPointer();
    this.preset("precessing");
  }

  // ---- particle construction -------------------------------------------

  /**
   * Build a particle from a coordinate 3-velocity (dx/dt, dy/dt) at (x, y).
   * V = (1, vx, vy, 0) is normalized with the full metric to a 4-velocity;
   * covariant momentum follows by lowering the index.
   */
  makeParticle(x, y, vx, vy, color) {
    const a = this.spin;
    const X4 = [x, y, 0];
    let V = [1, vx, vy, 0];
    // keep V timelike: shrink spatial part until g(V,V) < 0 (sub-lightspeed)
    for (let i = 0; i < 60 && metricDot(V, V, X4, a) >= -0.02; i++) {
      V = [1, V[1] * 0.92, V[2] * 0.92, 0];
    }
    const n = Math.sqrt(-metricDot(V, V, X4, a));
    const u = V.map((c) => c / n);
    const pcov = lower(u, X4, a);
    return {
      X: [x, y, 0],
      p: [pcov[1], pcov[2], 0],
      pt: pcov[0],
      E0: -pcov[0],
      L0: x * pcov[2] - y * pcov[1],
      trail: [],
      color,
      captured: false,
      tau: 0,
    };
  }

  /** Circular-orbit velocity at BL radius r, scaled by `factor`. */
  circularAt(r, factor, prograde, color) {
    const a = this.spin;
    const R = Math.sqrt(r * r + a * a); // KS Cartesian radius of the BL-r circle
    const Om = omegaCirc(r, a, prograde) * factor;
    return this.makeParticle(R, 0, 0, Om * R, color);
  }

  preset(name) {
    this.particles = [];
    const a = this.spin;
    const ri = isco(a, true);
    switch (name) {
      case "isco":
        // marginally stable circular orbit — the disk's inner edge
        this.particles.push(this.circularAt(ri * 1.001, 1, true, "#7dd3fc"));
        break;
      case "precessing":
        // eccentric orbit: relativistic periapsis precession (rosette)
        this.particles.push(this.circularAt(9, 0.82, true, "#fbbf24"));
        break;
      case "zoomwhirl":
        // near-separatrix orbit: dives from apoapsis, whirls at periapsis
        // (~ISCO radius), zooms back out — capture boundary is at f≈0.71
        this.particles.push(this.circularAt(11, 0.72, true, "#f472b6"));
        break;
      case "plunge":
        this.particles.push(this.circularAt(7, 0.5, true, "#f87171"));
        break;
      case "framedrag":
        // dropped from rest (L = 0): spacetime itself drags it around
        this.particles.push(this.makeParticle(Math.sqrt(64 + a * a), 0, 0, 0, "#a78bfa"));
        break;
      case "retrograde": {
        // retrograde vs prograde circular at the same |Omega| scale factor:
        // retrograde ISCO is much farther out — spin breaks the symmetry
        this.particles.push(this.circularAt(isco(a, false) * 1.001, 1, false, "#fb923c"));
        this.particles.push(this.circularAt(ri * 1.001, 1, true, "#7dd3fc"));
        break;
      }
    }
  }

  // ---- integration -------------------------------------------------------

  step(dtFrame) {
    const a = this.spin;
    const rH = horizonOuter(a);
    for (const pt of this.particles) {
      if (pt.captured) continue;
      // substeps of proper time, adaptive near the hole
      let budget = dtFrame * 20 * this.speed;
      while (budget > 0) {
        const r = ksRadius(pt.X, a);
        if (r < rH + 0.01) { pt.captured = true; break; }
        const h = Math.min(Math.max(0.004 * r, 0.002), 0.08, budget);
        rk4Step(pt, pt.pt, a, h);
        pt.tau += h;
        budget -= h;
      }
      pt.trail.push([pt.X[0], pt.X[1]]);
      if (pt.trail.length > 2200) pt.trail.shift();
    }
  }

  // ---- drawing ------------------------------------------------------------

  draw() {
    const { ctx, canvas } = this;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth * dpr, h = canvas.clientHeight * dpr;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

    const a = this.spin;
    const cx = w / 2, cy = h / 2;
    const scale = Math.min(w, h) / (2 * this.viewRadius);
    const toScreen = (x, y) => [cx + x * scale, cy - y * scale];

    ctx.fillStyle = "#05060a";
    ctx.fillRect(0, 0, w, h);

    // reference circles: BL radius r on the equator sits at
    // KS Cartesian radius sqrt(r^2 + a^2)
    const circle = (rBL, style, dash, label, labelAngle) => {
      const R = Math.sqrt(rBL * rBL + a * a) * scale;
      ctx.beginPath();
      ctx.setLineDash(dash);
      ctx.strokeStyle = style;
      ctx.lineWidth = 1.2 * dpr;
      ctx.arc(cx, cy, R, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      if (label) {
        ctx.fillStyle = style;
        ctx.font = `${11 * dpr}px system-ui, sans-serif`;
        const lx = cx + (R + 6 * dpr) * Math.cos(labelAngle);
        const ly = cy - (R + 6 * dpr) * Math.sin(labelAngle);
        ctx.fillText(label, lx, ly);
      }
    };

    // event horizon (filled)
    const rH = horizonOuter(a);
    ctx.beginPath();
    ctx.fillStyle = "#000";
    ctx.strokeStyle = "#2dd4bf";
    ctx.lineWidth = 1.5 * dpr;
    ctx.arc(cx, cy, Math.sqrt(rH * rH + a * a) * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    circle(ergoEquator(), "rgba(244,114,182,0.7)", [6 * dpr, 5 * dpr], "ergosphere", 2.4);
    circle(photonOrbit(a, true), "rgba(251,191,36,0.55)", [2 * dpr, 4 * dpr], "photon orbit", -1.1);
    circle(isco(a, true), "rgba(125,211,252,0.6)", [8 * dpr, 4 * dpr], "ISCO", 0.7);

    // spin direction indicator
    ctx.fillStyle = "rgba(148,163,184,0.8)";
    ctx.font = `${11 * dpr}px system-ui, sans-serif`;
    ctx.fillText(`spin a = ${a.toFixed(3)}  (counter-clockwise)`, 10 * dpr, h - 10 * dpr);

    // particles + trails
    for (const p of this.particles) {
      if (p.trail.length > 1) {
        ctx.beginPath();
        for (let i = 0; i < p.trail.length; i++) {
          const [sx, sy] = toScreen(p.trail[i][0], p.trail[i][1]);
          i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
        }
        ctx.strokeStyle = p.color;
        ctx.globalAlpha = 0.55;
        ctx.lineWidth = 1.3 * dpr;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      const [sx, sy] = toScreen(p.X[0], p.X[1]);
      ctx.beginPath();
      ctx.fillStyle = p.captured ? "#334155" : p.color;
      ctx.arc(sx, sy, 4 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }

    // drag-to-launch arrow
    if (this.launch) {
      const [sx, sy] = toScreen(this.launch.x, this.launch.y);
      const [ex, ey] = toScreen(this.launch.x2, this.launch.y2);
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1.5 * dpr;
      ctx.stroke();
    }

    this._updateReadout();
  }

  _updateReadout() {
    if (!this.readout) return;
    const a = this.spin;
    const rows = this.particles.map((p, i) => {
      const r = ksRadius(p.X, a);
      const E = -p.pt;
      const L = p.X[0] * p.p[1] - p.X[1] * p.p[0];
      const dE = Math.abs((E - p.E0) / p.E0);
      const dL = p.L0 !== 0 ? Math.abs((L - p.L0) / p.L0) : Math.abs(L - p.L0);
      return `<div class="row" style="color:${p.color}">
        <span>#${i + 1}${p.captured ? " (captured)" : ""}</span>
        <span>r = ${r.toFixed(2)} M</span>
        <span>E = ${E.toFixed(4)}</span>
        <span>L = ${L.toFixed(3)}</span>
        <span>drift ${Math.max(dE, dL).toExponential(1)}</span>
      </div>`;
    });
    this.readout.innerHTML = rows.join("") ||
      `<div class="row muted">no particles — pick a preset or drag to launch</div>`;
  }

  // ---- interaction ---------------------------------------------------------

  _bindPointer() {
    const toWorld = (ev) => {
      const rect = this.canvas.getBoundingClientRect();
      const cx = rect.width / 2, cy = rect.height / 2;
      const scale = Math.min(rect.width, rect.height) / (2 * this.viewRadius);
      return [
        (ev.clientX - rect.left - cx) / scale,
        -(ev.clientY - rect.top - cy) / scale,
      ];
    };
    this.canvas.addEventListener("pointerdown", (ev) => {
      const [x, y] = toWorld(ev);
      const a = this.spin;
      if (Math.hypot(x, y) < Math.sqrt(horizonOuter(a) ** 2 + a * a)) return;
      this.launch = { x, y, x2: x, y2: y };
      this.canvas.setPointerCapture(ev.pointerId);
    });
    this.canvas.addEventListener("pointermove", (ev) => {
      if (!this.launch) return;
      const [x, y] = toWorld(ev);
      this.launch.x2 = x;
      this.launch.y2 = y;
    });
    this.canvas.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      this.viewRadius = Math.min(40, Math.max(4, this.viewRadius * Math.exp(ev.deltaY * 0.001)));
    }, { passive: false });
    this.canvas.addEventListener("pointerup", () => {
      if (!this.launch) return;
      const { x, y, x2, y2 } = this.launch;
      // drag vector -> coordinate velocity (scaled so a modest drag ~ orbital speed)
      const vx = (x2 - x) * 0.045;
      const vy = (y2 - y) * 0.045;
      this.particles.push(this.makeParticle(x, y, vx, vy, randColor()));
      if (this.particles.length > 8) this.particles.shift();
      this.launch = null;
    });
  }
}

function randColor() {
  const palette = ["#7dd3fc", "#fbbf24", "#f472b6", "#a78bfa", "#4ade80", "#fb923c"];
  return palette[Math.floor(Math.random() * palette.length)];
}
