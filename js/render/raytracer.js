// WebGL2 pipeline for the Kerr ray tracer + the relativistic camera.
// The camera is a static observer: its local frame is an orthonormal tetrad
// built by Gram–Schmidt under the full Kerr metric (kerr.js), so ray
// directions are exact null 4-momenta — no flat-space approximation.

import { VERTEX_SRC } from "./shaders/vertex.js";
import { FRAGMENT_SRC } from "./shaders/fragment.js";
import { horizonOuter, isco, staticTetrad } from "../physics/kerr.js";

export class RayTracer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { antialias: false });
    if (!gl) throw new Error("WebGL2 not available");
    this.gl = gl;

    this.program = buildProgram(gl, VERTEX_SRC, FRAGMENT_SRC);
    gl.useProgram(this.program);
    this.u = {};
    for (const name of [
      "uRes", "uTime", "uSpin", "uHorizon", "uIsco", "uDiskOuter",
      "uDiskOn", "uCinematic", "uHotSpots", "uMaxSteps", "uFocal", "uExposure", "uCamPos",
      "uE0", "uE1", "uE2", "uE3",
    ]) {
      this.u[name] = gl.getUniformLocation(this.program, name);
    }
    gl.createVertexArray && gl.bindVertexArray(gl.createVertexArray());

    // camera state (spherical around the hole; BH spin axis = +z)
    this.distance = 26;
    this.inclination = (82 * Math.PI) / 180; // from spin axis
    this.azimuth = -0.5;

    // physics / quality settings
    this.spin = 0.9;
    this.diskOn = true;
    this.cinematic = true;
    this.hotSpots = true;
    this.diskOuter = 16;
    this.maxSteps = 320;
    this.focal = 1.6;
    this.exposure = 1.15;
    this.renderScale = 0.6;
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr * this.renderScale));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr * this.renderScale));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  cameraPosition() {
    const si = Math.sin(this.inclination), ci = Math.cos(this.inclination);
    return [
      this.distance * si * Math.cos(this.azimuth),
      this.distance * si * Math.sin(this.azimuth),
      this.distance * ci,
    ];
  }

  render(timeSec) {
    this.resize();
    const gl = this.gl;
    const a = this.spin;
    const pos = this.cameraPosition();

    // Euclidean look-at pre-basis, then orthonormalized under the Kerr metric
    const len = Math.hypot(...pos);
    const fwd = pos.map((c) => -c / len);
    let up = [0, 0, 1];
    // remove forward component, guard near-pole degeneracy
    const d = up[0] * fwd[0] + up[1] * fwd[1] + up[2] * fwd[2];
    up = up.map((c, i) => c - d * fwd[i]);
    const un = Math.hypot(...up) || 1;
    up = up.map((c) => c / un);

    const { e0, e1, e2, e3 } = staticTetrad(pos, a, fwd, up);

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.program);
    gl.uniform2f(this.u.uRes, this.canvas.width, this.canvas.height);
    gl.uniform1f(this.u.uTime, timeSec);
    gl.uniform1f(this.u.uSpin, a);
    gl.uniform1f(this.u.uHorizon, horizonOuter(a));
    gl.uniform1f(this.u.uIsco, isco(a, true));
    gl.uniform1f(this.u.uDiskOuter, this.diskOuter);
    gl.uniform1f(this.u.uDiskOn, this.diskOn ? 1 : 0);
    gl.uniform1f(this.u.uCinematic, this.cinematic ? 1 : 0);
    gl.uniform1f(this.u.uHotSpots, this.hotSpots ? 1 : 0);
    gl.uniform1i(this.u.uMaxSteps, this.maxSteps);
    gl.uniform1f(this.u.uFocal, this.focal);
    gl.uniform1f(this.u.uExposure, this.exposure);
    gl.uniform3f(this.u.uCamPos, pos[0], pos[1], pos[2]);
    gl.uniform4f(this.u.uE0, ...e0);
    gl.uniform4f(this.u.uE1, ...e1);
    gl.uniform4f(this.u.uE2, ...e2);
    gl.uniform4f(this.u.uE3, ...e3);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

function buildProgram(gl, vsSrc, fsSrc) {
  const compile = (type, src) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error("Shader compile error:\n" + gl.getShaderInfoLog(sh));
    }
    return sh;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("Program link error:\n" + gl.getProgramInfoLog(prog));
  }
  return prog;
}
