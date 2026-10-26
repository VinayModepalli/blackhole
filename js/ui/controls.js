// Wires DOM controls to the ray tracer and orbit sim.
import { horizonOuter, isco, ergoEquator, photonOrbit } from "../physics/kerr.js";

export function initControls({ rt, orbits, onTabChange }) {
  const $ = (id) => document.getElementById(id);

  // ---- tabs ----
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((t) =>
    t.addEventListener("click", () => {
      tabs.forEach((x) => x.classList.toggle("active", x === t));
      document.querySelectorAll(".view").forEach((v) =>
        v.classList.toggle("active", v.id === t.dataset.view));
      onTabChange(t.dataset.view);
    }));

  // ---- shared spin slider ----
  const spin = $("spin"), spinVal = $("spinVal");
  const applySpin = () => {
    const a = parseFloat(spin.value);
    spinVal.textContent = a.toFixed(3);
    rt.spin = a;
    orbits.spin = a;
    orbits.particles = [];   // old momenta belong to a different spacetime
    updateReadout(a);
  };
  spin.addEventListener("input", applySpin);

  // ---- ray tracer controls ----
  bindRange($("quality"), (v) => { rt.renderScale = v; rt.resize(); });
  bindRange($("steps"), (v) => { rt.maxSteps = Math.round(v); });
  bindRange($("fov"), (v) => { rt.focal = v; });
  $("disk").addEventListener("change", (e) => { rt.diskOn = e.target.checked; });
  $("cine").addEventListener("change", (e) => { rt.cinematic = e.target.checked; });
  $("spots").addEventListener("change", (e) => { rt.hotSpots = e.target.checked; });
  $("pause").addEventListener("change", (e) => { rt.paused = e.target.checked; });

  // drag = orbit camera, wheel = zoom
  const cv = rt.canvas;
  let drag = null;
  cv.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, inc: rt.inclination, az: rt.azimuth };
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointermove", (e) => {
    if (!drag) return;
    rt.azimuth = drag.az - (e.clientX - drag.x) * 0.005;
    rt.inclination = clamp(drag.inc - (e.clientY - drag.y) * 0.005, 0.05, Math.PI - 0.05);
  });
  cv.addEventListener("pointerup", () => (drag = null));
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    // static-observer camera must stay outside the ergosphere
    rt.distance = clamp(rt.distance * Math.exp(e.deltaY * 0.001), 4.5, 55);
  }, { passive: false });

  // ---- orbit lab controls ----
  document.querySelectorAll("[data-preset]").forEach((b) =>
    b.addEventListener("click", () => orbits.preset(b.dataset.preset)));
  bindRange($("orbitSpeed"), (v) => { orbits.speed = v; });
  $("clearOrbits").addEventListener("click", () => { orbits.particles = []; });

  // ---- characteristic radii readout ----
  function updateReadout(a) {
    $("rH").textContent = horizonOuter(a).toFixed(3);
    $("rIsco").textContent = isco(a, true).toFixed(3);
    $("rIscoR").textContent = isco(a, false).toFixed(3);
    $("rPh").textContent = photonOrbit(a, true).toFixed(3);
    $("rErgo").textContent = ergoEquator().toFixed(1);
  }

  applySpin();
}

function bindRange(el, fn) {
  const apply = () => fn(parseFloat(el.value));
  el.addEventListener("input", apply);
  apply();
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}
