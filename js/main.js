import { RayTracer } from "./render/raytracer.js";
import { OrbitSim } from "./orbits/orbitsim.js";
import { initControls } from "./ui/controls.js";

const rtCanvas = document.getElementById("rtCanvas");
const orbitCanvas = document.getElementById("orbitCanvas");

let rt;
try {
  rt = new RayTracer(rtCanvas);
} catch (err) {
  document.getElementById("glError").textContent =
    "Could not start WebGL2 renderer: " + err.message;
  document.getElementById("glError").style.display = "block";
  throw err;
}

const orbits = new OrbitSim(orbitCanvas, document.getElementById("orbitReadout"));
window.rt = rt;          // debug handles
window.orbits = orbits;

let activeView = "rayView";
initControls({ rt, orbits, onTabChange: (v) => (activeView = v) });

let last = performance.now();
let simTime = 0;

function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;

  if (activeView === "rayView") {
    if (!rt.paused) simTime += dt;
    rt.render(simTime);
  } else {
    orbits.step(dt);
    orbits.draw();
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
