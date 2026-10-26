// Fragment shader: general-relativistic ray tracing in the Kerr spacetime.
// Mirrors js/physics/geodesic.js — same Kerr–Schild Hamiltonian formulation,
// integrated per pixel with RK4. Nothing here is a "warp effect": lensing,
// the shadow, the photon ring, Doppler beaming and gravitational redshift
// all emerge from the geodesic equations and the disk's orbital kinematics.

export const FRAGMENT_SRC = `#version 300 es
precision highp float;

uniform vec2  uRes;
uniform float uTime;
uniform float uSpin;       // a  (M = 1)
uniform float uHorizon;    // r+ = 1 + sqrt(1 - a^2)
uniform float uIsco;       // prograde ISCO radius (disk inner edge)
uniform float uDiskOuter;
uniform float uDiskOn;
uniform float uCinematic;  // 1 = Gargantua look: warm gold, beaming muted (as the movie did)
uniform float uHotSpots;   // 1 = bright blobs on Keplerian orbits (cf. GRAVITY's Sgr A* hot spots)
uniform int   uMaxSteps;
uniform float uFocal;      // camera focal length (fov control)
uniform float uExposure;
uniform vec3  uCamPos;
// Orthonormal tetrad of the static camera observer (contravariant 4-vectors,
// built CPU-side by Gram-Schmidt under the full metric): e0 = 4-velocity,
// e1 = right, e2 = up, e3 = forward.
uniform vec4  uE0, uE1, uE2, uE3;

out vec4 fragColor;

const float ESCAPE_R = 55.0;

// ---------- Kerr-Schild geometry ----------

float ksR(vec3 X, float a) {
  float rho2 = dot(X, X);
  float b = rho2 - a * a;
  float r2 = 0.5 * (b + sqrt(b * b + 4.0 * a * a * X.z * X.z));
  return sqrt(max(r2, 1e-10));
}

void fk(vec3 X, float a, out float f, out vec3 kv) {
  float r = ksR(X, a);
  float r2 = r * r;
  float R2 = r2 + a * a;
  f = 2.0 * r2 * r / (r2 * r2 + a * a * X.z * X.z);
  kv = vec3((r * X.x + a * X.y) / R2, (r * X.y - a * X.x) / R2, X.z / r);
}

// Geodesic RHS from H = 1/2 [eta^{mn} p_m p_n - f (k.p)^2].
// X: position, p: covariant spatial momentum, pt: conserved p_t.
void geoRHS(vec3 X, vec3 p, float pt, float a, out vec3 dX, out vec3 dp) {
  float x = X.x, y = X.y, z = X.z;
  float rho2 = dot(X, X);
  float b = rho2 - a * a;
  float s = sqrt(b * b + 4.0 * a * a * z * z);
  float r2 = 0.5 * (b + s);
  float r = sqrt(max(r2, 1e-10));
  float D = max(2.0 * r2 - b, 1e-10);

  vec3 gr = vec3(x * r / D, y * r / D, z * (r2 + a * a) / (r * D));

  float R2 = r2 + a * a;
  vec3 kv = vec3((r * x + a * y) / R2, (r * y - a * x) / R2, z / r);

  float W = r2 * r2 + a * a * z * z;
  float f = 2.0 * r2 * r / W;
  vec3 gW = 4.0 * r2 * r * gr + vec3(0.0, 0.0, 2.0 * a * a * z);
  vec3 gf = (6.0 * r2 / W) * gr - (f / W) * gW;

  float inv = 1.0 / R2;
  vec3 tr = (2.0 * r * inv) * gr;
  // gradient of each k component
  vec3 gkx = (gr * x + vec3(r, a, 0.0)) * inv - kv.x * tr;
  vec3 gky = (gr * y + vec3(-a, r, 0.0)) * inv - kv.y * tr;
  vec3 gkz = vec3(0.0, 0.0, 1.0 / r) - (z / r2) * gr;

  float kp = -pt + dot(kv, p);   // k^mu p_mu  (k^t = -1)

  dX = p - f * kv * kp;
  dp = 0.5 * gf * kp * kp + (f * kp) * (gkx * p.x + gky * p.y + gkz * p.z);
}

void rk4(inout vec3 X, inout vec3 p, float pt, float a, float h) {
  vec3 k1x, k1p, k2x, k2p, k3x, k3p, k4x, k4p;
  geoRHS(X, p, pt, a, k1x, k1p);
  geoRHS(X + 0.5 * h * k1x, p + 0.5 * h * k1p, pt, a, k2x, k2p);
  geoRHS(X + 0.5 * h * k2x, p + 0.5 * h * k2p, pt, a, k3x, k3p);
  geoRHS(X + h * k3x, p + h * k3p, pt, a, k4x, k4p);
  X += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
  p += (h / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
}

// ---------- noise (disk turbulence, background) ----------

float hash13(vec3 q) {
  q = fract(q * 0.1031);
  q += dot(q, q.zyx + 31.32);
  return fract((q.x + q.y) * q.z);
}

float vnoise(vec3 q) {
  vec3 i = floor(q), fr = fract(q);
  vec3 u = fr * fr * (3.0 - 2.0 * fr);
  return mix(
    mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), u.x),
        mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), u.x), u.y),
    mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), u.x),
        mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), u.x), u.y),
    u.z);
}

float fbm(vec3 q) {
  float v = 0.0, w = 0.5;
  for (int i = 0; i < 4; i++) { v += w * vnoise(q); q *= 2.03; w *= 0.5; }
  return v;
}

// Approximate blackbody chromaticity for temperature T in kelvin.
vec3 blackbody(float T) {
  T = clamp(T, 1000.0, 40000.0) / 100.0;
  vec3 c;
  c.r = T <= 66.0 ? 1.0 : clamp(1.2929 * pow(T - 60.0, -0.1332), 0.0, 1.0);
  c.g = T <= 66.0 ? clamp(0.3900 * log(T) - 0.6318, 0.0, 1.0)
                  : clamp(1.1299 * pow(T - 60.0, -0.0755), 0.0, 1.0);
  c.b = T >= 66.0 ? 1.0 : (T <= 19.0 ? 0.0
                  : clamp(0.5432 * log(T - 10.0) - 1.1962, 0.0, 1.0));
  return c;
}

// Cinematic color grade: ember -> gold -> hot pale core (Gargantua palette).
// linear-space values chosen so they read as ember/gold AFTER display gamma
vec3 goldRamp(float t) {
  vec3 ember = vec3(1.0, 0.19, 0.01);
  vec3 gold  = vec3(1.0, 0.53, 0.08);
  vec3 core  = vec3(1.0, 0.75, 0.36);
  return t < 0.6 ? mix(ember, gold, t / 0.6)
                 : mix(gold, core, clamp((t - 0.6) / 0.55, 0.0, 1.0));
}

// ---------- accretion disk emission ----------
// Thin disk on the equatorial plane between the ISCO and uDiskOuter.
// Matter follows circular geodesics with Omega = 1/(r^{3/2} + a).
// The redshift factor g = E_obs / E_emitted is computed EXACTLY from the
// photon momentum and the emitter 4-velocity; observed intensity scales as
// g^4 (relativistic invariant I_nu/nu^3, frequency-integrated).

vec3 diskEmission(vec3 Xc, vec3 p, float pt, float a, out float alpha) {
  float r = ksR(Xc, a);
  float f; vec3 kv; fk(Xc, a, f, kv);

  float Om = 1.0 / (pow(r, 1.5) + a);           // prograde Keplerian
  // emitter 4-velocity u = ut * (1, -Om*y, Om*x, 0), normalized with FULL metric
  float etaVV = -1.0 + Om * Om * (Xc.x * Xc.x + Xc.y * Xc.y);
  float kV = 1.0 + Om * (Xc.x * kv.y - Xc.y * kv.x);
  float n2 = etaVV + f * kV * kV;
  float ut = 1.0 / sqrt(max(-n2, 1e-6));

  // photon energy measured by emitter: E_em = -p_mu u^mu
  float Eem = -ut * (pt + Om * (Xc.x * p.y - Xc.y * p.x));
  float g = 1.0 / max(Eem, 1e-4);               // camera-frame energy is 1 by construction

  // Novikov-Thorne-style radial profile: T(r) ~ [ (1 - sqrt(r_isco/r)) / r^3 ]^{1/4}
  float prof = pow(max(1.0 - sqrt(uIsco / r), 0.0) / (r * r * r), 0.25);
  float rPeak = uIsco * 49.0 / 36.0;            // analytic peak of the profile
  float profPeak = pow(max(1.0 - sqrt(uIsco / rPeak), 0.0) / (rPeak * rPeak * rPeak), 0.25);
  float Tn = prof / profPeak;                    // normalized temperature, peak = 1

  // co-rotating texture, mildly banded so shear reads as swirl, not as
  // radial "waves" (high radial frequency + differential rotation looked
  // like ripples radiating outward — pure texture artifact)
  float phi = atan(Xc.y, Xc.x) - Om * uTime * 9.0;
  float lanes = fbm(vec3(cos(phi), sin(phi), 0.0) * 2.5
                    + vec3(0.0, 0.0, 4.0 * log(r)));
  float turb = 0.45 + 1.1 * lanes;

  // hot spots: over-dense blobs on circular Keplerian orbits at their own
  // local Omega (inner ones visibly lap outer ones). Their light picks up
  // the full redshift/beaming treatment below like everything else.
  if (uHotSpots > 0.5) {
    for (int s = 0; s < 3; s++) {
      float Rs = uIsco * (1.25 + 0.7 * float(s));
      float Oms = 1.0 / (pow(Rs, 1.5) + a);
      float ang = Oms * uTime * 9.0 + float(s) * 2.1;
      vec2 spot = sqrt(Rs * Rs + a * a) * vec2(cos(ang), sin(ang));
      float d2 = dot(Xc.xy - spot, Xc.xy - spot);
      turb += 2.8 * exp(-d2 / 0.35);
    }
  }

  // cinematic mode mutes beaming and color shift, exactly the liberty
  // the Interstellar team took; physics mode applies them exactly
  float gI = mix(g, 1.0, uCinematic * 0.80);     // intensity beaming
  float gC = mix(g, 1.0, uCinematic * 0.90);     // color shift
  float g4 = gI * gI * gI * gI;
  float edge = smoothstep(uDiskOuter, uDiskOuter * 0.8, r);
  float radial = pow(Tn, mix(3.5, 1.5, uCinematic));
  float I = g4 * radial * turb * edge * mix(2.2, 2.6, uCinematic);
  // soft rolloff: bright regions stay golden instead of clipping to white
  I = I / (1.0 + uCinematic * 0.7 * I);

  vec3 physColor = blackbody(7200.0 * Tn * gC);
  vec3 cineColor = goldRamp(Tn * gC * (0.4 + 0.4 * turb));
  alpha = clamp(0.45 + 0.65 * Tn, 0.0, 0.95) * edge;
  return mix(physColor, cineColor, uCinematic) * I;
}

// ---------- background starfield ----------

vec3 background(vec3 dir) {
  vec3 col = vec3(0.0);
  // two layers of hash stars
  for (int L = 0; L < 2; L++) {
    float S = L == 0 ? 90.0 : 220.0;
    vec3 cell = floor(dir * S);
    float h = hash13(cell);
    if (h > 0.997) {
      vec3 cc = (cell + 0.5) / S;
      float d = length(normalize(cc) - dir) * S;
      float star = exp(-d * d * 12.0) * (0.4 + 12.0 * pow(fract(h * 91.7), 4.0));
      float tint = fract(h * 57.3);
      col += star * mix(vec3(0.72, 0.80, 1.0), vec3(1.0, 0.82, 0.62), tint) * 0.55;
    }
  }
  // faint nebula + galactic band
  float neb = fbm(dir * 4.0);
  vec3 bandN = normalize(vec3(0.35, 0.15, 0.92));
  float band = exp(-14.0 * abs(dot(dir, bandN)));
  col += band * (0.012 + 0.05 * neb) * vec3(0.55, 0.6, 0.85);
  col += 0.004 * neb * vec3(0.4, 0.45, 0.7);
  return col;
}

// ---------- main ----------

void main() {
  float a = uSpin;
  vec2 ndc = (2.0 * gl_FragCoord.xy - uRes) / uRes.y;

  // Local ray direction in the camera's orthonormal frame
  vec3 d = normalize(vec3(ndc.x, ndc.y, uFocal));
  // Null 4-momentum: p^mu = e0 + d.x e1 + d.y e2 + d.z e3
  vec4 P4 = uE0 + d.x * uE1 + d.y * uE2 + d.z * uE3;

  // Lower the index at the camera: p_mu = eta_munu P^nu + f k_mu (k_nu P^nu)
  vec3 X = uCamPos;
  float f0; vec3 k0; fk(X, a, f0, k0);
  float kP = P4.x + dot(k0, P4.yzw);
  float pt = -P4.x + f0 * kP;         // conserved along the geodesic
  vec3 p = P4.yzw + f0 * k0 * kP;

  vec3 col = vec3(0.0);
  float trans = 1.0;                   // transmittance through disk crossings
  bool escaped = false;
  vec3 escDir = vec3(0.0, 0.0, 1.0);

  for (int i = 0; i < 600; i++) {
    if (i >= uMaxSteps) break;
    float r = ksR(X, a);

    // captured: piled up on the horizon
    if (r < uHorizon + 0.02) { break; }

    // escaped outward
    if (dot(X, X) > ESCAPE_R * ESCAPE_R && dot(X, p) > 0.0) {
      vec3 v, dumm;
      geoRHS(X, p, pt, a, v, dumm);
      escDir = normalize(v);
      escaped = true;
      break;
    }

    // adaptive step: fine near the photon shell, coarse far away
    float h = clamp(0.05 * (r - 0.9 * uHorizon), 0.02, 0.45);

    vec3 Xprev = X;
    vec3 pPrev = p;
    rk4(X, p, pt, a, h);

    // disk crossing: z changed sign inside [isco, outer]
    if (uDiskOn > 0.5 && Xprev.z * X.z < 0.0) {
      float t = Xprev.z / (Xprev.z - X.z);
      vec3 Xc = mix(Xprev, X, t);
      vec3 pc = mix(pPrev, p, t);
      Xc.z = 0.0;
      float rc = ksR(Xc, a);
      if (rc > uIsco && rc < uDiskOuter) {
        float alpha;
        vec3 e = diskEmission(Xc, pc, pt, a, alpha);
        col += trans * alpha * e;
        trans *= (1.0 - alpha);
        if (trans < 0.02) break;
      }
    }
  }

  if (escaped) col += trans * background(escDir);

  // exposure tone map + gamma
  col = 1.0 - exp(-uExposure * col);
  col = pow(col, vec3(0.4545));
  fragColor = vec4(col, 1.0);
}
`;
