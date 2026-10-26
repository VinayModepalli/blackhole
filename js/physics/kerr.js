// Kerr metric in Cartesian Kerr–Schild (KS) form, geometrized units G = c = M = 1.
//
//   g_munu = eta_munu + f * k_mu k_nu
//
//   f   = 2 r^3 / (r^4 + a^2 z^2)
//   k_mu = (1, (r x + a y)/(r^2+a^2), (r y - a x)/(r^2+a^2), z/r)   (null vector)
//
// where r is the Boyer–Lindquist radius, defined implicitly by
//   x^2 + y^2 + z^2 = r^2 + a^2 (1 - z^2/r^2).
//
// This form is an EXACT solution of Einstein's vacuum field equations
// (Kerr 1963), regular at the horizon — ideal for numerical integration.

/** Boyer–Lindquist radius r from KS Cartesian position. */
export function ksRadius(X, a) {
  const rho2 = X[0] * X[0] + X[1] * X[1] + X[2] * X[2];
  const b = rho2 - a * a;
  const r2 = 0.5 * (b + Math.sqrt(b * b + 4 * a * a * X[2] * X[2]));
  return Math.sqrt(Math.max(r2, 1e-12));
}

/** Scalar f and spatial part of the null vector k at position X. k_t = 1. */
export function fAndK(X, a) {
  const [x, y, z] = X;
  const r = ksRadius(X, a);
  const r2 = r * r;
  const R2 = r2 + a * a;
  const f = (2 * r2 * r) / (r2 * r2 + a * a * z * z);
  const k = [(r * x + a * y) / R2, (r * y - a * x) / R2, z / r];
  return { r, f, k };
}

/** Metric inner product g(u, v) of two contravariant 4-vectors at X. */
export function metricDot(u, v, X, a) {
  const { f, k } = fAndK(X, a);
  // eta(u,v) with signature (-,+,+,+)
  const eta = -u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3];
  // k_mu u^mu  (k_t = 1)
  const ku = u[0] + k[0] * u[1] + k[1] * u[2] + k[2] * u[3];
  const kv = v[0] + k[0] * v[1] + k[1] * v[2] + k[2] * v[3];
  return eta + f * ku * kv;
}

/** Lower an index: p_mu = g_munu u^nu. Returns covariant components. */
export function lower(u, X, a) {
  const { f, k } = fAndK(X, a);
  const ku = u[0] + k[0] * u[1] + k[1] * u[2] + k[2] * u[3];
  return [
    -u[0] + f * 1 * ku,
    u[1] + f * k[0] * ku,
    u[2] + f * k[1] * ku,
    u[3] + f * k[2] * ku,
  ];
}

/** Outer event horizon radius r+ = 1 + sqrt(1 - a^2). */
export function horizonOuter(a) {
  return 1 + Math.sqrt(Math.max(0, 1 - a * a));
}

/** Ergosphere radius on the equator (always r = 2M). */
export function ergoEquator() {
  return 2;
}

/**
 * Innermost stable circular orbit (Bardeen, Press & Teukolsky 1972).
 * prograde = true for orbits co-rotating with the spin.
 */
export function isco(a, prograde = true) {
  const s = Math.abs(a);
  const Z1 =
    1 +
    Math.cbrt(1 - s * s) * (Math.cbrt(1 + s) + Math.cbrt(1 - s));
  const Z2 = Math.sqrt(3 * s * s + Z1 * Z1);
  const sign = prograde ? -1 : +1;
  return 3 + Z2 + sign * Math.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2));
}

/** Equatorial circular photon orbit radius (prograde). */
export function photonOrbit(a, prograde = true) {
  const s = prograde ? a : -a;
  return 2 * (1 + Math.cos((2 / 3) * Math.acos(-s)));
}

/**
 * Angular velocity dphi/dt of a circular equatorial orbit at BL radius r:
 *   Omega = +-1 / (r^{3/2} +- a)      (Kepler's law, GR version)
 */
export function omegaCirc(r, a, prograde = true) {
  return prograde ? 1 / (Math.pow(r, 1.5) + a) : -1 / (Math.pow(r, 1.5) - a);
}

/**
 * Build an orthonormal tetrad {e0,e1,e2,e3} for a STATIC observer at X,
 * via Gram–Schmidt under the full Kerr metric. e0 is the observer's
 * 4-velocity; e1,e2,e3 span the local rest frame (right, up, forward).
 * Valid outside the ergosphere.
 */
export function staticTetrad(X, a, forward, up) {
  const pos = X;
  const g = (u, v) => metricDot(u, v, pos, a);

  // e0: static observer, u = (1,0,0,0)/sqrt(-g_tt)
  const T = [1, 0, 0, 0];
  const gtt = g(T, T);
  if (gtt >= 0) throw new Error("camera inside ergosphere — no static observer");
  const e0 = T.map((c) => c / Math.sqrt(-gtt));

  const spatial = (v3) => [0, v3[0], v3[1], v3[2]];
  const projOut = (w, e, sgn) => w.map((c, i) => c - sgn * g(w, e) * e[i]);
  const normalize = (w) => {
    const n = Math.sqrt(g(w, w));
    return w.map((c) => c / n);
  };

  // Forward leg (project out e0; g(e0,e0) = -1 so the projection ADDS g(w,e0)e0)
  let w3 = spatial(forward).map((c, i) => c + g(spatial(forward), e0) * e0[i]);
  const e3 = normalize(w3);
  // Right leg
  const right = spatial(crossN(forward, up));
  let r4 = right.map((c, i) => c + g(right, e0) * e0[i]);
  r4 = projOut(r4, e3, +1);
  const e1 = normalize(r4);
  // Up leg
  let u4 = spatial(up).map((c, i) => c + g(spatial(up), e0) * e0[i]);
  u4 = projOut(u4, e3, +1);
  u4 = projOut(u4, e1, +1);
  const e2 = normalize(u4);

  return { e0, e1, e2, e3 };
}

function crossN(a, b) {
  const c = [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const n = Math.hypot(c[0], c[1], c[2]) || 1;
  return [c[0] / n, c[1] / n, c[2] / n];
}
