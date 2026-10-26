// Geodesic integration in Kerr–Schild coordinates via the super-Hamiltonian
//
//   H(x, p) = 1/2 g^{munu} p_mu p_nu = 1/2 [ eta^{munu} p_mu p_nu - f (k^mu p_mu)^2 ]
//
// (g^{munu} = eta^{munu} - f k^mu k^nu is exact because k is null.)
// Hamilton's equations give the geodesic:
//
//   dx^i/dlam = p_i - f k_i (k.p)
//   dp_i/dlam = 1/2 (d_i f)(k.p)^2 + f (k.p) (d_i k_j) p_j
//
// The metric is stationary and axisymmetric, so p_t (energy) and
// L = x p_y - y p_x (angular momentum) are conserved — we use both as
// live accuracy checks on the integration.

/**
 * Right-hand side of the geodesic equations.
 * X = [x,y,z], p = [px,py,pz] covariant, pt = conserved p_t, a = spin.
 * Writes derivatives into dX, dp (length-3 arrays).
 */
export function geodesicRHS(X, p, pt, a, dX, dp) {
  const [x, y, z] = X;
  const rho2 = x * x + y * y + z * z;
  const b = rho2 - a * a;
  const s = Math.sqrt(b * b + 4 * a * a * z * z);
  const r2 = 0.5 * (b + s);
  const r = Math.sqrt(Math.max(r2, 1e-12));
  const D = Math.max(2 * r2 - b, 1e-12); // == s

  // grad r
  const grx = (x * r) / D;
  const gry = (y * r) / D;
  const grz = (z * (r2 + a * a)) / (r * D);

  const R2 = r2 + a * a;
  const kx = (r * x + a * y) / R2;
  const ky = (r * y - a * x) / R2;
  const kz = z / r;

  const W = r2 * r2 + a * a * z * z;
  const f = (2 * r2 * r) / W;

  // grad f = 6 r^2 grad(r)/W - f grad(W)/W,  grad W = 4 r^3 grad r + (0,0,2 a^2 z)
  const c1 = (6 * r2) / W;
  const c2 = f / W;
  const gWx = 4 * r2 * r * grx;
  const gWy = 4 * r2 * r * gry;
  const gWz = 4 * r2 * r * grz + 2 * a * a * z;
  const gfx = c1 * grx - c2 * gWx;
  const gfy = c1 * gry - c2 * gWy;
  const gfz = c1 * grz - c2 * gWz;

  // grad of each k component (product rule with grad r)
  const inv = 1 / R2;
  const tr = (2 * r) * inv;
  const gkx_x = (grx * x + r) * inv - kx * tr * grx;
  const gkx_y = (gry * x + a) * inv - kx * tr * gry;
  const gkx_z = (grz * x) * inv - kx * tr * grz;
  const gky_x = (grx * y - a) * inv - ky * tr * grx;
  const gky_y = (gry * y + r) * inv - ky * tr * gry;
  const gky_z = (grz * y) * inv - ky * tr * grz;
  const gkz_x = -(z / r2) * grx;
  const gkz_y = -(z / r2) * gry;
  const gkz_z = 1 / r - (z / r2) * grz;

  // k^mu p_mu  (k^t = -1 after raising with eta)
  const kp = -pt + kx * p[0] + ky * p[1] + kz * p[2];

  dX[0] = p[0] - f * kx * kp;
  dX[1] = p[1] - f * ky * kp;
  dX[2] = p[2] - f * kz * kp;

  const h = f * kp;
  dp[0] = 0.5 * gfx * kp * kp + h * (gkx_x * p[0] + gky_x * p[1] + gkz_x * p[2]);
  dp[1] = 0.5 * gfy * kp * kp + h * (gkx_y * p[0] + gky_y * p[1] + gkz_y * p[2]);
  dp[2] = 0.5 * gfz * kp * kp + h * (gkx_z * p[0] + gky_z * p[1] + gkz_z * p[2]);
}

/** Classic RK4 step of size h on state {X, p} with conserved pt. */
export function rk4Step(state, pt, a, h) {
  const { X, p } = state;
  const k1x = [0, 0, 0], k1p = [0, 0, 0];
  const k2x = [0, 0, 0], k2p = [0, 0, 0];
  const k3x = [0, 0, 0], k3p = [0, 0, 0];
  const k4x = [0, 0, 0], k4p = [0, 0, 0];
  const tX = [0, 0, 0], tp = [0, 0, 0];

  geodesicRHS(X, p, pt, a, k1x, k1p);
  for (let i = 0; i < 3; i++) { tX[i] = X[i] + 0.5 * h * k1x[i]; tp[i] = p[i] + 0.5 * h * k1p[i]; }
  geodesicRHS(tX, tp, pt, a, k2x, k2p);
  for (let i = 0; i < 3; i++) { tX[i] = X[i] + 0.5 * h * k2x[i]; tp[i] = p[i] + 0.5 * h * k2p[i]; }
  geodesicRHS(tX, tp, pt, a, k3x, k3p);
  for (let i = 0; i < 3; i++) { tX[i] = X[i] + h * k3x[i]; tp[i] = p[i] + h * k3p[i]; }
  geodesicRHS(tX, tp, pt, a, k4x, k4p);

  for (let i = 0; i < 3; i++) {
    X[i] += (h / 6) * (k1x[i] + 2 * k2x[i] + 2 * k3x[i] + k4x[i]);
    p[i] += (h / 6) * (k1p[i] + 2 * k2p[i] + 2 * k3p[i] + k4p[i]);
  }
}
