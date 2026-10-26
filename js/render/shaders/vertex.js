// Fullscreen-triangle vertex shader — all the physics lives in the fragment stage.
export const VERTEX_SRC = `#version 300 es
void main() {
  // oversized triangle covering the viewport, no buffers needed
  vec2 v = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(v * 2.0 - 1.0, 0.0, 1.0);
}
`;
