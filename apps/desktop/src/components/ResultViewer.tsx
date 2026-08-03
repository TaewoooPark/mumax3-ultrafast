import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { desktop } from "../api";
import type { ResultSet, VectorFrame } from "../types";
import { GlassButton } from "./GlassPrimitives";

type GlyphMode = "arrows" | "cuboids";
type ColorMode = "direction" | "magnitude";

interface SceneRuntime {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  fieldGroup: THREE.Group | null;
  resizeObserver: ResizeObserver;
  animationFrame: number;
}

interface ResultViewerProps {
  resultSet: ResultSet;
  onBack: () => void;
}

function disposeGroup(group: THREE.Group) {
  group.traverse((object) => {
    if (!(object instanceof THREE.Mesh || object instanceof THREE.LineSegments)) return;
    object.geometry.dispose();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach((material) => material.dispose());
  });
}

function glyphColor(color: THREE.Color, mode: ColorMode, vector: THREE.Vector3, magnitude: number, maxMagnitude: number) {
  if (mode === "magnitude") {
    const normalized = maxMagnitude > 0 ? Math.min(1, magnitude / maxMagnitude) : 0;
    color.setHSL(0.64 - normalized * 0.64, 0.84, 0.52);
    return color;
  }
  const hue = (Math.atan2(vector.z, vector.x) / (Math.PI * 2) + 1) % 1;
  color.setHSL(hue, 0.82, 0.49 + Math.max(-0.08, Math.min(0.08, vector.y * 0.08)));
  return color;
}

export function ResultViewer({ resultSet, onBack }: ResultViewerProps) {
  const quantities = useMemo(
    () => Array.from(new Set(resultSet.frames.map((item) => item.quantity))),
    [resultSet.frames],
  );
  const defaultQuantity = quantities.includes("m") ? "m" : quantities[0] || "";
  const [quantity, setQuantity] = useState(defaultQuantity);
  const frames = useMemo(
    () => resultSet.frames.filter((item) => item.quantity === quantity),
    [quantity, resultSet.frames],
  );
  const viewportRef = useRef<HTMLDivElement>(null);
  const sceneRuntimeRef = useRef<SceneRuntime | null>(null);
  const frameCacheRef = useRef(new Map<string, VectorFrame>());
  const [frameIndex, setFrameIndex] = useState(Math.max(0, frames.length - 1));
  const [frame, setFrame] = useState<VectorFrame | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(10);
  const [glyphMode, setGlyphMode] = useState<GlyphMode>("arrows");
  const [colorMode, setColorMode] = useState<ColorMode>("direction");
  const [glyphScale, setGlyphScale] = useState(1);

  useEffect(() => {
    const container = viewportRef.current;
    if (!container) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0xf3f3f3, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0xf3f3f3, 5.2, 9);
    const camera = new THREE.PerspectiveCamera(38, 1, 0.005, 100);
    camera.position.set(1.8, 1.45, 1.9);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.target.set(0, 0, 0);
    controls.minDistance = 0.18;
    controls.maxDistance = 12;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x6d6d6d, 2.4));
    const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
    keyLight.position.set(-2, 4, 3);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0xbfd5ff, 1.1);
    rimLight.position.set(3, 1, -4);
    scene.add(rimLight);

    const resizeObserver = new ResizeObserver(() => {
      const { width, height } = container.getBoundingClientRect();
      renderer.setSize(Math.max(1, width), Math.max(1, height), false);
      camera.aspect = Math.max(1, width) / Math.max(1, height);
      camera.updateProjectionMatrix();
    });
    resizeObserver.observe(container);

    const render = () => {
      controls.update();
      renderer.render(scene, camera);
      const runtime = sceneRuntimeRef.current;
      if (runtime) runtime.animationFrame = window.requestAnimationFrame(render);
    };
    const runtime: SceneRuntime = {
      renderer,
      scene,
      camera,
      controls,
      fieldGroup: null,
      resizeObserver,
      animationFrame: 0,
    };
    sceneRuntimeRef.current = runtime;
    runtime.animationFrame = window.requestAnimationFrame(render);

    return () => {
      window.cancelAnimationFrame(runtime.animationFrame);
      resizeObserver.disconnect();
      controls.dispose();
      if (runtime.fieldGroup) disposeGroup(runtime.fieldGroup);
      renderer.dispose();
      renderer.domElement.remove();
      sceneRuntimeRef.current = null;
    };
  }, []);

  useEffect(() => {
    setFrameIndex(Math.max(0, frames.length - 1));
    setPlaying(false);
    setFrame(null);
  }, [frames.length, quantity]);

  useEffect(() => {
    const selected = frames[frameIndex];
    if (!selected) {
      setLoading(false);
      setError("No OVF frames were produced by this simulation.");
      return;
    }
    const cached = frameCacheRef.current.get(selected.fileName);
    if (cached) {
      setFrame(cached);
      setLoading(false);
      setError("");
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError("");
    void desktop.loadResultFrame(selected.fileName)
      .then((next) => {
        frameCacheRef.current.set(selected.fileName, next);
        if (!cancelled) setFrame(next);
      })
      .catch((reason) => {
        if (!cancelled) setError(String(reason));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [frameIndex, frames]);

  useEffect(() => {
    if (!playing || loading || frames.length < 2) return;
    const timer = window.setInterval(() => {
      setFrameIndex((current) => (current + 1) % frames.length);
    }, 1000 / fps);
    return () => window.clearInterval(timer);
  }, [fps, frames.length, loading, playing]);

  useEffect(() => {
    const runtime = sceneRuntimeRef.current;
    if (!runtime || !frame) return;
    if (runtime.fieldGroup) {
      runtime.scene.remove(runtime.fieldGroup);
      disposeGroup(runtime.fieldGroup);
    }

    const group = new THREE.Group();
    runtime.fieldGroup = group;
    runtime.scene.add(group);

    const [nx, ny, nz] = frame.dimensions;
    const [stepX, stepY, stepZ] = frame.sampleStep;
    const maxDimension = Math.max(nx, ny, nz);
    const width = 2.4 * (nx / maxDimension);
    const depth = 2.4 * (ny / maxDimension);
    const height = nz > 1 ? 2.4 * (nz / maxDimension) : 0.035;
    const xSpacing = width * (stepX / nx);
    const zSpacing = depth * (stepY / ny);
    const ySpacing = nz > 1 ? height * (stepZ / nz) : Math.max(xSpacing, zSpacing);
    const glyphLength = Math.max(0.012, Math.min(xSpacing, zSpacing, ySpacing) * 0.82 * glyphScale);
    const glyphCount = frame.glyphs.length / 6;

    const grid = new THREE.GridHelper(2.4, Math.min(64, maxDimension), 0xc6c6c6, 0xdcdcdc);
    grid.scale.set(width / 2.4, 1, depth / 2.4);
    grid.position.y = -height / 2;
    group.add(grid);
    const outlineGeometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(width, height, depth));
    const outline = new THREE.LineSegments(
      outlineGeometry,
      new THREE.LineBasicMaterial({ color: 0x8a8a8a, transparent: true, opacity: 0.72 }),
    );
    group.add(outline);

    const material = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.06 });
    const up = new THREE.Vector3(0, 1, 0);
    const direction = new THREE.Vector3();
    const point = new THREE.Vector3();
    const center = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const matrix = new THREE.Matrix4();
    const scale = new THREE.Vector3(1, 1, 1);
    const color = new THREE.Color();

    let shaft: THREE.InstancedMesh | null = null;
    let head: THREE.InstancedMesh | null = null;
    let cuboid: THREE.InstancedMesh | null = null;
    const shaftLength = glyphLength * 0.67;
    const headLength = glyphLength * 0.33;
    if (glyphMode === "arrows") {
      shaft = new THREE.InstancedMesh(
        new THREE.CylinderGeometry(glyphLength * 0.07, glyphLength * 0.07, shaftLength, 7),
        material,
        glyphCount,
      );
      head = new THREE.InstancedMesh(
        new THREE.ConeGeometry(glyphLength * 0.18, headLength, 9),
        material.clone(),
        glyphCount,
      );
      group.add(shaft, head);
    } else {
      cuboid = new THREE.InstancedMesh(
        new THREE.BoxGeometry(glyphLength * 0.2, glyphLength, glyphLength * 0.2),
        material,
        glyphCount,
      );
      group.add(cuboid);
    }

    for (let index = 0; index < glyphCount; index += 1) {
      const offset = index * 6;
      const x = frame.glyphs[offset];
      const y = frame.glyphs[offset + 1];
      const z = frame.glyphs[offset + 2];
      const mx = frame.glyphs[offset + 3];
      const my = frame.glyphs[offset + 4];
      const mz = frame.glyphs[offset + 5];
      point.set(
        ((x + 0.5) / nx - 0.5) * width,
        nz > 1 ? ((z + 0.5) / nz - 0.5) * height : 0,
        ((y + 0.5) / ny - 0.5) * depth,
      );
      direction.set(mx, mz, my);
      const magnitude = direction.length();
      if (magnitude === 0) continue;
      direction.normalize();
      quaternion.setFromUnitVectors(up, direction);
      glyphColor(color, colorMode, direction, magnitude, frame.magnitudeMax);

      if (shaft && head) {
        center.copy(point).addScaledVector(direction, -headLength * 0.5);
        matrix.compose(center, quaternion, scale);
        shaft.setMatrixAt(index, matrix);
        shaft.setColorAt(index, color);
        center.copy(point).addScaledVector(direction, shaftLength * 0.5);
        matrix.compose(center, quaternion, scale);
        head.setMatrixAt(index, matrix);
        head.setColorAt(index, color);
      } else if (cuboid) {
        matrix.compose(point, quaternion, scale);
        cuboid.setMatrixAt(index, matrix);
        cuboid.setColorAt(index, color);
      }
    }
    if (shaft && head) {
      shaft.instanceMatrix.needsUpdate = true;
      head.instanceMatrix.needsUpdate = true;
      if (shaft.instanceColor) shaft.instanceColor.needsUpdate = true;
      if (head.instanceColor) head.instanceColor.needsUpdate = true;
    }
    if (cuboid) {
      cuboid.instanceMatrix.needsUpdate = true;
      if (cuboid.instanceColor) cuboid.instanceColor.needsUpdate = true;
    }
  }, [colorMode, frame, glyphMode, glyphScale]);

  const setView = (axis: "x" | "y" | "z" | "reset") => {
    const runtime = sceneRuntimeRef.current;
    if (!runtime) return;
    const distance = 3.2;
    if (axis === "x") runtime.camera.position.set(distance, 0, 0);
    // Scene Y represents the simulation Z axis, while scene Z represents simulation Y.
    if (axis === "y") runtime.camera.position.set(0, 0, distance);
    if (axis === "z") runtime.camera.position.set(0, distance, 0.001);
    if (axis === "reset") runtime.camera.position.set(1.8, 1.45, 1.9);
    runtime.controls.target.set(0, 0, 0);
    runtime.controls.update();
  };

  const selectedFrame = frames[frameIndex];
  const glyphCount = frame ? frame.glyphs.length / 6 : 0;

  return (
    <main className="result-shell">
      <div className="title-drag" data-tauri-drag-region>
        <span className="brand-mark">m³</span>
        <strong>mumax³ ultrafast</strong>
        <span>Result viewer</span>
      </div>

      <header className="result-header">
        <div>
          <span className="eyebrow">OVF VECTOR FIELD</span>
          <h1>{frame?.title || "Simulation result"}</h1>
          <p title={resultSet.outputDirectory}>{resultSet.outputDirectory}</p>
        </div>
        <div className="result-header-actions">
          <span className="result-frame-name">{selectedFrame?.fileName || "No frames"}</span>
          <GlassButton onClick={onBack}>← Back to workspace</GlassButton>
        </div>
      </header>

      <section className="result-layout">
        <aside className="result-controls glass">
          {quantities.length > 1 && (
            <div className="control-section">
              <span className="eyebrow">FIELD QUANTITY</span>
              <select disabled={loading} value={quantity} onChange={(event) => setQuantity(event.target.value)}>
                {quantities.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </div>
          )}
          <div className="control-section">
            <span className="eyebrow">PLAYBACK</span>
            <button className="viewer-primary" type="button" disabled={loading || frames.length < 2} onClick={() => setPlaying((value) => !value)}>
              {playing ? "Pause" : "Play frames"} {playing ? "Ⅱ" : "▶"}
            </button>
            <label>
              <span>Frame</span>
              <strong>{frameIndex + 1} / {frames.length}</strong>
              <input disabled={loading} type="range" min={0} max={Math.max(0, frames.length - 1)} value={frameIndex} onChange={(event) => setFrameIndex(Number(event.target.value))} />
            </label>
            <label>
              <span>Playback speed</span>
              <strong>{fps} FPS</strong>
              <input type="range" min={1} max={30} value={fps} onChange={(event) => setFps(Number(event.target.value))} />
            </label>
          </div>

          <div className="control-section">
            <span className="eyebrow">GLYPHS</span>
            <div className="segmented-control">
              <button className={glyphMode === "arrows" ? "active" : ""} type="button" onClick={() => setGlyphMode("arrows")}>Arrows</button>
              <button className={glyphMode === "cuboids" ? "active" : ""} type="button" onClick={() => setGlyphMode("cuboids")}>Cuboids</button>
            </div>
            <label>
              <span>Glyph scale</span>
              <strong>{glyphScale.toFixed(1)}×</strong>
              <input type="range" min={0.4} max={2.2} step={0.1} value={glyphScale} onChange={(event) => setGlyphScale(Number(event.target.value))} />
            </label>
          </div>

          <div className="control-section">
            <span className="eyebrow">COLORS</span>
            <div className="segmented-control">
              <button className={colorMode === "direction" ? "active" : ""} type="button" onClick={() => setColorMode("direction")}>Direction</button>
              <button className={colorMode === "magnitude" ? "active" : ""} type="button" onClick={() => setColorMode("magnitude")}>Magnitude</button>
            </div>
            <div className={`color-legend ${colorMode}`}><span /><span /><span /></div>
          </div>

          <div className="control-section camera-controls">
            <span className="eyebrow">CAMERA</span>
            <div>
              <button type="button" onClick={() => setView("x")}>X</button>
              <button type="button" onClick={() => setView("y")}>Y</button>
              <button type="button" onClick={() => setView("z")}>Z</button>
              <button type="button" onClick={() => setView("reset")}>Reset</button>
            </div>
            <p>Drag to rotate · scroll to zoom · right-drag to move</p>
          </div>
        </aside>

        <div className="result-viewport glass">
          <div ref={viewportRef} className="webgl-stage" />
          <div className="viewer-readout">
            <span>{frame ? `${frame.dimensions.join(" × ")} cells` : "Reading frame…"}</span>
            <span>{glyphCount.toLocaleString()} glyphs</span>
            <span>{frame?.timeLabel || ""}</span>
          </div>
          {loading && <div className="viewer-overlay">Reading {selectedFrame?.fileName}…</div>}
          {error && <div className="viewer-overlay error">{error}</div>}
        </div>
      </section>
    </main>
  );
}
