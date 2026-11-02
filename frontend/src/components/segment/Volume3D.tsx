"use client";

import { OrbitControls, useGLTF, useProgress } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { EffectComposer, SelectiveBloom } from "@react-three/postprocessing";
import { Component, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import * as THREE from "three";

import { color, regionColor } from "@/design/tokens";
import type { LabelLayer } from "@/lib/api";

import styles from "./Segment.module.css";

/**
 * Interactive 3D view of a segmentation result: the brain as translucent glass and the predicted
 * tumour regions in the finding colours, with a glow (selective bloom) on the tumour only. Loads the result's mesh.glb,
 * built by the API on first request (backend/services/meshes.py).
 *
 * Rendering is on demand (a frame only when something changes), the pixel ratio is capped, and
 * the brain's glass look is a cheap fresnel shader rather than a transmission pass, so it stays
 * smooth on an ordinary laptop GPU.
 */

type Props = {
  url: string;
  visible: Record<LabelLayer, boolean>;
  /** The label overlay opacity, 0-1, as in the 2D viewer. */
  opacity: number;
};

type State = "loading" | "ready" | "error" | "unsupported";

function webglAvailable() {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") ?? canvas.getContext("webgl"));
  } catch {
    return false;
  }
}

export default function Volume3D({ url, visible, opacity }: Props) {
  // Rendered only in the browser (loaded with ssr: false), so this check can run on first render.
  const [supported] = useState(webglAvailable);
  const [ready, setReady] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  // A one-time cue that the view can be moved: shown when the surfaces appear, gone after a few
  // seconds or at the first drag or scroll.
  const [hint, setHint] = useState(true);
  const state: State = !supported ? "unsupported" : failure ? "error" : ready ? "ready" : "loading";

  useEffect(() => {
    if (state !== "ready") return;
    const timer = window.setTimeout(() => setHint(false), 4000);
    return () => window.clearTimeout(timer);
  }, [state]);
  const shown = (Object.keys(visible) as LabelLayer[]).filter((k) => visible[k]);

  return (
    <div
      className={styles.volume3d}
      data-state={state}
      data-layers={shown.join(",")}
      onPointerDown={() => setHint(false)}
      onWheel={() => setHint(false)}
    >
      {supported && !failure && (
        <LoadBoundary key={attempt} onError={setFailure}>
          <Canvas
            className={styles.volumeCanvas}
            frameloop="demand"
            flat
            dpr={[1, 1.5]}
            gl={{ antialias: true, powerPreference: "default" }}
            camera={{ fov: 27, near: 1, far: 4000, position: [0, 0, 450] }}
            onCreated={({ gl }) => {
              // Keep the first reason: removing a failed canvas also fires this event.
              gl.domElement.addEventListener("webglcontextlost", () =>
                setFailure((f) => f ?? "The browser stopped the 3D view (graphics context lost)."),
              );
            }}
          >
            <color attach="background" args={[color.bg]} />
            <Suspense fallback={null}>
              <Surfaces url={url} visible={visible} opacity={opacity} onReady={() => setReady(true)} />
            </Suspense>
          </Canvas>
        </LoadBoundary>
      )}
      {state === "loading" && <Loading />}
      {state === "ready" && (
        <>
          <p className={styles.volumeCue} data-hidden={!hint} aria-hidden={!hint}>
            <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
              <path d="M16 10a6 6 0 1 1-1.8-4.3M16 3v3.5h-3.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
            </svg>
            Drag to rotate · Scroll to zoom
          </p>
          <p className={`caption ${styles.volumeHint}`}>3D · drag to rotate · scroll or pinch to zoom</p>
        </>
      )}
      {state === "error" && (
        <div className={styles.volumeMessage} role="alert">
          <p>The 3D surfaces could not be shown.</p>
          <p className="caption">{failure}</p>
          <button
            type="button"
            className={styles.volumeRetry}
            onClick={() => {
              setFailure(null);
              setReady(false);
              useGLTF.clear(url);
              setAttempt((a) => a + 1);
            }}
          >
            <span className="u">Try again</span>
          </button>
          <p className="caption">The slice views and downloads are unaffected.</p>
        </div>
      )}
      {state === "unsupported" && (
        <div className={styles.volumeMessage} role="status">
          <p>The 3D view needs WebGL, which this browser does not provide or has turned off.</p>
          <p className="caption">
            The axial, coronal and sagittal views show the same result, and the surfaces can still be downloaded as a
            .glb file for any 3D viewer.
          </p>
        </div>
      )}
    </div>
  );
}

function Loading() {
  const { active, progress } = useProgress();
  return (
    <div className={styles.volumeMessage} aria-live="polite">
      <p>{active && progress > 0 ? `Loading the 3D surfaces · ${Math.round(progress)}%` : "Building the 3D surfaces…"}</p>
      <p className="caption">The surfaces are computed on the server the first time a result is opened in 3D.</p>
    </div>
  );
}

/** Catches a failed download or parse so the rest of the results screen keeps working. */
class LoadBoundary extends Component<{ children: ReactNode; onError: (message: string) => void }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error) {
    this.props.onError(error.message || "The surfaces file could not be loaded.");
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}

// ---- Scene ----

const BRAIN_VERTEX = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vView;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vView = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }
`;

/** Glass: faint where the surface faces the viewer, brighter towards the silhouette. */
const BRAIN_FRAGMENT = /* glsl */ `
  uniform vec3 uColor;
  uniform vec3 uRim;
  uniform float uOpacity;
  varying vec3 vNormal;
  varying vec3 vView;
  void main() {
    vec3 n = normalize(vNormal);
    float facing = abs(dot(n, normalize(vView)));
    float fresnel = pow(1.0 - facing, 3.0);
    float light = 0.55 + 0.45 * max(dot(n, normalize(vec3(0.4, 0.7, 0.6))), 0.0);
    vec3 col = mix(uColor * light, uRim, fresnel);
    gl_FragColor = vec4(col, uOpacity * (0.08 + 0.92 * fresnel));
  }
`;

// Opacity and self-illumination per layer, as in the static render: a translucent whole-tumour
// shell, a solid necrotic core and a faint enhancing rim.
const TUMOUR = {
  edema: { colour: regionColor.WT, opacity: 0.5, glow: 0.3 },
  necrotic: { colour: regionColor.TC, opacity: 0.95, glow: 0.35 },
  enhancing: { colour: regionColor.ET, opacity: 0.32, glow: 0.2 },
} as const;

function Surfaces({
  url,
  visible,
  opacity,
  onReady,
}: Props & { onReady: () => void }) {
  const gltf = useGLTF(url, false, false);
  const { camera, scene, invalidate } = useThree();
  const reduceMotion = useMemo(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches, []);

  const extras = (gltf.scene.userData ?? {}) as { radius_mm?: number; camera_direction?: number[] };
  const radius = extras.radius_mm ?? 100;
  const geometries = useMemo(() => {
    const out: Partial<Record<"brain" | LabelLayer, THREE.BufferGeometry>> = {};
    gltf.scene.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) out[o.name as "brain" | LabelLayer] = (o as THREE.Mesh).geometry;
    });
    return out;
  }, [gltf]);

  const brainMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: BRAIN_VERTEX,
        fragmentShader: BRAIN_FRAGMENT,
        uniforms: {
          uColor: { value: new THREE.Color(color.textMuted) },
          uRim: { value: new THREE.Color(color.barShare) },
          uOpacity: { value: 0.55 },
        },
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
    [],
  );
  // Opacity and visibility come in as props; draw a fresh frame when they change.
  useEffect(() => {
    invalidate();
  }, [visible, opacity, invalidate]);

  // Lights travel with the camera, as in the static render, so the lit side is the side in view.
  const lights = useMemo(() => {
    const key = new THREE.DirectionalLight("#ffffff", 1.6);
    key.position.set(1, 1.3, 1);
    const fill = new THREE.DirectionalLight("#ffffff", 0.45);
    fill.position.set(-1.3, 0.2, 0.7);
    const under = new THREE.DirectionalLight("#ffffff", 0.8);
    under.position.set(0.2, 0.5, -1.6);
    return [new THREE.AmbientLight("#ffffff", 0.35), key, fill, under];
  }, []);

  // Frame the brain from the same direction as the Overview's static render.
  useEffect(() => {
    const d = new THREE.Vector3(...(extras.camera_direction ?? [0.42, 0.46, 0.8])).normalize();
    const fov = ((camera as THREE.PerspectiveCamera).fov * Math.PI) / 180;
    camera.position.copy(d.multiplyScalar((radius / Math.sin(fov / 2)) * 1.02));
    camera.lookAt(0, 0, 0);
    camera.add(...lights);
    scene.add(camera);
    invalidate();
    onReady();
    return () => {
      camera.remove(...lights);
      scene.remove(camera);
    };
    // Run once per loaded file.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gltf]);

  // Only the tumour layers glow; the brain stays clean glass.
  const edema = useRef<THREE.Mesh>(null);
  const necrotic = useRef<THREE.Mesh>(null);
  const enhancing = useRef<THREE.Mesh>(null);

  return (
    <>
      {/* Inside out: each layer blends over the ones it contains. */}
      {(["necrotic", "enhancing", "edema"] as const).map((name, i) => {
        const geometry = geometries[name];
        if (!geometry) return null;
        const spec = TUMOUR[name];
        const alpha = spec.opacity * opacity;
        return (
          <mesh
            key={name}
            ref={{ necrotic, edema, enhancing }[name]}
            geometry={geometry}
            visible={visible[name]}
            renderOrder={i + 1}
          >
            <meshStandardMaterial
              color={spec.colour}
              emissive={spec.colour}
              emissiveIntensity={spec.glow}
              roughness={0.45}
              metalness={0}
              transparent
              opacity={alpha}
              // Every tumour layer writes depth: the glow pass picks out surfaces by depth.
              depthWrite
              side={THREE.DoubleSide}
            />
          </mesh>
        );
      })}
      {geometries.brain && <mesh geometry={geometries.brain} material={brainMaterial} renderOrder={4} />}
      <OrbitControls
        makeDefault
        enablePan={false}
        enableDamping={!reduceMotion}
        dampingFactor={0.08}
        rotateSpeed={0.7}
        zoomSpeed={0.8}
        minDistance={radius * 1.6}
        maxDistance={radius * 7}
      />
      <EffectComposer multisampling={4}>
        <SelectiveBloom
          selection={[edema, necrotic, enhancing]}
          lights={lights}
          mipmapBlur
          ignoreBackground
          luminanceThreshold={0}
          intensity={1.1}
          radius={0.75}
        />
      </EffectComposer>
    </>
  );
}
