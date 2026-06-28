// The 3D city — the hero view. A tilted night-city: causal links are vertical
// congestion towers rising from a glowing street-grid floor, connected by
// ground-level causal roads. A cinematic flythrough eases the camera to a 45°
// hero angle on load. Bloom makes every tower a real light source; fog sinks
// the far city into the dark.

import { useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stars } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import * as THREE from 'three'
import { useStore } from '../store/useStore'
import LinkTower from './LinkTower'
import CausalRoads from './CausalEdges'
import CascadePulse from './CascadePulse'
import Ground from './Ground'
import Flythrough from './Flythrough'

function Scene() {
  const nodes = useStore((s) => s.nodes)
  const clearSelected = useStore((s) => s.clearSelected)
  const controlsRef = useRef()

  return (
    <>
      <color attach="background" args={['#04060c']} />
      <fog attach="fog" args={['#04060c', 55, 165]} />

      {/* night-city lighting: cool ambient + two colored key lights */}
      <ambientLight intensity={0.35} />
      <hemisphereLight args={['#2a3a66', '#04060c', 0.5]} />
      <directionalLight position={[30, 50, 20]} intensity={0.7} color="#9ec5ff" />
      <pointLight position={[0, 30, 0]} intensity={60} color="#4cc9f0" distance={160} />
      <pointLight position={[-30, 12, -20]} intensity={40} color="#ff3d7f" distance={120} />

      <Stars radius={200} depth={80} count={1400} factor={3} saturation={0} fade speed={0.4} />

      <Ground />

      {/* click empty floor to clear selection */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]} onClick={clearSelected}>
        <planeGeometry args={[400, 400]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>

      <CausalRoads />
      <CascadePulse />
      {nodes.map((n) => (
        <LinkTower key={n.id} node={n} />
      ))}

      <Flythrough controlsRef={controlsRef} />
      <OrbitControls
        ref={controlsRef}
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.5}
        minDistance={20}
        maxDistance={170}
        maxPolarAngle={Math.PI / 2.15}   // don't let camera go under the floor
        autoRotate
        autoRotateSpeed={0.18}            // whisper-soft idle drift
        target={[0, 4, 0]}
      />

      <EffectComposer>
        <Bloom
          intensity={0.9}
          luminanceThreshold={0.2}
          luminanceSmoothing={0.9}
          mipmapBlur
        />
        <Vignette eskil={false} offset={0.2} darkness={0.85} />
      </EffectComposer>
    </>
  )
}

export default function Constellation() {
  return (
    <Canvas
      camera={{ position: [40, 38, 52], fov: 48, near: 0.1, far: 400 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, powerPreference: 'high-performance', toneMapping: THREE.ACESFilmicToneMapping }}
    >
      <Scene />
    </Canvas>
  )
}
