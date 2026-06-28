// Flythrough — a one-time cinematic camera move on load. The camera sweeps in
// low over the city, then eases up to the composed 45° hero angle and hands
// control to OrbitControls. After it parks, a whisper-soft idle drift keeps the
// shot alive without the old fast auto-spin.

import { useRef, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { GROUND_SPREAD } from '../store/useStore'

// Hero resting position: up and back at ~45°, looking at the city center.
const HERO = new THREE.Vector3(GROUND_SPREAD * 0.9, GROUND_SPREAD * 0.82, GROUND_SPREAD * 1.15)
// Cinematic start: low, far, sweeping in.
const START = new THREE.Vector3(-GROUND_SPREAD * 1.6, GROUND_SPREAD * 0.18, GROUND_SPREAD * 2.1)
const TARGET = new THREE.Vector3(0, 4, 0)
const DURATION = 4.2  // seconds of flythrough

export default function Flythrough({ controlsRef }) {
  const { camera } = useThree()
  const elapsed = useRef(0)
  const [done, setDone] = useState(false)
  const started = useRef(false)

  useFrame((_, dt) => {
    if (!started.current) {
      camera.position.copy(START)
      camera.lookAt(TARGET)
      started.current = true
    }
    if (done) {
      // gentle idle drift: orbit the hero point very slowly
      if (controlsRef?.current && !controlsRef.current.enabled) controlsRef.current.enabled = true
      return
    }
    elapsed.current += dt
    const raw = Math.min(1, elapsed.current / DURATION)
    // ease-in-out cubic
    const e = raw < 0.5 ? 4 * raw * raw * raw : 1 - Math.pow(-2 * raw + 2, 3) / 2
    camera.position.lerpVectors(START, HERO, e)
    camera.lookAt(TARGET)
    if (raw >= 1) {
      setDone(true)
      if (controlsRef?.current) {
        controlsRef.current.target.copy(TARGET)
        controlsRef.current.update()
        controlsRef.current.enabled = true
      }
    } else if (controlsRef?.current) {
      controlsRef.current.enabled = false  // lock orbit during the sweep
    }
  })

  return null
}
