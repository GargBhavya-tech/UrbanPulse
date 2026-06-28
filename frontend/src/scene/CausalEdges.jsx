// CausalRoads — the causal edges as glowing roads laid flat on the city floor,
// connecting tower bases. Thickness ~ correlation strength. When a tower is
// selected, only its roads stay lit (upstream cyan = causes, downstream orange
// = effects); the rest dim into the grid.

import { useMemo } from 'react'
import * as THREE from 'three'
import { useStore, linkSource, linkTarget } from '../store/useStore'

const GROUND_Y = 0.06  // sit just above the floor so roads don't z-fight the grid

function Road({ a, b, strength, role }) {
  const geometry = useMemo(() => {
    const start = new THREE.Vector3(a.gx ?? a.x, GROUND_Y, a.gz ?? a.z)
    const end = new THREE.Vector3(b.gx ?? b.x, GROUND_Y, b.gz ?? b.z)
    const curve = new THREE.LineCurve3(start, end)
    const radius = 0.07 + strength * 0.22
    return new THREE.TubeGeometry(curve, 1, radius, 6, false)
  }, [a.gx, a.gz, b.gx, b.gz, strength])

  const color = role === 'upstream' ? '#4cc9f0' : role === 'downstream' ? '#ff8c42' : '#2c3a5c'
  const emissive = role === 'idle' ? 0.05 : role === 'neutral' ? 0.25 : 0.9
  const opacity = role === 'idle' ? 0.18 : 0.85

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={emissive}
        transparent
        opacity={opacity}
        roughness={0.4}
        depthWrite={false}
      />
    </mesh>
  )
}

export default function CausalRoads() {
  const links = useStore((s) => s.links)
  const byId = useStore((s) => s.byId)
  const selectedId = useStore((s) => s.selectedId)

  return (
    <group>
      {links.map((l, i) => {
        const sId = linkSource(l)
        const tId = linkTarget(l)
        const a = byId[sId]
        const b = byId[tId]
        if (!a || !b) return null
        let role = 'neutral'
        if (selectedId != null) {
          if (tId === selectedId) role = 'upstream'
          else if (sId === selectedId) role = 'downstream'
          else role = 'idle'
        }
        return <Road key={i} a={a} b={b} strength={l.strength} role={role} />
      })}
    </group>
  )
}
