// CascadePulse — when the July 1 cascade plays, a bright pulse races from the
// source link along the ground roads, reaching each downstream link exactly at
// its real lag_minutes. Travels at ground level (the roads), then the arriving
// tower shoots up and reddens (handled in LinkTower). All timings are the real
// B8 lags.

import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import {
  useStore,
  cascadeInfection,
  CASCADE_SECONDS_PER_LAG_MIN,
} from '../store/useStore'

const PULSE_Y = 0.5  // ride just above the roads

function Pulse({ from, to, arriveAt, clock }) {
  const t = THREE.MathUtils.clamp(clock / arriveAt, 0, 1)
  const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2
  const fx = from.gx ?? from.x, fz = from.gz ?? from.z
  const tx = to.gx ?? to.x, tz = to.gz ?? to.z
  const x = fx + (tx - fx) * e
  const z = fz + (tz - fz) * e
  const arrived = clock >= arriveAt
  const scale = arrived ? 0.5 : 0.5 + Math.sin(t * Math.PI) * 0.35
  return (
    <mesh position={[x, PULSE_Y, z]} scale={scale}>
      <sphereGeometry args={[1, 12, 12]} />
      <meshBasicMaterial color="#fff3b0" transparent opacity={arrived ? 0 : 0.95} depthWrite={false} />
    </mesh>
  )
}

export default function CascadePulse() {
  const cascade = useStore((s) => s.cascade)
  const playing = useStore((s) => s.cascadePlaying)
  const clock = useStore((s) => s.cascadeClock)
  const tick = useStore((s) => s.tickCascade)
  const byId = useStore((s) => s.byId)

  useFrame((_, dt) => { if (playing) tick(dt) })

  if (!cascade) return null
  const inf = cascadeInfection(useStore.getState())
  if (!inf.active) return null

  const src = byId[cascade.source_link]
  if (!src) return null

  return (
    <group>
      {cascade.downstream.map((d) => {
        const tgt = byId[d.link_id]
        if (!tgt) return null
        const arriveAt = d.lag_minutes * CASCADE_SECONDS_PER_LAG_MIN
        if (clock > arriveAt + 0.4) return null
        return <Pulse key={d.link_id} from={src} to={tgt} arriveAt={arriveAt} clock={clock} />
      })}
    </group>
  )
}
