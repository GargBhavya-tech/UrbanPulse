// Ground — the city floor. A dark plane with a faint glowing grid that gives
// the scene its "3D city at night" depth. Sits at y=0; towers rise from it and
// roads hug it. The grid fades into the fog at the edges.

import { Grid } from '@react-three/drei'
import { GROUND_SPREAD } from '../store/useStore'

export default function Ground() {
  const size = GROUND_SPREAD * 2.6
  return (
    <group>
      {/* solid dark floor to catch the towers' glow and block the starfield */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]} receiveShadow>
        <planeGeometry args={[size, size]} />
        <meshStandardMaterial color="#070b14" roughness={1} metalness={0} />
      </mesh>

      {/* luminous grid — the street-grid feel */}
      <Grid
        position={[0, 0, 0]}
        args={[size, size]}
        cellSize={4}
        cellThickness={0.6}
        cellColor="#16203a"
        sectionSize={20}
        sectionThickness={1.1}
        sectionColor="#26406b"
        fadeDistance={GROUND_SPREAD * 2.4}
        fadeStrength={2.5}
        followCamera={false}
        infiniteGrid={false}
      />
    </group>
  )
}
