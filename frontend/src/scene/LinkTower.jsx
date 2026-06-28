// LinkTower — one road link as a vertical congestion spike rising from the city
// floor. Height = how congested the road is (worse health = taller). Color =
// metabolic state. A glowing cap sits on top; hover/select lifts it. During the
// cascade, reached towers shoot up and redden in sequence.

import { useRef, useState, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Billboard, Text } from '@react-three/drei'
import * as THREE from 'three'
import { useStore, cascadeInfection, towerHeight } from '../store/useStore'

export default function LinkTower({ node }) {
  const groupRef = useRef()
  const barRef = useRef()
  const capRef = useRef()
  const [hovered, setHovered] = useState(false)
  const selectedId = useStore((s) => s.selectedId)
  const setSelected = useStore((s) => s.setSelected)

  const selected = selectedId === node.id
  const dimmed = selectedId != null && !selected
  const collapsed = node.state === 'Collapsed'

  const baseColor = useMemo(() => new THREE.Color(node.color), [node.color])
  const alarmColor = useMemo(() => new THREE.Color('#ff3d7f'), [])
  const baseHeight = node.height ?? towerHeight(node.health)
  const footprint = 0.55 + (node.radius ?? 1) * 0.45  // bar thickness

  // smoothed animated height (so cascade growth eases)
  const curH = useRef(baseHeight)

  useFrame((state) => {
    const t = state.clock.elapsedTime
    const inf = cascadeInfection(useStore.getState())
    const isSource = inf.active && inf.source === node.id
    const reach = inf.reached?.[node.id] ?? 0
    const infected = reach > 0

    // target height: grows when infected / source
    let targetH = baseHeight
    if (isSource) targetH = baseHeight + 4.5 + Math.sin(t * 6) * 0.4
    else if (infected) targetH = baseHeight + reach * 4.0
    if (hovered || selected) targetH += 1.2
    // ease toward target
    curH.current += (targetH - curH.current) * 0.12

    if (barRef.current && capRef.current) {
      const h = Math.max(0.4, curH.current)
      barRef.current.scale.y = h
      barRef.current.position.y = h / 2
      capRef.current.position.y = h + 0.25

      // color: base state color, shifting to alarm when infected
      const mat = barRef.current.material
      const capMat = capRef.current.material
      if (infected && !selected) {
        mat.color.lerpColors(baseColor, alarmColor, reach * 0.85)
        mat.emissive.copy(mat.color)
        capMat.color.copy(mat.color); capMat.emissive.copy(mat.color)
      } else if (!mat.color.equals(baseColor)) {
        mat.color.copy(baseColor); mat.emissive.copy(baseColor)
        capMat.color.copy(baseColor); capMat.emissive.copy(baseColor)
      }

      let emissive = selected ? 1.5 : collapsed ? 1.0 : 0.55
      if (isSource) emissive = 2.4 + Math.sin(t * 8) * 0.6
      else if (infected) emissive = 0.6 + reach * 1.8
      mat.emissiveIntensity = emissive
      capMat.emissiveIntensity = emissive + 0.6
      mat.opacity = dimmed ? 0.4 : 0.92
      capMat.opacity = dimmed ? 0.4 : 1
    }
  })

  return (
    <group ref={groupRef} position={[node.gx ?? node.x, 0, node.gz ?? node.z]}>
      {/* the congestion bar */}
      <mesh
        ref={barRef}
        position={[0, baseHeight / 2, 0]}
        scale={[1, baseHeight, 1]}
        onPointerOver={(e) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = 'pointer' }}
        onPointerOut={() => { setHovered(false); document.body.style.cursor = 'auto' }}
        onClick={(e) => { e.stopPropagation(); setSelected(node.id) }}
      >
        <cylinderGeometry args={[footprint, footprint * 1.15, 1, 12]} />
        <meshStandardMaterial
          color={node.color}
          emissive={node.color}
          emissiveIntensity={0.55}
          transparent
          opacity={0.92}
          roughness={0.3}
          metalness={0.2}
        />
      </mesh>

      {/* glowing cap on top */}
      <mesh ref={capRef} position={[0, baseHeight + 0.25, 0]}>
        <sphereGeometry args={[footprint * 1.05, 16, 16]} />
        <meshStandardMaterial
          color={node.color}
          emissive={node.color}
          emissiveIntensity={1.1}
          transparent
          opacity={1}
          roughness={0.2}
        />
      </mesh>

      {/* ground footprint ring — anchors the tower to the city floor */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.03, 0]}>
        <ringGeometry args={[footprint * 1.1, footprint * 1.5, 20]} />
        <meshBasicMaterial color={node.color} transparent opacity={dimmed ? 0.08 : 0.28} depthWrite={false} />
      </mesh>

      {(hovered || selected) && (
        <Billboard position={[0, (node.height ?? 2) + 2.2, 0]}>
          <Text fontSize={1.0} color="#e8edf7" anchorX="center" anchorY="bottom"
            outlineWidth={0.05} outlineColor="#05070d">
            {`LINK ${node.id}`}
          </Text>
        </Billboard>
      )}
    </group>
  )
}
