// CityMap — the hero view: a real Pangyo (Google-Maps-style) dark basemap with
// the 66 causal links drawn as glowing road segments, colored by metabolic
// state. Click a road to select it; the cascade lights up reached roads in
// sequence along the network. Replaces the R3F constellation.
//
// Geography note: the dataset has no coordinates, so each LINK_ID is mapped to
// a real, named Pangyo road segment (frontend/public/link_geometry.geojson,
// produced by scripts/12_link_geometry.py). Real streets; conventional mapping.

import { useEffect, useRef } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { useStore, cascadeInfection } from '../store/useStore'
import { STATE_COLOR } from '../data/api'

const PANGYO_CENTER = [127.108, 37.402]  // [lng, lat]
// CARTO dark-matter — free, keyless vector basemap.
const BASEMAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'

const STATE_FALLBACK = '#56638a'

export default function CityMap() {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const byId = useStore((s) => s.byId)
  const setSelected = useStore((s) => s.setSelected)
  const clearSelected = useStore((s) => s.clearSelected)

  // --- init map once ----------------------------------------------------- //
  useEffect(() => {
    if (mapRef.current) return
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASEMAP_STYLE,
      center: [127.108, 37.402],
      zoom: 13.2,
      pitch: 55,          // tilt for the "3D city" feel
      bearing: -17,
      attributionControl: true,
      antialias: true,
    })
    mapRef.current = map
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'bottom-right')

    map.on('load', async () => {
      const gj = await fetch('/link_geometry.geojson').then((r) => r.json())
      // attach current state to each feature so data-driven styling works
      paintFeatures(gj, byId)
      map.addSource('links', { type: 'geojson', data: gj })

      // glow underlay (wide, blurred)
      map.addLayer({
        id: 'links-glow',
        type: 'line',
        source: 'links',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': ['get', 'color'],
          'line-width': ['interpolate', ['linear'], ['zoom'], 11, 6, 15, 16],
          'line-opacity': 0.28,
          'line-blur': 6,
        },
      })
      // crisp core line
      map.addLayer({
        id: 'links-core',
        type: 'line',
        source: 'links',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': ['get', 'color'],
          'line-width': ['interpolate', ['linear'], ['zoom'], 11, 1.8, 15, 5],
          'line-opacity': 0.95,
        },
      })
      // selection highlight
      map.addLayer({
        id: 'links-selected',
        type: 'line',
        source: 'links',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        filter: ['==', ['get', 'link_id'], -1],
        paint: {
          'line-color': '#ffffff',
          'line-width': ['interpolate', ['linear'], ['zoom'], 11, 3.5, 15, 8],
          'line-opacity': 0.9,
          'line-blur': 1,
        },
      })

      // interactions
      const hit = ['links-core', 'links-glow']
      map.on('click', (e) => {
        const f = map.queryRenderedFeatures(e.point, { layers: hit })[0]
        if (f) setSelected(f.properties.link_id)
        else clearSelected()
      })
      map.on('mouseenter', 'links-core', () => { map.getCanvas().style.cursor = 'pointer' })
      map.on('mouseleave', 'links-core', () => { map.getCanvas().style.cursor = '' })

      map.__ready = true
    })

    return () => { map.remove(); mapRef.current = null }
  }, [])  // eslint-disable-line

  // --- recolor when node states load / change ---------------------------- //
  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.__ready) return
    const src = map.getSource('links')
    if (!src) return
    const gj = src._data
    if (!gj) return
    paintFeatures(gj, byId)
    src.setData(gj)
  }, [byId])

  // --- selection filter -------------------------------------------------- //
  const selectedId = useStore((s) => s.selectedId)
  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.__ready) return
    map.setFilter('links-selected', ['==', ['get', 'link_id'], selectedId ?? -1])
  }, [selectedId])

  // --- cascade animation: recolor reached links over time ---------------- //
  const cascadeClock = useStore((s) => s.cascadeClock)
  const cascadePlaying = useStore((s) => s.cascadePlaying)
  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.__ready) return
    const src = map.getSource('links')
    if (!src || !src._data) return
    const inf = cascadeInfection(useStore.getState())
    if (!inf.active) {
      // restore base colors
      paintFeatures(src._data, byId)
      src.setData(src._data)
      return
    }
    const gj = src._data
    for (const f of gj.features) {
      const id = f.properties.link_id
      const base = STATE_COLOR[byId[id]?.state] ?? STATE_FALLBACK
      if (id === inf.source) {
        f.properties.color = '#ffffff'
      } else if (inf.reached?.[id] > 0) {
        f.properties.color = lerpHex(base, STATE_COLOR.Collapsed, Math.min(1, inf.reached[id]))
      } else {
        f.properties.color = base
      }
    }
    src.setData(gj)
  }, [cascadeClock, cascadePlaying, byId])

  return <div ref={containerRef} className="map-root" />
}

// stamp each feature with its link's current state color
function paintFeatures(gj, byId) {
  for (const f of gj.features) {
    const node = byId[f.properties.link_id]
    f.properties.color = STATE_COLOR[node?.state] ?? STATE_FALLBACK
  }
}

// hex color lerp
function lerpHex(a, b, t) {
  const ca = parseInt(a.slice(1), 16), cb = parseInt(b.slice(1), 16)
  const ar = (ca >> 16) & 255, ag = (ca >> 8) & 255, ab = ca & 255
  const br = (cb >> 16) & 255, bg = (cb >> 8) & 255, bb = cb & 255
  const r = Math.round(ar + (br - ar) * t)
  const g = Math.round(ag + (bg - ag) * t)
  const bl = Math.round(ab + (bb - ab) * t)
  return '#' + ((1 << 24) | (r << 16) | (g << 8) | bl).toString(16).slice(1)
}
