// UrbanPulse — global store.
// Loads the causal graph + ecosystem state + archetypes, merges them into a
// single node/link model, and runs a 3D force simulation to position the 66
// links in space. The constellation reads this; click selection lives here too.

import { create } from 'zustand'
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
} from 'd3-force-3d'
import { api, stateColor, healthRadius } from '../data/api'

// Cascade replay pacing: real seconds of animation per minute of causal lag.
// At 0.9s/min, a 5-min lag arrives in 4.5s, 10-min in 9s — slow enough to read
// the spread, fast enough to stay under ~15s for the whole event.
export const CASCADE_SECONDS_PER_LAG_MIN = 0.9

// --- Timeline scrubber (Stage 2) ------------------------------------------ //
// Real-ms between autoplay ticks. Speed (1x/5x/30x) changes how many axis
// ticks we advance per interval, NOT the interval cadence — keeps the fetch/
// repaint rate constant regardless of speed so 30x doesn't hammer the API.
const SCRUB_TICK_MS = 150
// module-level (not store state — an interval handle isn't serializable/
// reactive data, and only one scrubber instance ever runs).
let scrubTimer = null
// tiny frame cache so scrubbing back and forth doesn't refetch.
const scrubFrameCache = new Map()

// Advance `steps` axis ticks from (day, minute), wrapping day14/last-minute
// back to day1/00:00. Works for any step count, including a full lap.
function advanceTimelineTicks(axis, day, minute, steps) {
  const { days, minutes } = axis
  let di = days.indexOf(day)
  let mi = minutes.indexOf(minute)
  if (di === -1) di = 0
  if (mi === -1) mi = 0
  const span = days.length * minutes.length
  let total = di * minutes.length + mi + steps
  total = ((total % span) + span) % span
  return { day: days[Math.floor(total / minutes.length)], minute: minutes[total % minutes.length] }
}

// City scale. Nodes are laid out on the ground plane (X,Z) by a 2D force sim,
// then lifted on Y by congestion (severity). These constants tune the look.
export const GROUND_SPREAD = 46      // half-width the layout is scaled to fill
export const TOWER_MAX_HEIGHT = 11   // tallest congestion spike (worst health)
export const TOWER_MIN_HEIGHT = 0.8  // shortest spike (healthy road)

// congestion 0..1 from health (100 healthy -> 0, 0 gridlock -> 1)
export function severityFromHealth(health) {
  const h = Math.max(0, Math.min(100, health ?? 50))
  return (100 - h) / 100
}
export function towerHeight(health) {
  return TOWER_MIN_HEIGHT + severityFromHealth(health) * (TOWER_MAX_HEIGHT - TOWER_MIN_HEIGHT)
}

function layout(nodes, links) {
  // 2D force layout on the ground plane: we run d3-force in 2 dims and map the
  // result to (x, z). Y is reserved for the congestion tower height. Running to
  // convergence synchronously (66 nodes) so the city renders settled.
  const sim = forceSimulation(nodes, 2)
    .force('link', forceLink(links).id((d) => d.id).distance(9).strength(0.35))
    .force('charge', forceManyBody().strength(-26))
    .force('center', forceCenter(0, 0))
    .force('collide', forceCollide((d) => d.radius + 1.4))
    .stop()
  for (let i = 0; i < 400; i++) sim.tick()

  // Center + scale to fill the ground footprint, then map x,y -> x,z.
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  for (const n of nodes) {
    minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x)
    minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y)
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2
  const span = Math.max(maxX - minX, maxY - minY) || 1
  const scale = (GROUND_SPREAD * 2) / span
  for (const n of nodes) {
    const gx = (n.x - cx) * scale
    const gz = (n.y - cy) * scale
    n.gx = gx                 // ground X
    n.gz = gz                 // ground Z
    n.height = towerHeight(n.health)
    // keep x/y/z too (some code reads them); base sits on the ground plane
    n.x = gx
    n.y = 0
    n.z = gz
  }
  return nodes
}

export const useStore = create((set, get) => ({
  status: 'idle', // idle | loading | ready | error
  error: null,
  nodes: [],      // {id, x,y,z, state, regime, health, archetype, radius, color}
  links: [],      // {source, target, lag, strength}
  byId: {},       // id -> node
  archetypes: {}, // id -> full archetype record
  snapshot: { day: 1, minute: 545 },
  selectedId: null,
  viewMode: 'planner', // planner | citizen

  // --- Cascade replay (Stage 2) ---------------------------------------- //
  // cascade: the loaded "July 1" propagation event (source + downstream+lags).
  // cascadePlaying: whether the pulse animation is running.
  // cascadeClock: elapsed playback seconds (driven by the scene's useFrame).
  cascade: null,
  cascadePlaying: false,
  cascadeClock: 0,
  cascadeMaxLag: 0,

  // --- Timeline scrubber (Stage 2) -------------------------------------- //
  // timelineAxis: {days, minutes, interval_minutes} from /echo/timeline/axis.
  // scrubDay/scrubMinute: the frame currently shown on the slider/map.
  // scrubPlaying: autoplay on/off. scrubSpeed: 1 | 5 | 30 (axis ticks/step).
  // scrubLoading: a frame fetch is in flight (for a subtle UI hint only).
  timelineAxis: null,
  scrubDay: 1,
  scrubMinute: 0,
  scrubPlaying: false,
  scrubSpeed: 1,
  scrubLoading: false,

  // --- Deep-dive drawer: counterfactual + LLM (Stage 2) ----------------- //
  counterfactualByLink: {},
  counterfactualLoading: {},
  llmBriefingByLink: {},
  llmBriefingLoading: {},
  llmBriefingError: {},
  llmQuestion: '',
  llmAnswerByLink: {},
  llmAsking: false,
  llmAskError: null,

  setSelected: (id) => set({ selectedId: id }),
  clearSelected: () => set({ selectedId: null }),
  setViewMode: (m) => set({ viewMode: m }),

  // Load the headline cascade (Link 36, Day 1 — the July 1 disaster) once.
  loadCascade: async () => {
    if (get().cascade) return
    try {
      const events = await api.cascadesDetailed(36, 1)
      if (!events.length) return
      // Pick the earliest-firing Link 36 event = the moment it goes Saturated.
      const ev = events.sort((a, b) => a.minute_of_day - b.minute_of_day)[0]
      const maxLag = Math.max(0, ...ev.downstream.map((d) => d.lag_minutes))
      set({ cascade: ev, cascadeMaxLag: maxLag })
    } catch (e) {
      // cascade is optional polish — never break the app over it
      console.warn('cascade load failed', e)
    }
  },

  playCascade: () => {
    const { cascade } = get()
    const start = (ev) => {
      const maxLag = Math.max(0, ...ev.downstream.map((d) => d.lag_minutes))
      set({ cascade: ev, cascadeMaxLag: maxLag, cascadePlaying: true, cascadeClock: 0, selectedId: ev.source_link })
      // self-driving rAF loop (the map view has no R3F render loop)
      let last = performance.now()
      const step = (now) => {
        const dt = (now - last) / 1000
        last = now
        const st = get()
        if (!st.cascadePlaying) return
        const total = st.cascadeMaxLag * CASCADE_SECONDS_PER_LAG_MIN + 1.2
        const next = st.cascadeClock + dt
        if (next >= total) { set({ cascadePlaying: false, cascadeClock: total }); return }
        set({ cascadeClock: next })
        requestAnimationFrame(step)
      }
      requestAnimationFrame(step)
    }
    if (cascade) start(cascade)
    else get().loadCascade().then(() => { const ev = get().cascade; if (ev) start(ev) })
  },
  stopCascade: () => set({ cascadePlaying: false, cascadeClock: 0 }),
  // kept for any R3F-driven views; harmless if unused
  tickCascade: (dt) => {
    const { cascadePlaying, cascadeClock, cascadeMaxLag } = get()
    if (!cascadePlaying) return
    const total = cascadeMaxLag * CASCADE_SECONDS_PER_LAG_MIN + 1.2
    const next = cascadeClock + dt
    if (next >= total) { set({ cascadePlaying: false, cascadeClock: total }); return }
    set({ cascadeClock: next })
  },

  // --- Timeline scrubber (Stage 2) --------------------------------------- //
  // Loads the day/minute axis once. Does NOT fetch or repaint a frame — the
  // map keeps showing the live snapshot until the person actually plays or
  // drags the slider, so mounting the scrubber has zero visible effect.
  loadTimelineAxis: async () => {
    if (get().timelineAxis) return
    try {
      const axis = await api.timelineAxis()
      set({ timelineAxis: axis, scrubDay: axis.days[0], scrubMinute: axis.minutes[0] })
    } catch (e) {
      console.warn('timeline axis load failed', e)
    }
  },

  setScrubSpeed: (speed) => set({ scrubSpeed: speed }),

  // Jump to (day, minute): updates the slider position immediately (so the UI
  // never feels stuck), fetches/caches the frame, then repaints the map —
  // UNLESS a cascade replay is currently playing, in which case the cascade
  // keeps owning link colors and this frame's colors are dropped silently.
  // (Position is still recorded, so whoever calls this later — autoplay's
  // next tick, or Timeline's own cascade-end watcher — picks up from here.)
  applyFrame: async (day, minute) => {
    set({ scrubDay: day, scrubMinute: minute })
    const key = `${day}-${minute}`
    let frame = scrubFrameCache.get(key)
    if (!frame) {
      set({ scrubLoading: true })
      try {
        const res = await api.timelineFrame(day, minute)
        frame = res.links
        scrubFrameCache.set(key, frame)
      } catch (e) {
        console.warn('timeline frame load failed', e)
        set({ scrubLoading: false })
        return
      }
      set({ scrubLoading: false })
    }

    const st = get()
    // stale guard: a newer jump/tick has already moved past this request
    if (st.scrubDay !== day || st.scrubMinute !== minute) return
    // cascade owns the colors while it's playing — defer the repaint
    if (st.cascadePlaying) return

    const frameById = {}
    for (const f of frame) frameById[f.link_id] = f
    const nodes = st.nodes.map((n) => {
      const f = frameById[n.id]
      if (!f) return n
      return {
        ...n,
        state: f.state,
        health: f.health,
        queueS: f.queue_s,
        occup: f.occup,
        radius: healthRadius(f.health),
        color: stateColor(f.state),
      }
    })
    const byId = {}
    for (const n of nodes) byId[n.id] = n
    set({ nodes, byId })
  },

  // Called by Timeline.jsx on slider drag / scrub. Alias kept separate from
  // applyFrame in case position-setting ever needs to diverge from fetching.
  setScrubPosition: (day, minute) => get().applyFrame(day, minute),

  playScrub: () => {
    if (scrubTimer) return
    set({ scrubPlaying: true })
    scrubTimer = setInterval(() => {
      const st = get()
      if (!st.timelineAxis) return
      if (st.cascadePlaying) return // yield the whole tick — cascade is mid-flight
      const { day, minute } = advanceTimelineTicks(st.timelineAxis, st.scrubDay, st.scrubMinute, st.scrubSpeed)
      st.applyFrame(day, minute)
    }, SCRUB_TICK_MS)
  },
  pauseScrub: () => {
    set({ scrubPlaying: false })
    if (scrubTimer) { clearInterval(scrubTimer); scrubTimer = null }
  },

  // --- Deep-dive drawer: counterfactual --------------------------------- //
  loadCounterfactual: async (linkId) => {
    if (linkId == null) return
    const { counterfactualByLink, counterfactualLoading } = get()
    if (linkId in counterfactualByLink || counterfactualLoading[linkId]) return
    set({ counterfactualLoading: { ...counterfactualLoading, [linkId]: true } })
    try {
      const rec = await api.counterfactualLink(linkId)
      set((s) => ({
        counterfactualByLink: { ...s.counterfactualByLink, [linkId]: rec },
        counterfactualLoading: { ...s.counterfactualLoading, [linkId]: false },
      }))
    } catch (e) {
      const is404 = /\b404\b/.test(String(e.message ?? e))
      set((s) => {
        const next = { ...s.counterfactualByLink }
        if (is404) next[linkId] = null // confirmed no result — don't retry
        return {
          counterfactualByLink: next,
          counterfactualLoading: { ...s.counterfactualLoading, [linkId]: false },
        }
      })
      if (!is404) console.warn('counterfactual load failed', e)
    }
  },

  // --- Deep-dive drawer: LLM planner briefing --------------------------- //
  loadPlannerBriefing: async (linkId) => {
    if (linkId == null) return
    set((s) => ({
      llmBriefingLoading: { ...s.llmBriefingLoading, [linkId]: true },
      llmBriefingError: { ...s.llmBriefingError, [linkId]: null },
    }))
    try {
      const res = await api.llmGenerate({ link_id: linkId, output_type: 'planner_briefing' })
      set((s) => ({
        llmBriefingByLink: { ...s.llmBriefingByLink, [linkId]: res.text },
        llmBriefingLoading: { ...s.llmBriefingLoading, [linkId]: false },
      }))
    } catch (e) {
      set((s) => ({
        llmBriefingLoading: { ...s.llmBriefingLoading, [linkId]: false },
        llmBriefingError: { ...s.llmBriefingError, [linkId]: String(e.message ?? e) },
      }))
    }
  },

  setLlmQuestion: (q) => set({ llmQuestion: q }),

  // --- Deep-dive drawer: LLM free-text Q&A ------------------------------ //
  askLLM: async (linkId, question) => {
    const q = (question ?? '').trim()
    if (linkId == null || !q) return
    set({ llmAsking: true, llmAskError: null })
    try {
      const res = await api.llmAsk({ link_id: linkId, question: q, audience: 'planner' })
      set((s) => ({
        llmAnswerByLink: { ...s.llmAnswerByLink, [linkId]: { question: q, answer: res.answer } },
        llmAsking: false,
        llmQuestion: '',
      }))
    } catch (e) {
      set({ llmAsking: false, llmAskError: String(e.message ?? e) })
    }
  },

  load: async () => {
    set({ status: 'loading', error: null })
    try {
      const [graph, eco, archList] = await Promise.all([
        api.causalGraph(),
        api.ecosystemState(),
        api.archetypes(),
      ])

      const archById = {}
      for (const a of archList) archById[a.link_id] = a

      const stateById = {}
      for (const l of eco.links) stateById[l.link_id] = l

      // Build node set from the graph's node list (the authoritative 66).
      const nodes = graph.nodes.map((id) => {
        const s = stateById[id] ?? {}
        const health = s.road_health_score ?? 50
        return {
          id,
          state: s.state ?? 'Stressed',
          regime: s.regime ?? 'forward',
          health,
          archetype: (archById[id] && archById[id].archetype) ?? s.archetype ?? null,
          radius: healthRadius(health),
          color: stateColor(s.state ?? 'Stressed'),
          // seed positions on a plane so the 2D ground sim starts spread out
          x: (Math.random() - 0.5) * 40,
          y: (Math.random() - 0.5) * 40,
          z: 0,
        }
      })

      const links = graph.edges.map((e) => ({
        source: e.source,
        target: e.target,
        lag: e.lag_minutes,
        strength: e.correlation_strength,
      }))

      layout(nodes, links)

      const byId = {}
      for (const n of nodes) byId[n.id] = n

      set({
        status: 'ready',
        nodes,
        links,
        byId,
        archetypes: archById,
        snapshot: { day: eco.day_number, minute: eco.minute_of_day },
      })

      // Prefetch the headline cascade so "Play" is instant (non-blocking).
      get().loadCascade()
    } catch (e) {
      set({ status: 'error', error: String(e.message ?? e) })
    }
  },
}))

// Selectors
export const selectSelectedNode = (s) =>
  s.selectedId == null ? null : s.byId[s.selectedId] ?? null

// d3-force replaces source/target with node objects after simulation.
export const linkSource = (l) => (typeof l.source === 'object' ? l.source.id : l.source)
export const linkTarget = (l) => (typeof l.target === 'object' ? l.target.id : l.target)

// --- Cascade derived state ------------------------------------------------ //
// Given the playback clock, which downstream links have been "reached" yet, and
// how strongly (0..1 glow, fading in over a short window as the pulse lands).
export function cascadeInfection(s) {
  const { cascade, cascadeClock, cascadePlaying } = s
  if (!cascade) return { source: null, reached: {}, active: false }
  const reached = {}
  for (const d of cascade.downstream) {
    const arriveAt = d.lag_minutes * CASCADE_SECONDS_PER_LAG_MIN
    const since = cascadeClock - arriveAt
    if (since >= 0) {
      // glow ramps 0->1 over 0.6s after arrival, then holds
      reached[d.link_id] = Math.min(1, since / 0.6)
    }
  }
  return {
    source: cascade.source_link,
    reached,
    active: cascadePlaying || cascadeClock > 0,
    downstream: cascade.downstream,
  }
}
