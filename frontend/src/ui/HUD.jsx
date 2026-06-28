// HUD — the instrument chrome floating over the constellation: title block,
// snapshot readout, view-mode toggle (planner/citizen), and the state legend.

import { useStore, cascadeInfection } from '../store/useStore'
import { STATE_ORDER, STATE_COLOR } from '../data/api'
import './hud.css'

function minuteToClock(m) {
  const h = Math.floor(m / 60)
  const mm = String(m % 60).padStart(2, '0')
  const ap = h < 12 ? 'AM' : 'PM'
  const h12 = ((h + 11) % 12) + 1
  return `${h12}:${mm} ${ap}`
}

function CascadeControl() {
  const cascade = useStore((s) => s.cascade)
  const playing = useStore((s) => s.cascadePlaying)
  const clock = useStore((s) => s.cascadeClock)
  const play = useStore((s) => s.playCascade)
  const stop = useStore((s) => s.stopCascade)

  if (!cascade) return null

  // live readout: how many downstream links reached so far
  const inf = cascadeInfection(useStore.getState())
  const reachedCount = Object.keys(inf.reached ?? {}).length
  const total = cascade.downstream.length
  const done = !playing && clock > 0

  return (
    <div className="cascade-ctl">
      <div className="cascade-head">
        <span className="eyebrow">ECHO · cascade replay</span>
        <span className="cascade-tag mono">JUL 1 · {minuteToClock(cascade.minute_of_day)}</span>
      </div>
      <div className="cascade-body">
        <button
          className={`cascade-btn ${playing ? 'playing' : ''}`}
          onClick={playing ? stop : play}
        >
          {playing ? '◼ Stop' : done ? '↻ Replay' : '▶ Play cascade'}
        </button>
        <div className="cascade-readout mono">
          <span className="cascade-src">LINK {cascade.source_link}</span>
          <span className="cascade-spread">
            {playing || done
              ? `spreading → ${reachedCount}/${total} links`
              : `${total} downstream links at risk`}
          </span>
        </div>
      </div>
    </div>
  )
}

export default function HUD() {
  const snapshot = useStore((s) => s.snapshot)
  const nodes = useStore((s) => s.nodes)
  const viewMode = useStore((s) => s.viewMode)
  const setViewMode = useStore((s) => s.setViewMode)

  const counts = STATE_ORDER.reduce((acc, st) => {
    acc[st] = nodes.filter((n) => n.state === st).length
    return acc
  }, {})
  const critical = (counts.Saturated ?? 0) + (counts.Collapsed ?? 0)

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">◉</div>
          <div>
            <div className="brand-name">UrbanPulse</div>
            <div className="eyebrow">Ecosystem Causal Highway Observatory</div>
          </div>
        </div>

        <div className="snapshot mono">
          <span className="snap-day">DAY {snapshot.day}</span>
          <span className="snap-sep">·</span>
          <span className="snap-time">{minuteToClock(snapshot.minute)}</span>
          <span className="snap-live"><i /> SNAPSHOT</span>
        </div>

        <div className="toggle" role="tablist" aria-label="View mode">
          <button
            className={viewMode === 'planner' ? 'active' : ''}
            onClick={() => setViewMode('planner')}
          >
            Planner
          </button>
          <button
            className={viewMode === 'citizen' ? 'active' : ''}
            onClick={() => setViewMode('citizen')}
          >
            Citizen
          </button>
        </div>
      </header>

      <div className="legend">
        <div className="legend-title eyebrow">Network metabolic state</div>
        <div className="legend-items">
          {STATE_ORDER.map((st) => (
            <div className="legend-item" key={st}>
              <i className="swatch" style={{ background: STATE_COLOR[st], boxShadow: `0 0 8px ${STATE_COLOR[st]}` }} />
              <span className="legend-lbl">{st}</span>
              <span className="legend-count mono">{counts[st] ?? 0}</span>
            </div>
          ))}
        </div>
        <div className="legend-crit mono">
          {critical} / {nodes.length} links critical
        </div>
      </div>

      {viewMode === 'planner' && <CascadeControl />}
    </>
  )
}
