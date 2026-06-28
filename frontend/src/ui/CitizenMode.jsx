// Citizen mode — plain-English commute view, cascade warning, and LLM assistant.
// Bible §9.3: no jargon, no raw numbers except minutes of delay.

import { useState, useEffect, useMemo } from 'react'
import { useStore, cascadeInfection, CASCADE_SECONDS_PER_LAG_MIN } from '../store/useStore'
import { stateColor, api } from '../data/api'
import './citizen.css'

// ── Helpers ──────────────────────────────────────────────────────────────── //

// Derive plain-English label + delay minutes from a node.
// queueS is present after a timeline frame has been applied; fall back to
// state-based estimates so the card is always populated.
function commuteStatus(node) {
  const queueMin = node.queueS != null ? node.queueS / 60 : null

  if (node.state === 'Healthy') {
    return { label: 'Clear', detail: 'No delays expected', delay: null, severity: 'clear' }
  }
  if (node.state === 'Stressed') {
    const mins = queueMin != null ? Math.max(1, Math.round(queueMin)) : 3
    return { label: 'Moderate', detail: `Expect about ${mins} min delay`, delay: mins, severity: 'moderate' }
  }
  if (node.state === 'Saturated') {
    const mins = queueMin != null ? Math.max(2, Math.round(queueMin)) : 8
    return { label: 'Heavy', detail: `Expect about ${mins} min delay`, delay: mins, severity: 'heavy' }
  }
  // Collapsed
  const mins = queueMin != null ? Math.max(5, Math.round(queueMin)) : 15
  return { label: 'Severe', detail: `Expect ${mins}+ min delay — avoid if possible`, delay: mins, severity: 'severe' }
}

// Severity → CSS modifier + color token
const SEVERITY_COLOR = {
  clear:    'var(--state-healthy)',
  moderate: 'var(--state-stressed)',
  heavy:    'var(--state-saturated)',
  severe:   'var(--state-collapsed)',
}

// ── My Commute cards ─────────────────────────────────────────────────────── //

function CommuteCards({ nodes }) {
  // Show the 5 worst-health links — the ones citizens most need to know about.
  const worst = useMemo(
    () => [...nodes].sort((a, b) => a.health - b.health).slice(0, 5),
    [nodes]
  )

  if (!worst.length) {
    return <p className="citizen-note">Loading road data…</p>
  }

  return (
    <div className="commute-cards">
      {worst.map((n) => {
        const s = commuteStatus(n)
        const color = SEVERITY_COLOR[s.severity]
        return (
          <div className="commute-card" key={n.id} style={{ '--c': color }}>
            <div className="commute-card-left">
              <div className="commute-road">Road {n.id}</div>
              <div className="commute-detail">{s.detail}</div>
            </div>
            <div className="commute-badge" style={{ '--c': color }}>
              <i className="commute-dot" />
              <span className="commute-label">{s.label}</span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ── Cascade warning banner ────────────────────────────────────────────────── //

function CascadeBanner() {
  const cascade = useStore((s) => s.cascade)
  const cascadePlaying = useStore((s) => s.cascadePlaying)
  const cascadeClock = useStore((s) => s.cascadeClock)
  const loadCascade = useStore((s) => s.loadCascade)

  // Ensure cascade data is loaded (it may not be if planner mode was never visited).
  useEffect(() => { loadCascade() }, [loadCascade])

  const active = cascadePlaying || cascadeClock > 0
  if (!active || !cascade) return null

  // Find the next downstream link that hasn't arrived yet.
  // If all have arrived, show the last one with "has reached".
  const inf = cascadeInfection({ cascade, cascadeClock, cascadePlaying })
  const unreached = cascade.downstream.filter((d) => !(d.link_id in inf.reached))
  const next = unreached.sort((a, b) => a.lag_minutes - b.lag_minutes)[0]

  let message
  if (next) {
    // How many real seconds until it arrives?
    const arrivesInSec = next.lag_minutes * CASCADE_SECONDS_PER_LAG_MIN - cascadeClock
    const arrivesInMin = Math.max(1, Math.ceil(arrivesInSec / CASCADE_SECONDS_PER_LAG_MIN))
    message = `Road ${next.link_id} may worsen in about ${arrivesInMin} min as congestion spreads from Road ${cascade.source_link}.`
  } else {
    // All links reached — show a "spreading complete" note
    const count = cascade.downstream.length
    message = `Congestion from Road ${cascade.source_link} has spread to ${count} nearby road${count !== 1 ? 's' : ''}. Allow extra time.`
  }

  return (
    <div className="cascade-banner" role="alert">
      <span className="cascade-banner-icon">⚠</span>
      <div className="cascade-banner-body">
        <div className="cascade-banner-head">Traffic alert</div>
        <div className="cascade-banner-text">{message}</div>
      </div>
    </div>
  )
}

// ── Ask the assistant ─────────────────────────────────────────────────────── //

function AskAssistant() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    const q = question.trim()
    if (!q || loading) return
    setLoading(true)
    setError(null)
    setAnswer(null)
    try {
      const res = await api.llmAsk({ question: q, audience: 'citizen' })
      setAnswer({ question: q, text: res.answer })
      setQuestion('')
    } catch {
      setError('Something went wrong — please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="ask-section">
      <div className="ask-label">Ask about the roads</div>
      <form className="ask-box" onSubmit={handleSubmit}>
        <input
          className="ask-input"
          type="text"
          placeholder="e.g. Is Road 36 clear right now?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
        />
        <button
          className="ask-btn"
          type="submit"
          disabled={loading || !question.trim()}
        >
          {loading ? <span className="ask-dots"><i/><i/><i/></span> : 'Ask'}
        </button>
      </form>

      {error && <div className="ask-error">{error}</div>}

      {answer && (
        <div className="ask-answer">
          <div className="ask-q">"{answer.question}"</div>
          <p className="ask-a">{answer.text}</p>
        </div>
      )}
    </div>
  )
}

// ── Root component ────────────────────────────────────────────────────────── //

export default function CitizenMode() {
  const nodes = useStore((s) => s.nodes)

  return (
    <div className="citizen">
      <div className="citizen-inner">
        <div className="eyebrow">Right now in Pangyo</div>
        <h1 className="citizen-h1">How are the roads?</h1>

        <CascadeBanner />

        <p className="citizen-sub">Roads to watch right now.</p>
        <CommuteCards nodes={nodes} />

        <AskAssistant />
      </div>
    </div>
  )
}
