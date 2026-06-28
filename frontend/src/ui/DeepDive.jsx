// Deep-dive drawer — slides in when a node is selected. Shows the link's
// metabolic state, archetype card, and causal connections (upstream causes /
// downstream effects). This is the planner's "Link Deep Dive" (Bible §9.2 p2),
// living as an overlay on the constellation rather than a separate page.

import { useEffect, useMemo } from 'react'
import { useStore, selectSelectedNode, linkSource, linkTarget } from '../store/useStore'
import { ARCHETYPE_GLYPH, stateColor, normalizeCounterfactual } from '../data/api'
import ShapExplorer from './ShapExplorer'
import './drawer.css'

function StatePill({ state }) {
  return (
    <span className="pill" style={{ '--c': stateColor(state) }}>
      <i className="dot" /> {state}
    </span>
  )
}

function ConnRow({ conn, kind, onSelect }) {
  return (
    <button className="conn-row" onClick={() => onSelect(conn.id)}>
      <span className="conn-arrow">{kind === 'upstream' ? '←' : '→'}</span>
      <span className="mono conn-id">LINK {conn.id}</span>
      <span className="conn-meta mono">
        {conn.lag}m · r={conn.strength.toFixed(2)}
      </span>
    </button>
  )
}

// --- Counterfactual card --------------------------------------------------- //
function CounterfactualCard({ linkId }) {
  const rawByLink = useStore((s) => s.counterfactualByLink)
  const loadingByLink = useStore((s) => s.counterfactualLoading)
  const loadCounterfactual = useStore((s) => s.loadCounterfactual)

  useEffect(() => { loadCounterfactual(linkId) }, [linkId, loadCounterfactual])

  const loading = !!loadingByLink[linkId]
  const hasFetched = linkId in rawByLink
  const cf = useMemo(() => normalizeCounterfactual(rawByLink[linkId]), [rawByLink, linkId])

  return (
    <section className="card">
      <div className="eyebrow" style={{ marginBottom: 10 }}>Counterfactual — what-if intervention</div>

      {loading && <div className="cf-empty">running structural causal model…</div>}

      {!loading && hasFetched && !cf && (
        <div className="cf-empty">no counterfactual intervention modeled for this link</div>
      )}

      {!loading && cf && (
        <>
          <p className="card-desc" style={{ marginBottom: 14 }}>{cf.intervention}</p>

          <div className="cf-row">
            <div className="cf-stat">
              <span className="cf-stat-lbl">Observed queue</span>
              <span className="cf-stat-val mono">{Math.round(cf.baselineQueueS ?? 0)}s</span>
            </div>
            <span className="cf-arrow">→</span>
            <div className="cf-stat">
              <span className="cf-stat-lbl">Counterfactual</span>
              <span className="cf-stat-val mono cf-good">{Math.round(cf.counterfactualQueueS ?? 0)}s</span>
            </div>
          </div>

          <div className="cf-reduction mono">
            <b>−{(cf.queueReductionPct ?? 0).toFixed(1)}%</b> queue delay
          </div>

          {cf.vehicleHoursSaved != null && (
            <div className="cf-metric">
              <span className="eyebrow">Network benefit</span>
              <span className="cf-metric-val mono">{cf.vehicleHoursSaved.toFixed(0)} vehicle-hours saved</span>
            </div>
          )}

          {cf.cascadePrevented != null && (
            <span className={`pill cf-pill ${cf.cascadePrevented ? 'cf-pill-good' : 'cf-pill-bad'}`}>
              <i className="dot" /> {cf.cascadePrevented ? 'Cascade prevented' : 'Cascade not prevented'}
            </span>
          )}

          {!cf.isCentrepiece && (
            <div className="cf-foot eyebrow" style={{ marginTop: 10 }}>
              policy simulation · no historical cascade event in this window
            </div>
          )}
        </>
      )}
    </section>
  )
}

// --- LLM briefing + Q&A card ----------------------------------------------- //
function LlmBriefingCard({ linkId }) {
  const briefingByLink = useStore((s) => s.llmBriefingByLink)
  const briefingLoadingByLink = useStore((s) => s.llmBriefingLoading)
  const briefingErrorByLink = useStore((s) => s.llmBriefingError)
  const loadPlannerBriefing = useStore((s) => s.loadPlannerBriefing)

  const llmQuestion = useStore((s) => s.llmQuestion)
  const setLlmQuestion = useStore((s) => s.setLlmQuestion)
  const answerByLink = useStore((s) => s.llmAnswerByLink)
  const asking = useStore((s) => s.llmAsking)
  const askError = useStore((s) => s.llmAskError)
  const askLLM = useStore((s) => s.askLLM)

  const briefing = briefingByLink[linkId]
  const briefingLoading = !!briefingLoadingByLink[linkId]
  const briefingError = briefingErrorByLink[linkId]
  const answer = answerByLink[linkId]

  function onAsk(e) {
    e.preventDefault()
    askLLM(linkId, llmQuestion)
  }

  return (
    <section className="card">
      <div className="eyebrow" style={{ marginBottom: 10 }}>ECHO · LLM briefing</div>

      {!briefing && !briefingLoading && (
        <button className="btn-primary" onClick={() => loadPlannerBriefing(linkId)}>
          ▤ Generate planner briefing
        </button>
      )}

      {briefingLoading && <div className="cf-empty">drafting briefing…</div>}

      {briefingError && !briefingLoading && (
        <div className="llm-error">
          briefing failed: {briefingError}
          <button className="link-btn" onClick={() => loadPlannerBriefing(linkId)}>retry</button>
        </div>
      )}

      {briefing && !briefingLoading && (
        <>
          <p className="card-desc llm-text">{briefing}</p>
          <button className="link-btn" onClick={() => loadPlannerBriefing(linkId)}>regenerate</button>
        </>
      )}

      <form className="llm-ask" onSubmit={onAsk}>
        <input
          className="llm-input"
          type="text"
          placeholder="Ask about this link…"
          value={llmQuestion}
          onChange={(e) => setLlmQuestion(e.target.value)}
        />
        <button className="btn-primary btn-sm" type="submit" disabled={asking || !llmQuestion.trim()}>
          {asking ? '…' : 'Ask'}
        </button>
      </form>

      {askError && <div className="llm-error">{askError}</div>}

      {answer && (
        <div className="llm-answer">
          <div className="conn-meta mono" style={{ marginBottom: 4 }}>"{answer.question}"</div>
          <p className="card-desc llm-text">{answer.answer}</p>
        </div>
      )}
    </section>
  )
}

export default function DeepDive() {
  const node = useStore(selectSelectedNode)
  const links = useStore((s) => s.links)
  const neighbors = useMemo(() => {
    if (node == null) return { upstream: [], downstream: [] }
    const upstream = links
      .filter((l) => linkTarget(l) === node.id)
      .map((l) => ({ id: linkSource(l), lag: l.lag, strength: l.strength }))
    const downstream = links
      .filter((l) => linkSource(l) === node.id)
      .map((l) => ({ id: linkTarget(l), lag: l.lag, strength: l.strength }))
    return { upstream, downstream }
  }, [node, links])
  const archetypes = useStore((s) => s.archetypes)
  const setSelected = useStore((s) => s.setSelected)
  const clear = useStore((s) => s.clearSelected)

  const open = !!node
  const arch = node ? archetypes[node.id] : null

  return (
    <aside className={`drawer ${open ? 'open' : ''}`}>
      {node && (
        <>
          <header className="drawer-head">
            <div>
              <div className="eyebrow">Road link</div>
              <h2 className="drawer-title">Link {node.id}</h2>
            </div>
            <button className="icon-btn" onClick={clear} aria-label="Close">✕</button>
          </header>

          <div className="drawer-row">
            <StatePill state={node.state} />
            <span className="health mono">
              <b>{node.health.toFixed(0)}</b><span className="health-max">/100</span>
              <span className="health-lbl">health</span>
            </span>
          </div>

          {/* Archetype card */}
          {node.archetype && (
            <section className="card">
              <div className="card-head">
                <span className="glyph">{ARCHETYPE_GLYPH[node.archetype] ?? '·'}</span>
                <div>
                  <div className="eyebrow">Personality archetype</div>
                  <div className="card-title">{node.archetype}</div>
                </div>
              </div>
              {arch?.description && <p className="card-desc">{arch.description}</p>}
              {arch?.policy_class && (
                <div className="policy">
                  <span className="eyebrow">Intervention class</span>
                  <span className="policy-text">{arch.policy_class}</span>
                </div>
              )}
              {arch && (
                <div className="stability mono">
                  stability {(arch.stability_score ?? 0).toFixed(2)}
                  {arch.stable ? ' · stable' : ' · unstable ⚠'}
                </div>
              )}
            </section>
          )}

          {/* Causal connections */}
          <section className="card">
            <div className="eyebrow" style={{ marginBottom: 10 }}>Causal connections</div>

            <div className="conn-group">
              <div className="conn-label">Caused by ({neighbors.upstream.length})</div>
              {neighbors.upstream.length === 0 && <div className="conn-empty">no upstream causes</div>}
              {neighbors.upstream.map((c) => (
                <ConnRow key={`u${c.id}`} conn={c} kind="upstream" onSelect={setSelected} />
              ))}
            </div>

            <div className="conn-group">
              <div className="conn-label">Spreads to ({neighbors.downstream.length})</div>
              {neighbors.downstream.length === 0 && <div className="conn-empty">no downstream effects</div>}
              {neighbors.downstream.map((c) => (
                <ConnRow key={`d${c.id}`} conn={c} kind="downstream" onSelect={setSelected} />
              ))}
            </div>
          </section>

          {/* Counterfactual + LLM (Stage 2) */}
          <CounterfactualCard linkId={node.id} />
          <LlmBriefingCard linkId={node.id} />

          {/* SHAP Explainability (Stage 3) */}
          <ShapExplorer linkId={node.id} />

          <p className="drawer-foot eyebrow">
            ECHO · UrbanPulse Observatory
          </p>
        </>
      )}
    </aside>
  )
}
