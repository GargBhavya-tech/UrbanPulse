// UrbanPulse — API client + domain helpers.
// All backend access goes through here. In dev, Vite proxies /api -> :8000.

const BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function get(path) {
  const r = await fetch(`${BASE}${path}`)
  if (!r.ok) throw new Error(`${path} -> ${r.status}`)
  return r.json()
}
async function post(path, body) {
  const r = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`${path} -> ${r.status}`)
  return r.json()
}

export const api = {
  health: () => get('/health'),
  links: () => get('/links'),
  archetypes: () => get('/archetypes'),
  causalGraph: () => get('/echo/causal-graph'),
  ecosystemState: () => get('/echo/ecosystem-state'),
  cascades: (limit = 50) => get(`/echo/cascades?limit=${limit}`),
  cascadesDetailed: (sourceLink, day) => {
    const q = new URLSearchParams()
    if (sourceLink != null) q.set('source_link', sourceLink)
    if (day != null) q.set('day', day)
    const qs = q.toString()
    return get(`/echo/cascades/detailed${qs ? `?${qs}` : ''}`)
  },
  timelineAxis: () => get('/echo/timeline/axis'),
  timelineFrame: (day, minute) => get(`/echo/timeline?day=${day}&minute=${minute}`),
  counterfactual: () => get('/echo/counterfactual'),
  counterfactualLink: (linkId) => get(`/echo/counterfactual/${linkId}`),
  modelMetrics: () => get('/models/metrics'),
  llmGenerate: (req) => post('/llm/generate', req),
  llmAsk: (req) => post('/llm/ask', req),
  shapSummary: () => get('/shap/summary'),
  shapLink: (linkId) => get(`/shap/link/${linkId}`),
}

// --- Metabolic state -> color (single source of truth, mirrors tokens.css) ---
export const STATE_COLOR = {
  Healthy: '#2ee6a6',
  Stressed: '#ffd166',
  Saturated: '#ff8c42',
  Collapsed: '#ff3d7f',
}
export const STATE_ORDER = ['Healthy', 'Stressed', 'Saturated', 'Collapsed']

export function stateColor(state) {
  return STATE_COLOR[state] ?? '#56638a'
}

// Health (0-100) -> node radius. Worse health = bigger, angrier node.
export function healthRadius(health) {
  const h = Math.max(0, Math.min(100, health ?? 50))
  return 0.6 + (100 - h) / 100 * 1.6 // 0.6 (healthy) .. 2.2 (collapsed)
}

export const ARCHETYPE_GLYPH = {
  Landmine: '✺',
  Chronic: '◷',
  Saturator: '◉',
  Ghost: '◌',
  Commuter: '⇄',
  Chameleon: '◑',
  Unknown: '·',
}

// /echo/counterfactual/{id} returns one of two shapes: the July 1 Link 36
// "centrepiece" (richer — vehicle_hours_saved, cascade_prevented, narrative)
// or a generic per-link policy-simulation record. Normalize both into one
// display shape so the deep-dive card doesn't need to know which it got.
export function normalizeCounterfactual(rec) {
  if (!rec) return null
  const isCentrepiece = 'observed_queue_s' in rec
  return {
    intervention: isCentrepiece ? rec.intervention : rec.intervention_description,
    baselineQueueS: isCentrepiece ? rec.observed_queue_s : rec.queue_control_s,
    counterfactualQueueS: isCentrepiece ? rec.counterfactual_queue_s : rec.queue_treated_s,
    queueReductionPct: rec.queue_reduction_pct,
    vehicleHoursSaved: isCentrepiece ? rec.vehicle_hours_saved : null,
    cascadePrevented: isCentrepiece ? rec.cascade_prevented : null,
    narrative: isCentrepiece ? rec.narrative : null,
    isCentrepiece,
  }
}
