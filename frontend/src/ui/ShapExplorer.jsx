// SHAP Explorer — Stage 3 deep-dive drawer section.
// For Link 36 and Link 37: per-link waterfall (top-3 feature bars, plain-
// English sentences, expandable PNG). For all other links: global feature
// importance bar chart + a note that per-link detail is available for 36/37.
// Fetches once per link then caches in local state (not the global store —
// SHAP data is large and only needed when the drawer is open).

import { useState, useEffect, useRef } from 'react'
import { api } from '../data/api'
import './shap.css'

// ── Helpers ──────────────────────────────────────────────────────────────── //

// Sigmoid of log-odds → probability %
function logOddsToProb(lo) {
  return Math.round(100 / (1 + Math.exp(-lo)))
}

// Map a raw SHAP value to a bar width (%) capped at 100.
// The max SHAP in translations.json for link 36 is ~6.17 → use 7 as ceiling.
const SHAP_SCALE = 7
function shapPct(shap) {
  return Math.min(100, (Math.abs(shap) / SHAP_SCALE) * 100)
}

// Feature name → short human label
const FEAT_LABEL = {
  link_congestion_rate: 'Historical congestion rate',
  congestion_index:     'Congestion index',
  mean_occup:           'Road occupancy',
  minute_of_day:        'Time of day',
  mean_queue_s:         'Queue length',
  max_queue_s:          'Peak queue',
  mean_speed_kmh:       'Average speed',
  mean_speed_div:       'Speed divergence',
  road_health_score:    'Road health score',
  sin_hour:             'Time of day (cyclical)',
  cos_hour:             'Time of day (cyclical)',
  total_vehs:           'Vehicle count',
  lane_active_count:    'Active lanes',
  lane6_active:         'Lane 6 active',
  speed_var_across_lanes: 'Lane speed variance',
  is_am_peak:           'Morning peak hour',
  is_pm_peak:           'Evening peak hour',
  is_weekend:           'Weekend',
}
const featLabel = (f) => FEAT_LABEL[f] ?? f.replace(/_/g, ' ')

// ── Per-link waterfall panel ─────────────────────────────────────────────── //

function WaterfallPanel({ data, linkId }) {
  const [showPng, setShowPng] = useState(false)
  const prob = logOddsToProb(data.prediction_log_odds)
  const baseProb = logOddsToProb(data.base_value)

  return (
    <div className="shap-waterfall">
      <div className="shap-prob-row">
        <div className="shap-prob-block">
          <span className="shap-prob-lbl">Network baseline</span>
          <span className="shap-prob-val">{baseProb}%</span>
        </div>
        <span className="shap-prob-arrow">→</span>
        <div className="shap-prob-block">
          <span className="shap-prob-lbl">This road</span>
          <span className={`shap-prob-val ${prob >= 70 ? 'shap-danger' : prob >= 40 ? 'shap-warn' : 'shap-ok'}`}>
            {prob}%
          </span>
        </div>
        <span className="shap-prob-label-right">congestion risk</span>
      </div>

      <div className="shap-feat-list">
        {data.top_features.map((f) => {
          const positive = f.shap > 0
          const pct = shapPct(f.shap)
          return (
            <div className="shap-feat-row" key={f.feature}>
              <div className="shap-feat-meta">
                <span className="shap-feat-name">{featLabel(f.feature)}</span>
                <span className={`shap-feat-dir ${positive ? 'shap-up' : 'shap-down'}`}>
                  {positive ? '▲ increases risk' : '▼ reduces risk'}
                </span>
              </div>
              <div className="shap-bar-track">
                <div
                  className={`shap-bar ${positive ? 'shap-bar-pos' : 'shap-bar-neg'}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <p className="shap-plain">{f.plain_english}</p>
            </div>
          )
        })}
      </div>

      {data.waterfall_png && (
        <div className="shap-png-section">
          <button className="link-btn" onClick={() => setShowPng((v) => !v)}>
            {showPng ? '▲ Hide waterfall chart' : '▼ Show full waterfall chart'}
          </button>
          {showPng && (
            <img
              className="shap-png"
              src={data.waterfall_png}
              alt={`SHAP waterfall for Link ${linkId}`}
            />
          )}
        </div>
      )}
    </div>
  )
}

// ── Global importance panel (for links without per-link SHAP) ────────────── //

function GlobalPanel({ summary }) {
  const [showImg, setShowImg] = useState(null) // null | 'beeswarm' | 'importance'

  // Build a ranked importance list from the translations data — we don't have
  // global mean|SHAP| values directly, but we can show a note + the PNGs.
  return (
    <div className="shap-global">
      <p className="shap-global-note">
        Detailed feature analysis is precomputed for Roads{' '}
        {summary.computed_links.join(' and ')}. The charts below show which
        features drive congestion risk across the whole network.
      </p>

      <div className="shap-img-btns">
        <button
          className={`shap-img-tab ${showImg === 'importance' ? 'active' : ''}`}
          onClick={() => setShowImg((v) => (v === 'importance' ? null : 'importance'))}
        >
          Feature importance
        </button>
        <button
          className={`shap-img-tab ${showImg === 'beeswarm' ? 'active' : ''}`}
          onClick={() => setShowImg((v) => (v === 'beeswarm' ? null : 'beeswarm'))}
        >
          Beeswarm summary
        </button>
      </div>

      {showImg === 'importance' && (
        <img
          className="shap-png"
          src={summary.plots.importance_bar}
          alt="SHAP feature importance bar chart"
        />
      )}
      {showImg === 'beeswarm' && (
        <img
          className="shap-png"
          src={summary.plots.beeswarm}
          alt="SHAP beeswarm summary plot"
        />
      )}

      <p className="shap-global-hint eyebrow">
        Select Road 36 or Road 37 for per-road feature breakdown
      </p>
    </div>
  )
}

// ── Root component ────────────────────────────────────────────────────────── //

export default function ShapExplorer({ linkId }) {
  const [open, setOpen] = useState(false)
  const [linkData, setLinkData] = useState(null)   // per-link SHAP | null | 'not-available'
  const [summary, setSummary] = useState(null)      // global summary
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Cache fetched data by linkId so switching nodes is instant after first load
  const cacheRef = useRef({})

  useEffect(() => {
    if (!open) return
    let cancelled = false

    async function fetchAll() {
      setError(null)
      setLoading(true)

      // Always fetch summary if not already loaded
      let sum = summary
      if (!sum) {
        try {
          sum = await api.shapSummary()
          if (!cancelled) setSummary(sum)
        } catch (e) {
          if (!cancelled) setError('SHAP artifacts not available — run python scripts/b5_shap.py first.')
          if (!cancelled) setLoading(false)
          return
        }
      }

      // Per-link: check cache first
      if (linkId in cacheRef.current) {
        if (!cancelled) setLinkData(cacheRef.current[linkId])
        if (!cancelled) setLoading(false)
        return
      }

      try {
        const ld = await api.shapLink(linkId)
        cacheRef.current[linkId] = ld
        if (!cancelled) setLinkData(ld)
      } catch (e) {
        const is404 = /404/.test(String(e.message ?? e))
        const val = is404 ? 'not-available' : null
        cacheRef.current[linkId] = val
        if (!cancelled) setLinkData(val)
        if (!cancelled && !is404) setError('Could not load SHAP data for this link.')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    fetchAll()
    return () => { cancelled = true }
  }, [open, linkId]) // re-fetch when link changes while panel is open

  // Reset per-link data when switching links (keep summary, check cache)
  useEffect(() => {
    if (!open) return
    if (linkId in cacheRef.current) {
      setLinkData(cacheRef.current[linkId])
    } else {
      setLinkData(null)
    }
  }, [linkId])

  return (
    <section className="card shap-card">
      <button className="shap-toggle" onClick={() => setOpen((v) => !v)}>
        <span className="eyebrow">SHAP · why is this road congested?</span>
        <span className="shap-chevron">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="shap-body">
          {loading && <div className="shap-loading">Loading explainability data…</div>}
          {error && <div className="shap-error">{error}</div>}

          {!loading && !error && linkData && linkData !== 'not-available' && (
            <>
              <div className="shap-context eyebrow" style={{ marginBottom: 12 }}>
                {linkData.label}
              </div>
              <WaterfallPanel data={linkData} linkId={linkId} />
            </>
          )}

          {!loading && !error && linkData === 'not-available' && summary && (
            <GlobalPanel summary={summary} />
          )}

          {!loading && !error && !linkData && summary && (
            <GlobalPanel summary={summary} />
          )}
        </div>
      )}
    </section>
  )
}
