// Timeline — bottom scrubber that drives the map through all 14 days at 5-min
// resolution. Pushes a new node/byId frame into the store on every tick;
// CityMap.jsx repaints itself off that (its existing `[byId]` effect) — this
// file never touches the map directly.
//
// Independence from the cascade replay: while a cascade is playing, the store
// silently drops this scrubber's frame writes (cascade owns link colors).
// The watcher below re-applies the current scrub position the moment the
// cascade ends, so the scrubber "resumes" without the person lifting a finger.

import { useEffect, useRef } from 'react'
import { useStore } from '../store/useStore'
import './timeline.css'

const SPEEDS = [1, 5, 30]

function minuteToClock(m) {
  const h = Math.floor(m / 60)
  const mm = String(m % 60).padStart(2, '0')
  const ap = h < 12 ? 'AM' : 'PM'
  const h12 = ((h + 11) % 12) + 1
  return `${h12}:${mm} ${ap}`
}

export default function Timeline() {
  const timelineAxis = useStore((s) => s.timelineAxis)
  const scrubDay = useStore((s) => s.scrubDay)
  const scrubMinute = useStore((s) => s.scrubMinute)
  const scrubPlaying = useStore((s) => s.scrubPlaying)
  const scrubSpeed = useStore((s) => s.scrubSpeed)
  const scrubLoading = useStore((s) => s.scrubLoading)
  const cascadePlaying = useStore((s) => s.cascadePlaying)

  const loadTimelineAxis = useStore((s) => s.loadTimelineAxis)
  const setScrubSpeed = useStore((s) => s.setScrubSpeed)
  const setScrubPosition = useStore((s) => s.setScrubPosition)
  const playScrub = useStore((s) => s.playScrub)
  const pauseScrub = useStore((s) => s.pauseScrub)

  useEffect(() => {
    loadTimelineAxis()
    return () => pauseScrub() // stop the interval if Timeline ever unmounts
  }, [loadTimelineAxis, pauseScrub])

  // Cascade-end watcher: the moment a cascade replay finishes, re-apply the
  // scrubber's current position so the map repaints from wherever the
  // scrubber was left — "resumes after", with no action needed from anyone.
  // Gated on `engagedRef`: if the person never touched the scrubber, CityMap
  // already reverts to the live snapshot on its own (its `byId` effect) —
  // firing this unconditionally would wrongly stomp that with Day 1 00:00.
  const engagedRef = useRef(false)
  const wasCascadePlaying = useRef(false)
  useEffect(() => {
    if (engagedRef.current && wasCascadePlaying.current && !cascadePlaying) {
      setScrubPosition(scrubDay, scrubMinute)
    }
    wasCascadePlaying.current = cascadePlaying
  }, [cascadePlaying, scrubDay, scrubMinute, setScrubPosition])

  if (!timelineAxis) return null

  const { days, minutes } = timelineAxis
  const perDay = minutes.length
  const totalTicks = days.length * perDay

  const dayIdx = Math.max(0, days.indexOf(scrubDay))
  const minuteIdx = Math.max(0, minutes.indexOf(scrubMinute))
  const flatIndex = dayIdx * perDay + minuteIdx

  function onSlide(e) {
    engagedRef.current = true
    const idx = Number(e.target.value)
    const di = Math.floor(idx / perDay)
    const mi = idx % perDay
    setScrubPosition(days[di], minutes[mi])
  }

  return (
    <div className="timeline-bar">
      <button
        className={`timeline-play ${scrubPlaying ? 'playing' : ''}`}
        onClick={() => { engagedRef.current = true; scrubPlaying ? pauseScrub() : playScrub() }}
        aria-label={scrubPlaying ? 'Pause' : 'Play'}
      >
        {scrubPlaying ? '❙❙' : '▶'}
      </button>

      <div className="timeline-readout mono">
        <span className="timeline-day">DAY {scrubDay}</span>
        <span className="timeline-sep">·</span>
        <span className="timeline-time">{minuteToClock(scrubMinute)}</span>
        {cascadePlaying && <span className="timeline-yield">cascade active — holding</span>}
        {!cascadePlaying && scrubLoading && <span className="timeline-loading">·</span>}
      </div>

      <input
        className="timeline-slider"
        type="range"
        min={0}
        max={totalTicks - 1}
        step={1}
        value={flatIndex}
        onChange={onSlide}
        style={{ '--pct': `${(flatIndex / (totalTicks - 1)) * 100}%` }}
        aria-label="Scrub through the 14-day timeline"
      />

      <div className="timeline-speeds" role="group" aria-label="Playback speed">
        {SPEEDS.map((sp) => (
          <button
            key={sp}
            className={scrubSpeed === sp ? 'active' : ''}
            onClick={() => setScrubSpeed(sp)}
          >
            {sp}×
          </button>
        ))}
      </div>
    </div>
  )
}
