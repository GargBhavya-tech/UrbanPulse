import { useEffect } from 'react'
import { useStore } from './store/useStore'
import CityMap from './scene/CityMap'
import HUD from './ui/HUD'
import Timeline from './ui/Timeline'
import DeepDive from './ui/DeepDive'
import CitizenMode from './ui/CitizenMode'
import './ui/app.css'

function Boot({ status, error, onRetry }) {
  return (
    <div className="boot">
      <div className="boot-mark">◉</div>
      {status === 'error' ? (
        <>
          <div className="boot-title">Can't reach the observatory</div>
          <p className="boot-msg mono">{error}</p>
          <p className="boot-hint">
            Start the API: <code>python scripts/11_serve.py</code>
          </p>
          <button className="boot-retry" onClick={onRetry}>Retry</button>
        </>
      ) : (
        <>
          <div className="boot-title">Mapping the network…</div>
          <div className="boot-bar"><i /></div>
        </>
      )}
    </div>
  )
}

export default function App() {
  const status = useStore((s) => s.status)
  const error = useStore((s) => s.error)
  const viewMode = useStore((s) => s.viewMode)
  const load = useStore((s) => s.load)

  useEffect(() => { load() }, [load])

  if (status !== 'ready') {
    return <Boot status={status} error={error} onRetry={load} />
  }

  return (
    <div className="app">
      <CityMap />
      <HUD />
      <Timeline />
      {viewMode === 'planner' ? <DeepDive /> : <CitizenMode />}
    </div>
  )
}
