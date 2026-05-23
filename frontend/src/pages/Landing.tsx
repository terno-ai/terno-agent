import { useEffect, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchSdkConfig } from '../api'
import type { SdkConfigResponse } from '../api'

export default function Landing() {
  const navigate = useNavigate()
  const [input, setInput] = useState('')
  const [sdkConfig, setSdkConfig] = useState<SdkConfigResponse | null>(null)
  const [configError, setConfigError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchSdkConfig()
      .then((data) => {
        if (!cancelled) setSdkConfig(data)
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setConfigError(error instanceof Error ? error.message : 'Unable to load config')
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  const start = () => {
    const text = input.trim()
    if (!text) return
    const id = crypto.randomUUID()
    navigate(`/chat/${id}`, { state: { initialMessage: text } })
  }

  const submit = (e: FormEvent) => {
    e.preventDefault()
    start()
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      start()
    }
  }

  return (
    <main className="landing">
      <div className="landing-inner">
        <section className="landing-chat">
          <h1>Terno SDK Demo</h1>
          <p className="muted">Start a session backed by a FastAPI WebSocket and the Terno SDK.</p>
          <form className="landing-composer" onSubmit={submit}>
            <textarea
              autoFocus
              rows={3}
              placeholder="Ask the agent to inspect code, run a task, or explain the SDK setup..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
            />
            <div className="landing-composer-actions">
              <button
                type="button"
                className="icon-btn"
                title="Attach (coming soon)"
                disabled
                aria-label="Attach file"
              >
                📎
              </button>
              <button type="submit" className="primary" disabled={!input.trim()}>
                Send
              </button>
            </div>
          </form>
        </section>

        <aside className="sdk-panel" aria-label="SDK configuration">
          <div className="panel-heading">
            <h2>SDK config</h2>
            <span className={sdkConfig?.config.llm_api_key === 'configured' ? 'pill ok' : 'pill'}>
              {sdkConfig?.config.llm_api_key ?? 'loading'}
            </span>
          </div>
          {configError ? (
            <div className="config-error">{configError}</div>
          ) : (
            <dl className="config-grid">
              {sdkConfig ? (
                Object.entries(sdkConfig.config).map(([key, value]) => (
                  <div key={key}>
                    <dt>{key}</dt>
                    <dd>{String(value)}</dd>
                  </div>
                ))
              ) : (
                <div>
                  <dt>status</dt>
                  <dd>loading</dd>
                </div>
              )}
            </dl>
          )}
        </aside>
      </div>
    </main>
  )
}
