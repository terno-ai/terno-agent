import { useEffect, useRef, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { useChat } from '../hooks/useChat'
import type { ChatTurn, ToolCallDisplay } from '../types'

function ToolCallView({ call }: { call: ToolCallDisplay }) {
  const [open, setOpen] = useState(false)
  const status = call.result
    ? call.result.is_error
      ? 'error'
      : 'done'
    : 'running'

  return (
    <div className={`tool tool-${status}`}>
      <button type="button" className="tool-head" onClick={() => setOpen((o) => !o)}>
        <span className="tool-icon">{open ? '▾' : '▸'}</span>
        <span className="tool-name">{call.name}</span>
        <span className="tool-status">{status}</span>
      </button>
      {open && (
        <div className="tool-body">
          <div className="tool-section">
            <div className="tool-label">arguments</div>
            <pre>{JSON.stringify(call.arguments, null, 2)}</pre>
          </div>
          {call.result && (
            <div className="tool-section">
              <div className="tool-label">result</div>
              <pre>{call.result.content}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function TurnView({ turn }: { turn: ChatTurn }) {
  return (
    <div className={`turn turn-${turn.role}`}>
      <div className="turn-role">{turn.role}</div>
      <div className="turn-body">
        {turn.toolCalls.length > 0 && (
          <div className="turn-tools">
            {turn.toolCalls.map((c) => (
              <ToolCallView key={c.id} call={c} />
            ))}
          </div>
        )}
        {turn.text && <div className="turn-text">{turn.text}</div>}
        {turn.streaming && !turn.text && turn.toolCalls.length === 0 && (
          <div className="turn-text dim">…</div>
        )}
      </div>
    </div>
  )
}

export default function Chat() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const location = useLocation()
  const initialMessage = (location.state as { initialMessage?: string } | null)?.initialMessage
  const { turns, status, busy, error, send, cancel, clear } = useChat(sessionId ?? '')
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const initialSentRef = useRef(false)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [turns])

  useEffect(() => {
    if (initialSentRef.current) return
    if (status !== 'open') return
    if (!initialMessage) return
    initialSentRef.current = true
    send(initialMessage)
    // Drop the route state so a refresh won't re-send.
    window.history.replaceState(null, '')
  }, [status, initialMessage, send])

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || busy) return
    send(input)
    setInput('')
  }

  return (
    <main className="chat">
      <header className="chat-header">
        <Link to="/" className="back">← New chat</Link>
        <div className="session-id">session {sessionId?.slice(0, 8)}</div>
        <div className={`status status-${status}`}>{status}</div>
        <button type="button" className="ghost" onClick={clear} disabled={busy}>
          Clear
        </button>
      </header>

      <div className="messages" ref={scrollRef}>
        {turns.length === 0 && status === 'open' && (
          <div className="empty">Say something to start.</div>
        )}
        {turns.map((t) => (
          <TurnView key={t.id} turn={t} />
        ))}
      </div>

      {error && <div className="error-banner">{error}</div>}

      <form className="composer" onSubmit={submit}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Message the agent…"
          rows={2}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit(e as unknown as React.FormEvent)
            }
          }}
        />
        <button
          type="button"
          className="icon-btn"
          title="Attach (coming soon)"
          disabled
          aria-label="Attach file"
        >
          📎
        </button>
        {busy ? (
          <button type="button" className="primary" onClick={cancel}>
            Cancel
          </button>
        ) : (
          <button type="submit" className="primary" disabled={!input.trim() || status !== 'open'}>
            Send
          </button>
        )}
      </form>
    </main>
  )
}
