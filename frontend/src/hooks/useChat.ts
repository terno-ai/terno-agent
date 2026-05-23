import { useCallback, useEffect, useRef, useState } from 'react'
import { wsUrl } from '../api'
import type { ChatTurn, ClientFrame, HistoryMessage, ServerEvent, ToolCallDisplay } from '../types'

type Status = 'connecting' | 'open' | 'closed'

function extractText(content: unknown): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .map((p) => (p && typeof p === 'object' && 'text' in p ? String((p as { text: unknown }).text ?? '') : ''))
      .join('')
  }
  return ''
}

function historyToTurns(history: HistoryMessage[]): ChatTurn[] {
  const turns: ChatTurn[] = []
  const toolResults = new Map<string, { content: string; is_error: boolean }>()

  for (const msg of history) {
    if (msg.role === 'tool' && Array.isArray(msg.results)) {
      for (const r of msg.results) {
        toolResults.set(r.call_id, { content: r.content, is_error: r.is_error })
      }
    }
  }

  let counter = 0
  for (const msg of history) {
    if (msg.role === 'user') {
      turns.push({
        id: `h-${counter++}`,
        role: 'user',
        text: extractText(msg.content),
        toolCalls: [],
        streaming: false,
      })
    } else if (msg.role === 'assistant') {
      const calls: ToolCallDisplay[] = (msg.tool_calls ?? []).map((c) => ({
        id: c.id,
        name: c.name,
        arguments: c.arguments,
        result: toolResults.get(c.id),
      }))
      turns.push({
        id: `h-${counter++}`,
        role: 'assistant',
        text: extractText(msg.content),
        toolCalls: calls,
        streaming: false,
      })
    }
  }
  return turns
}

export function useChat(sessionId: string) {
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [status, setStatus] = useState<Status>('connecting')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const streamingIdRef = useRef<string | null>(null)

  const handleEvent = useCallback((event: ServerEvent) => {
    switch (event.type) {
      case 'ready':
      case 'turn_complete':
      case 'cleared':
        setTurns(historyToTurns(event.history))
        if (event.type !== 'ready') {
          setBusy(false)
          streamingIdRef.current = null
        }
        break

      case 'turn_start': {
        setBusy(true)
        const id = `t-${Date.now()}`
        streamingIdRef.current = id
        setTurns((prev) => [
          ...prev,
          { id, role: 'assistant', text: '', toolCalls: [], streaming: true },
        ])
        break
      }

      case 'text_delta': {
        const id = streamingIdRef.current
        if (!id) return
        setTurns((prev) =>
          prev.map((t) => (t.id === id ? { ...t, text: t.text + event.text } : t)),
        )
        break
      }

      case 'tool_call': {
        const id = streamingIdRef.current
        if (!id) return
        setTurns((prev) =>
          prev.map((t) =>
            t.id === id
              ? {
                  ...t,
                  toolCalls: [
                    ...t.toolCalls,
                    { id: event.id, name: event.name, arguments: event.arguments },
                  ],
                }
              : t,
          ),
        )
        break
      }

      case 'tool_result': {
        const id = streamingIdRef.current
        if (!id) return
        setTurns((prev) =>
          prev.map((t) =>
            t.id === id
              ? {
                  ...t,
                  toolCalls: t.toolCalls.map((c) =>
                    c.id === event.call_id
                      ? { ...c, result: { content: event.content, is_error: event.is_error } }
                      : c,
                  ),
                }
              : t,
          ),
        )
        break
      }

      case 'turn_end':
        // turn_complete will overwrite with authoritative history
        break

      case 'error':
        setError(event.message)
        setBusy(false)
        break
    }
  }, [])

  useEffect(() => {
    let aborted = false
    const ws = new WebSocket(wsUrl(sessionId))
    wsRef.current = ws

    ws.onopen = () => {
      if (aborted) {
        ws.close()
        return
      }
      setStatus('open')
    }
    ws.onclose = () => {
      if (!aborted) setStatus('closed')
    }
    ws.onerror = () => {
      if (!aborted) setError('WebSocket error')
    }

    ws.onmessage = (e) => {
      if (aborted) return
      let event: ServerEvent
      try {
        event = JSON.parse(e.data) as ServerEvent
      } catch {
        return
      }
      handleEvent(event)
    }

    return () => {
      aborted = true
      // Closing a still-CONNECTING socket logs a noisy "closed before
      // established" message; defer the close to onopen via the
      // aborted flag instead.
      if (ws.readyState === WebSocket.OPEN) ws.close()
      wsRef.current = null
    }
  }, [sessionId, handleEvent])

  const send = useCallback((text: string) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN || !text.trim()) return
    const userTurn: ChatTurn = {
      id: `u-${Date.now()}`,
      role: 'user',
      text,
      toolCalls: [],
      streaming: false,
    }
    setTurns((prev) => [...prev, userTurn])
    setError(null)
    const frame: ClientFrame = { type: 'user_message', text }
    ws.send(JSON.stringify(frame))
  }, [])

  const cancel = useCallback(() => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    const frame: ClientFrame = { type: 'cancel' }
    ws.send(JSON.stringify(frame))
  }, [])

  const clear = useCallback(() => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    const frame: ClientFrame = { type: 'clear' }
    ws.send(JSON.stringify(frame))
  }, [])

  return { turns, status, busy, error, send, cancel, clear }
}
