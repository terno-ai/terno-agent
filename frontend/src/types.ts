export type ServerEvent =
  | { type: 'ready'; history: HistoryMessage[] }
  | { type: 'turn_start' }
  | { type: 'iteration_start'; iteration: number }
  | { type: 'text_delta'; text: string }
  | { type: 'tool_call'; id: string; name: string; arguments: Record<string, unknown> }
  | { type: 'tool_result'; call_id: string; content: string; is_error: boolean }
  | { type: 'turn_end'; message: HistoryMessage }
  | { type: 'turn_complete'; history: HistoryMessage[] }
  | { type: 'cleared'; history: HistoryMessage[] }
  | { type: 'error'; message: string }

export type ClientFrame =
  | { type: 'user_message'; text: string }
  | { type: 'cancel' }
  | { type: 'clear' }

export interface HistoryMessage {
  role: string
  content?: unknown
  tool_calls?: Array<{ id: string; name: string; arguments: Record<string, unknown> }>
  results?: Array<{ call_id: string; content: string; is_error: boolean }>
}

export interface ToolCallDisplay {
  id: string
  name: string
  arguments: Record<string, unknown>
  result?: { content: string; is_error: boolean }
}

export interface ChatTurn {
  id: string
  role: 'user' | 'assistant'
  text: string
  toolCalls: ToolCallDisplay[]
  streaming: boolean
}
