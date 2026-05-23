const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

function apiUrl(path: string): string {
  return new URL(path, API_BASE).toString()
}

export function wsUrl(sessionId: string): string {
  const httpUrl = new URL(API_BASE)
  const protocol = httpUrl.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${httpUrl.host}/chat/${sessionId}`
}

export async function fetchSdkConfig(): Promise<SdkConfigResponse> {
  const response = await fetch(apiUrl('/api/config'))
  if (!response.ok) {
    throw new Error(`Config request failed with ${response.status}`)
  }
  return response.json() as Promise<SdkConfigResponse>
}

export interface SdkConfigResponse {
  config: Record<string, string | number | boolean>
}
