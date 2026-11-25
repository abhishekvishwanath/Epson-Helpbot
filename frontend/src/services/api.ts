import type { ChatResponse, DocumentInfo, SessionOut } from '../types/api'

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '/api'

export class ApiError extends Error {
  status: number
  code: string
  constructor(status: number, code: string, message: string) {
    super(message)
    this.status = status
    this.code = code
  }
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 95_000): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
    })
  } catch (err) {
    const aborted = err instanceof DOMException && err.name === 'AbortError'
    throw new ApiError(
      0,
      aborted ? 'timeout' : 'network',
      aborted
        ? 'The assistant is taking too long to respond. Please try again.'
        : "We couldn't reach the support service. Check your connection and try again.",
    )
  } finally {
    clearTimeout(timer)
  }
  if (res.status === 204) return undefined as T
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    const detail = body?.error ?? {}
    const validation = Array.isArray(body?.detail) ? 'Please check your message and try again.' : null
    throw new ApiError(res.status, detail.code ?? 'error', detail.message ?? validation ?? 'Something went wrong.')
  }
  return body as T
}

export const api = {
  chat: (message: string, sessionId: string | null, productModel?: string) =>
    request<ChatResponse>('/chat', {
      method: 'POST',
      body: JSON.stringify({ message, session_id: sessionId, product_model: productModel }),
    }),
  createSession: (productModel?: string) =>
    request<SessionOut>('/sessions', { method: 'POST', body: JSON.stringify({ product_model: productModel }) }),
  getSession: (id: string) => request<SessionOut>(`/sessions/${encodeURIComponent(id)}`),
  deleteSession: (id: string) => request<void>(`/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  documents: () => request<DocumentInfo[]>('/documents'),
}

export function pdfUrl(url: string | null): string | null {
  if (!url) return null
  return url.startsWith('/api') ? `${BASE}${url.slice(4)}` : url
}
