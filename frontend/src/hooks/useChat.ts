import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../services/api'
import type { ChatMessage, SessionMessage } from '../types/api'

const STORAGE_KEY = 'printcare.session'

function readStoredSession(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

function storeSession(id: string | null) {
  try {
    if (id) localStorage.setItem(STORAGE_KEY, id)
    else localStorage.removeItem(STORAGE_KEY)
  } catch {
    /* storage unavailable: session lives for this tab only */
  }
}

let counter = 0
const nextId = () => `m${Date.now()}-${counter++}`

const fromSession = (m: SessionMessage): ChatMessage => ({
  id: nextId(),
  role: m.role,
  content: m.content,
  citations: m.citations,
  responseType: m.response_type,
})

export function useChat(productModel?: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [restoring, setRestoring] = useState(true)
  const inFlight = useRef(false)

  // Restore the previous conversation (if the session is still alive on the server).
  useEffect(() => {
    const stored = readStoredSession()
    if (!stored) {
      setRestoring(false)
      return
    }
    api
      .getSession(stored)
      .then((s) => {
        setSessionId(s.session_id)
        setMessages(s.messages.map(fromSession))
      })
      .catch(() => storeSession(null))
      .finally(() => setRestoring(false))
  }, [])

  const send = useCallback(
    async (text: string) => {
      const message = text.trim()
      if (!message || inFlight.current) return // prevents duplicate submissions
      inFlight.current = true
      setSending(true)
      setError(null)
      const userMsg: ChatMessage = { id: nextId(), role: 'user', content: message, citations: [] }
      setMessages((prev) => [...prev, userMsg])
      try {
        const res = await api.chat(message, sessionId, productModel)
        setSessionId(res.session_id)
        storeSession(res.session_id)
        setMessages((prev) => [
          ...prev,
          { id: nextId(), role: 'assistant', content: res.answer, citations: res.citations, responseType: res.response_type },
        ])
      } catch (err) {
        const msg = err instanceof ApiError ? err.message : 'Something went wrong. Please try again.'
        setError(msg)
        setMessages((prev) => prev.map((m) => (m.id === userMsg.id ? { ...m, failed: true } : m)))
      } finally {
        inFlight.current = false
        setSending(false)
      }
    },
    [sessionId, productModel],
  )

  const retry = useCallback(() => {
    const failed = [...messages].reverse().find((m) => m.failed)
    if (!failed) return
    setMessages((prev) => prev.filter((m) => m.id !== failed.id))
    void send(failed.content)
  }, [messages, send])

  const newConversation = useCallback(async () => {
    if (inFlight.current) return
    const old = sessionId
    setMessages([])
    setError(null)
    setSessionId(null)
    storeSession(null)
    if (old) api.deleteSession(old).catch(() => undefined)
  }, [sessionId])

  return { messages, sending, error, restoring, sessionId, send, retry, newConversation, dismissError: () => setError(null) }
}
