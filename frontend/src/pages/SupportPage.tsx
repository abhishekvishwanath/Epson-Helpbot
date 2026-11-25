import { useEffect, useRef, useState } from 'react'
import { Composer } from '../components/Composer'
import { ErrorBanner } from '../components/ErrorBanner'
import { Header } from '../components/Header'
import { MessageBubble, TypingIndicator } from '../components/MessageBubble'
import { Welcome } from '../components/Welcome'
import { useChat } from '../hooks/useChat'
import { api } from '../services/api'

const FALLBACK_PRODUCT = { name: 'Epson L3310 Series', model: 'L3310' }

export function SupportPage() {
  const [product, setProduct] = useState(FALLBACK_PRODUCT)
  const chat = useChat(product.model)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api
      .documents()
      .then((docs) => {
        const doc = docs[0]
        if (doc) setProduct({ name: `${doc.manufacturer} ${doc.product_model} Series`.trim(), model: doc.product_model })
      })
      .catch(() => undefined)
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [chat.messages.length, chat.sending])

  const empty = chat.messages.length === 0 && !chat.restoring

  return (
    <div className="flex h-full flex-col">
      <Header productName={product.name} onNewConversation={chat.newConversation} canReset={chat.messages.length > 0 && !chat.sending} />
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl px-4">
          {empty ? (
            <Welcome productName={product.name} onPick={chat.send} disabled={chat.sending} />
          ) : (
            <div className="space-y-5 py-6" aria-live="polite">
              {chat.messages.map((m) => <MessageBubble key={m.id} message={m} />)}
              {chat.sending && <TypingIndicator />}
            </div>
          )}
          {chat.error && (
            <div className="pb-4">
              <ErrorBanner message={chat.error} onRetry={chat.retry} onDismiss={chat.dismissError} />
            </div>
          )}
          <div ref={endRef} />
        </div>
      </main>
      <div className="border-t border-slate-200 bg-slate-50/95 pt-3">
        <Composer onSend={chat.send} disabled={chat.sending} />
      </div>
    </div>
  )
}
