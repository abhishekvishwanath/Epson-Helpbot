import { AlertCircle, Bot } from 'lucide-react'
import type { ChatMessage } from '../types/api'
import { AnswerMarkdown } from './AnswerMarkdown'
import { CitationList } from './CitationList'

export function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] sm:max-w-[75%]">
          <div className="rounded-2xl rounded-br-md bg-brand-600 px-4 py-2.5 text-[15px] leading-relaxed whitespace-pre-wrap text-white shadow-sm">
            {message.content}
          </div>
          {message.failed && (
            <p className="mt-1 flex items-center justify-end gap-1 text-xs text-red-600">
              <AlertCircle className="h-3.5 w-3.5" aria-hidden /> Not sent
            </p>
          )}
        </div>
      </div>
    )
  }
  const muted = message.responseType === 'error'
  return (
    <div className="flex gap-3">
      <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-white text-brand-600 ring-1 ring-slate-200">
        <Bot className="h-4.5 w-4.5" aria-hidden />
      </div>
      <div
        data-testid="assistant-message"
        className={`min-w-0 flex-1 rounded-2xl rounded-tl-md border px-4 py-3 shadow-sm sm:px-5 ${muted ? 'border-amber-200 bg-amber-50/60' : 'border-slate-200 bg-white'}`}
      >
        <AnswerMarkdown text={message.content} citations={message.citations} />
        <CitationList citations={message.citations} />
      </div>
    </div>
  )
}

export function TypingIndicator() {
  return (
    <div className="flex gap-3" role="status" aria-live="polite">
      <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-white text-brand-600 ring-1 ring-slate-200">
        <Bot className="h-4.5 w-4.5" aria-hidden />
      </div>
      <div className="rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3 shadow-sm">
        <span className="sr-only">Searching the manual…</span>
        <div className="flex items-center gap-2 text-sm text-slate-500" aria-hidden>
          <span className="flex gap-1">
            {[0, 150, 300].map((d) => (
              <span key={d} className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand-500" style={{ animationDelay: `${d}ms` }} />
            ))}
          </span>
          Checking the manual…
        </div>
      </div>
    </div>
  )
}
