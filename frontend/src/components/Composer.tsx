import { ArrowUp } from 'lucide-react'
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'

interface Props {
  onSend: (text: string) => void
  disabled: boolean
}

const MAX = 2000

export function Composer({ onSend, disabled }: Props) {
  const [text, setText] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [text])

  const submit = (e?: FormEvent) => {
    e?.preventDefault()
    if (disabled || !text.trim()) return
    onSend(text)
    setText('')
  }

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <form onSubmit={submit} className="mx-auto w-full max-w-4xl px-4 pb-4">
      <div className="flex items-end gap-2 rounded-2xl border border-slate-300 bg-white p-2 shadow-sm transition focus-within:border-brand-500 focus-within:ring-4 focus-within:ring-brand-100">
        <label htmlFor="composer" className="sr-only">Describe your printer problem</label>
        <textarea
          id="composer"
          ref={ref}
          rows={1}
          value={text}
          maxLength={MAX}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          placeholder="Describe your printer problem…"
          className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] text-ink outline-none placeholder:text-slate-400"
        />
        <button
          type="submit"
          aria-label="Send message"
          disabled={disabled || !text.trim()}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-brand-600 text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          <ArrowUp className="h-5 w-5" aria-hidden />
        </button>
      </div>
      <p className="mt-2 text-center text-[11px] text-slate-400">
        Answers are based on the official User's Guide. For problems it can't solve, contact your manufacturer's support.
      </p>
    </form>
  )
}
