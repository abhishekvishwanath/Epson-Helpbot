import type { ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AlertTriangle } from 'lucide-react'
import { pdfUrl } from '../services/api'
import type { Citation } from '../types/api'
import { pageText } from './CitationList'

const SYMBOLS: Record<string, string> = {
  'Power button': '⏻ Power',
  'Stop button': '⊘ Stop',
  'Ink light': '💧 Ink light',
  icon: 'symbol',
}

/** Turn citation markers and manual pictogram placeholders into links the renderer styles. */
export function preprocess(text: string, citations: Citation[]): string {
  const ids = new Set(citations.map((c) => c.id))
  return text
    .replace(/\[(\d{1,2})\](?!\()/g, (m, n) => (ids.has(Number(n)) ? `[${n}](#cite-${n})` : m))
    .replace(/\*{0,2}\[(Power button|Stop button|Ink light|icon)\]\*{0,2}/g, (_, s) => `[${s}](#sym-${encodeURIComponent(s)})`)
}

function textOf(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textOf).join('')
  if (node && typeof node === 'object' && 'props' in node) return textOf((node as { props: { children?: ReactNode } }).props.children)
  return ''
}

export function AnswerMarkdown({ text, citations }: { text: string; citations: Citation[] }) {
  const byId = new Map(citations.map((c) => [c.id, c]))
  return (
    <div className="prose prose-slate max-w-none text-[15px] leading-relaxed prose-p:my-2 prose-ol:my-2 prose-ul:my-2 prose-li:my-0.5 prose-strong:text-ink prose-headings:text-ink prose-table:text-sm">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children }) {
            if (href?.startsWith('#cite-')) {
              const c = byId.get(Number(href.slice(6)))
              const url = c ? pdfUrl(c.url) : null
              return (
                <a
                  href={url ?? '#'}
                  target="_blank"
                  rel="noreferrer"
                  title={c ? `${c.section} — ${pageText(c)}` : undefined}
                  className="mx-0.5 inline-flex h-[18px] min-w-[18px] -translate-y-px items-center justify-center rounded bg-brand-50 px-1 align-middle text-[11px] font-semibold text-brand-700 no-underline hover:bg-brand-100"
                >
                  {children}
                </a>
              )
            }
            if (href?.startsWith('#sym-')) {
              const name = decodeURIComponent(href.slice(5))
              return (
                <span
                  title={name === 'icon' ? 'A symbol shown in the manual — open the cited page to see it' : name}
                  className="mx-0.5 inline-flex items-center rounded-md border border-slate-300 bg-slate-50 px-1.5 py-px align-baseline text-[12px] font-medium text-slate-700"
                >
                  {SYMBOLS[name] ?? name}
                </span>
              )
            }
            return <a href={href} target="_blank" rel="noreferrer">{children}</a>
          },
          blockquote({ children }) {
            return <Callout>{children}</Callout>
          },
          p({ children }) {
            const t = textOf(children).trim()
            if (/^(important|caution|warning|note)\b[:*]?/i.test(t)) return <Callout>{children}</Callout>
            return <p>{children}</p>
          },
        }}
      >
        {preprocess(text, citations)}
      </ReactMarkdown>
    </div>
  )
}

function Callout({ children }: { children: ReactNode }) {
  return (
    <div className="not-prose my-3 flex gap-2.5 rounded-xl border border-amber-200 bg-amber-50 px-3.5 py-2.5 text-[14px] leading-relaxed text-amber-900">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" aria-hidden />
      <div className="min-w-0 [&_p]:my-1">{children}</div>
    </div>
  )
}
