import { ExternalLink, FileText } from 'lucide-react'
import { pdfUrl } from '../services/api'
import type { Citation } from '../types/api'

export function pageText(c: Citation): string {
  const start = c.page_label ?? String(c.pdf_page)
  const end = c.page_label_end ?? String(c.pdf_page_end)
  return start === end ? `Page ${start}` : `Pages ${start}–${end}`
}

export function CitationList({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null
  return (
    <div className="mt-4 border-t border-slate-100 pt-3">
      <p className="mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">Sources</p>
      <ol className="space-y-1.5">
        {citations.map((c) => {
          const href = pdfUrl(c.url)
          const body = (
            <>
              <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-brand-50 text-[11px] font-semibold text-brand-700">
                {c.id}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate font-medium text-slate-800">{c.section}</span>
                <span className="flex items-center gap-1 text-xs text-slate-500">
                  <FileText className="h-3 w-3" aria-hidden />
                  {c.document_name} · {pageText(c)}
                </span>
              </span>
              {href && <ExternalLink className="mt-1 h-3.5 w-3.5 shrink-0 text-slate-400" aria-hidden />}
            </>
          )
          return (
            <li key={c.id} id={`cite-${c.chunk_id}`}>
              {href ? (
                <a href={href} target="_blank" rel="noreferrer"
                  className="flex items-start gap-2 rounded-lg p-1.5 text-sm transition hover:bg-slate-50">
                  {body}
                </a>
              ) : (
                <div className="flex items-start gap-2 p-1.5 text-sm">{body}</div>
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
