import { Printer, RotateCcw } from 'lucide-react'

interface Props {
  productName: string
  onNewConversation: () => void
  canReset: boolean
}

export function Header({ productName, onNewConversation, canReset }: Props) {
  return (
    <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-4xl items-center justify-between gap-3 px-4">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-brand-600 text-white shadow-sm">
            <Printer className="h-5 w-5" aria-hidden />
          </div>
          <div className="min-w-0">
            <p className="text-sm font-semibold tracking-tight text-ink">PrintCare Support</p>
            <p className="truncate text-xs text-slate-500" data-testid="product-name">{productName}</p>
          </div>
        </div>
        <button
          type="button"
          onClick={onNewConversation}
          disabled={!canReset}
          className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:border-brand-500 hover:text-brand-600 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <RotateCcw className="h-4 w-4" aria-hidden />
          <span>New conversation</span>
        </button>
      </div>
    </header>
  )
}
