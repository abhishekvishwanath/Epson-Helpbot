import { RefreshCw, WifiOff, X } from 'lucide-react'

interface Props {
  message: string
  onRetry: () => void
  onDismiss: () => void
}

export function ErrorBanner({ message, onRetry, onDismiss }: Props) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
      <WifiOff className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <p className="flex-1">{message}</p>
      <button type="button" onClick={onRetry} className="inline-flex items-center gap-1 font-medium text-red-700 hover:underline">
        <RefreshCw className="h-3.5 w-3.5" aria-hidden /> Retry
      </button>
      <button type="button" onClick={onDismiss} aria-label="Dismiss" className="text-red-500 hover:text-red-700">
        <X className="h-4 w-4" aria-hidden />
      </button>
    </div>
  )
}
