import { BookOpen, Sparkles } from 'lucide-react'

export const SUGGESTED_QUESTIONS = [
  'My printer is not printing.',
  'How do I clean the print head?',
  'Why is my print quality poor?',
  'How do I resolve a paper jam?',
  'What do the flashing lights mean?',
]

interface Props {
  productName: string
  onPick: (q: string) => void
  disabled: boolean
}

export function Welcome({ productName, onPick, disabled }: Props) {
  return (
    <section className="mx-auto max-w-2xl px-2 pt-10 pb-6 text-center sm:pt-16">
      <div className="mx-auto mb-5 flex h-12 w-12 items-center justify-center rounded-2xl bg-brand-50 text-brand-600">
        <Sparkles className="h-6 w-6" aria-hidden />
      </div>
      <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">How can we help with your printer?</h1>
      <p className="mx-auto mt-3 max-w-lg text-[15px] leading-relaxed text-slate-600">
        Ask about your {productName} in your own words. Answers come straight from the official User's Guide, with
        page references you can open.
      </p>
      <div className="mt-8 flex flex-wrap justify-center gap-2">
        {SUGGESTED_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            disabled={disabled}
            onClick={() => onPick(q)}
            className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm text-slate-700 shadow-sm transition hover:-translate-y-px hover:border-brand-500 hover:text-brand-700 disabled:opacity-50"
          >
            {q}
          </button>
        ))}
      </div>
      <p className="mt-8 inline-flex items-center gap-1.5 text-xs text-slate-400">
        <BookOpen className="h-3.5 w-3.5" aria-hidden /> Grounded in the official manual — no guesswork.
      </p>
    </section>
  )
}
