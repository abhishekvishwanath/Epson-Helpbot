import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import { preprocess } from '../components/AnswerMarkdown'
import type { Citation } from '../types/api'

const citation: Citation = {
  id: 1, chunk_id: 'c1', document_id: 'epson_l3310_manual', document_name: "Epson L3310 Series User's Guide",
  section: 'Aligning the Print Head', heading_path: ['Maintaining the Printer', 'Aligning the Print Head'],
  page_label: '66', page_label_end: '66', pdf_page: 66, pdf_page_end: 66,
  url: '/api/documents/epson_l3310_manual/pdf#page=66',
}

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }))
}

describe('preprocess', () => {
  it('links only known citation markers and marks pictograms', () => {
    const out = preprocess('Align it [1]. Press [Power button] then [icon] [9].', [citation])
    expect(out).toContain('[1](#cite-1)')
    expect(out).toContain('[Power button](#sym-Power%20button)')
    expect(out).toContain('[icon](#sym-icon)')
    expect(out).toContain('[9].')
  })
})

describe('SupportPage', () => {
  let fetchMock: ReturnType<typeof vi.fn>
  beforeEach(() => {
    localStorage.clear()
    fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (url.endsWith('/documents')) {
        return jsonResponse([{ document_id: 'epson_l3310_manual', document_name: 'Guide', manufacturer: 'Epson', product_model: 'L3310', page_count: 154 }])
      }
      if (url.endsWith('/chat')) {
        const body = JSON.parse(String(init?.body))
        return jsonResponse({
          session_id: 'sess-1', answer: `1. Load paper [1].\n\n**Important:** Do not unplug it.\n\nYou asked: ${body.message}`,
          response_type: 'answer', citations: [citation], confidence: 'high', standalone_query: body.message,
          model: 'm', latency_ms: 1, conversation_state: { current_topic: '', unresolved_issue: '', last_intent: '', last_query: '', turn_count: 1, recent_questions: [] },
        })
      }
      return jsonResponse({}, 404)
    })
    vi.stubGlobal('fetch', fetchMock)
  })
  afterEach(() => vi.unstubAllGlobals())

  it('shows product and suggested questions, answers with clickable page citation', async () => {
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getByTestId('product-name')).toHaveTextContent('Epson L3310 Series'))
    await user.click(screen.getByRole('button', { name: 'How do I clean the print head?' }))
    const answer = await screen.findByTestId('assistant-message')
    expect(answer).toHaveTextContent('Load paper')
    expect(answer).toHaveTextContent('Page 66')
    const links = screen.getAllByRole('link').filter((a) => a.getAttribute('href')?.includes('#page=66'))
    expect(links.length).toBeGreaterThanOrEqual(2)
    expect(localStorage.getItem('printcare.session')).toBe('sess-1')
    // follow-up sends the session id
    await user.type(screen.getByLabelText('Describe your printer problem'), 'and on Mac?{Enter}')
    await waitFor(() => expect(screen.getAllByTestId('assistant-message')).toHaveLength(2))
    const lastBody = JSON.parse(String(fetchMock.mock.calls.filter((c) => String(c[0]).endsWith('/chat')).at(-1)![1].body))
    expect(lastBody.session_id).toBe('sess-1')
  })

  it('prevents duplicate submissions while a reply is pending', async () => {
    let resolve!: (r: Response) => void
    fetchMock.mockImplementation((url: string) =>
      url.endsWith('/chat') ? new Promise<Response>((r) => (resolve = r)) : jsonResponse([]),
    )
    const user = userEvent.setup()
    render(<App />)
    const box = screen.getByLabelText('Describe your printer problem')
    await user.type(box, 'paper jam{Enter}')
    await user.type(box, 'again{Enter}')
    expect(fetchMock.mock.calls.filter((c) => String(c[0]).endsWith('/chat'))).toHaveLength(1)
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled()
    resolve(new Response(JSON.stringify({ error: { code: 'x', message: 'Server hiccup' } }), { status: 500 }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Server hiccup')
  })

  it('reports network failures with a retry option', async () => {
    fetchMock.mockImplementation((url: string) => (url.endsWith('/chat') ? Promise.reject(new TypeError('offline')) : jsonResponse([])))
    const user = userEvent.setup()
    render(<App />)
    await user.type(screen.getByLabelText('Describe your printer problem'), 'help{Enter}')
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent("couldn't reach")
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
  })
})
