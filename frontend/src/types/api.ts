export type ResponseType = 'answer' | 'clarification' | 'refusal' | 'greeting' | 'error'

export interface Citation {
  id: number
  chunk_id: string
  document_id: string
  document_name: string
  section: string
  heading_path: string[]
  page_label: string | null
  page_label_end: string | null
  pdf_page: number
  pdf_page_end: number
  url: string | null
}

export interface ConversationState {
  current_topic: string
  unresolved_issue: string
  last_intent: string
  last_query: string
  turn_count: number
  recent_questions: string[]
}

export interface ChatResponse {
  session_id: string
  answer: string
  response_type: ResponseType
  citations: Citation[]
  conversation_state: ConversationState
  confidence: string
  standalone_query: string
  model: string | null
  latency_ms: number
}

export interface SessionMessage {
  role: 'user' | 'assistant'
  content: string
  created_at: string
  citations: Citation[]
  response_type: ResponseType | null
}

export interface SessionOut {
  session_id: string
  product_model: string
  conversation_state: ConversationState
  messages: SessionMessage[]
}

export interface DocumentInfo {
  document_id: string
  document_name: string
  manufacturer: string
  product_model: string
  page_count: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  responseType?: ResponseType | null
  failed?: boolean
}
