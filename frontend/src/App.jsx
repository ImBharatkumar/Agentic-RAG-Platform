import { useState, useCallback, useEffect } from 'react'
import ChatWindow from './components/ChatWindow.jsx'
import InputBar from './components/InputBar.jsx'
import UploadPanel from './components/UploadPanel.jsx'

const API_BASE = '/api'

// ── Session ID: persists for the browser tab, clears on tab close ─────────────
function getOrCreateThreadId() {
  let id = sessionStorage.getItem('rag_thread_id')
  if (!id) {
    id = crypto.randomUUID()
    sessionStorage.setItem('rag_thread_id', id)
  }
  return id
}

export default function App() {
  const [threadId, setThreadId] = useState(getOrCreateThreadId)
  const [messages, setMessages] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingText, setStreamingText] = useState('')

  // ── Load history for the current session on mount / session change ──────────
  useEffect(() => {
    async function loadHistory() {
      try {
        const res = await fetch(`${API_BASE}/sessions/${threadId}/history`)
        if (!res.ok) return
        const data = await res.json()
        if (data.messages?.length) {
          setMessages(
            data.messages.map((m) => ({
              role: m.role,
              content: m.content,
              timestamp: null,
            }))
          )
        }
      } catch {
        // Silently ignore — first session will have no history
      }
    }
    loadHistory()
  }, [threadId])

  // ── New session: generate fresh UUID and clear messages ────────────────────
  const handleNewSession = useCallback(() => {
    const id = crypto.randomUUID()
    sessionStorage.setItem('rag_thread_id', id)
    setThreadId(id)
    setMessages([])
    setStreamingText('')
  }, [])

  // ── Send a query to the backend and consume the streaming response ──────────
  const handleSend = useCallback(
    async (query) => {
      if (isStreaming) return

      // Optimistically add user message
      const userMsg = { role: 'user', content: query, timestamp: Date.now() }
      setMessages((prev) => [...prev, userMsg])
      setIsStreaming(true)
      setStreamingText('')

      try {
        const res = await fetch(`${API_BASE}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: query, thread_id: threadId }),
        })

        if (!res.ok) {
          throw new Error(`Server error: ${res.status}`)
        }

        // Consume the plain-text stream
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let accumulated = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          accumulated += decoder.decode(value, { stream: true })
          setStreamingText(accumulated)
        }

        // Commit streamed text as a real message
        const aiMsg = {
          role: 'assistant',
          content: accumulated || '(No response received)',
          timestamp: Date.now(),
        }
        setMessages((prev) => [...prev, aiMsg])
      } catch (err) {
        const errMsg = {
          role: 'assistant',
          content: `⚠️ Error: ${err.message}`,
          timestamp: Date.now(),
        }
        setMessages((prev) => [...prev, errMsg])
      } finally {
        setIsStreaming(false)
        setStreamingText('')
      }
    },
    [isStreaming, threadId]
  )

  // Short display of thread ID for the sidebar badge
  const shortId = threadId.slice(0, 8) + '…'

  return (
    <div className="app-shell">
      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <aside className="sidebar">
        {/* Brand */}
        <div className="sidebar-brand">
          <div className="brand-icon">✦</div>
          <div className="brand-text">
            <h1>Enterprise RAG</h1>
            <p>Document Intelligence</p>
          </div>
        </div>

        {/* Session info */}
        <div>
          <span className="sidebar-section-label">Current Session</span>
          <div className="session-badge" style={{ marginTop: 8 }}>
            <span className="label">Thread ID</span>
            <span className="value">{shortId}</span>
          </div>
        </div>

        <div className="divider" />

        {/* Upload panel */}
        <UploadPanel />

        {/* Spacer to push version to bottom */}
        <div style={{ flex: 1 }} />

        <div style={{ fontSize: 10, color: 'var(--text-muted)', padding: '0 4px' }}>
          Enterprise RAG Platform v0.2 · Episodic Memory
        </div>
      </aside>

      {/* ── Main chat area ────────────────────────────────────────────────── */}
      <main className="chat-area">
        {/* Header */}
        <div className="chat-header">
          <div>
            <div className="chat-header-title">Chat</div>
            <div className="chat-header-sub">
              {messages.length > 0
                ? `${Math.ceil(messages.length / 2)} exchange${Math.ceil(messages.length / 2) !== 1 ? 's' : ''} · Session ${shortId}`
                : 'Start a new conversation'}
            </div>
          </div>
          <button className="new-session-btn" onClick={handleNewSession}>
            + New Session
          </button>
        </div>

        {/* Messages */}
        <ChatWindow
          messages={messages}
          isStreaming={isStreaming}
          streamingText={streamingText}
        />

        {/* Input */}
        <InputBar onSend={handleSend} disabled={isStreaming} />
      </main>
    </div>
  )
}
