import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble.jsx'

export default function ChatWindow({ messages, isStreaming, streamingText }) {
  const bottomRef = useRef(null)

  // Auto-scroll to bottom when messages or streaming text changes
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingText])

  if (messages.length === 0 && !isStreaming) {
    return (
      <div className="message-list">
        <div className="empty-state">
          <div className="empty-icon">🧠</div>
          <p className="empty-title">Enterprise RAG Platform</p>
          <p className="empty-subtitle">
            Ask any question about your documents. The system remembers your conversation
            within this session — follow-up questions just work.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="message-list">
      {messages.map((msg, i) => (
        <MessageBubble key={i} role={msg.role} content={msg.content} timestamp={msg.timestamp} />
      ))}

      {/* Live streaming bubble */}
      {isStreaming && (
        <div className="message-row assistant">
          <div className="avatar assistant">✦</div>
          <div>
            {streamingText ? (
              <div className="bubble assistant">
                {streamingText}
                <span className="cursor" />
              </div>
            ) : (
              <div className="bubble assistant">
                <div className="typing-indicator">
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
