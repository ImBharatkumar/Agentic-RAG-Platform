export default function MessageBubble({ role, content, timestamp }) {
  const isUser = role === 'user'
  const timeStr = timestamp
    ? new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : ''

  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      <div className={`avatar ${isUser ? 'user' : 'assistant'}`}>
        {isUser ? 'U' : '✦'}
      </div>
      <div>
        <div className={`bubble ${isUser ? 'user' : 'assistant'}`}>
          {content}
        </div>
        {timeStr && <div className="bubble-meta">{timeStr}</div>}
      </div>
    </div>
  )
}
