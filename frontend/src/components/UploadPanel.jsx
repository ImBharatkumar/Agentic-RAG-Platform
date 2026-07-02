import { useState, useRef } from 'react'

const API_BASE = '/api'

export default function UploadPanel() {
  const [files, setFiles] = useState([])
  const [status, setStatus] = useState(null)   // { type: 'loading'|'success'|'error', text: '' }
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const addFiles = (incoming) => {
    const unique = Array.from(incoming).filter(
      (f) => !files.some((e) => e.name === f.name && e.size === f.size)
    )
    setFiles((prev) => [...prev, ...unique])
    setStatus(null)
  }

  const removeFile = (idx) => setFiles((prev) => prev.filter((_, i) => i !== idx))

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files)
  }

  const handleIngest = async () => {
    if (!files.length) return
    setStatus({ type: 'loading', text: `Ingesting ${files.length} file(s)…` })

    const formData = new FormData()
    files.forEach((f) => formData.append('files', f))

    try {
      const res = await fetch(`${API_BASE}/ingest`, { method: 'POST', body: formData })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()

      const lines = data.results.map(
        (r) =>
          r.status === 'Success'
            ? `✓ ${r.filename} — ${r.chunks} chunks`
            : `✗ ${r.filename}: ${r.message}`
      )
      const allOk = data.results.every((r) => r.status === 'Success')
      setStatus({ type: allOk ? 'success' : 'error', text: lines.join('\n') })
      if (allOk) setFiles([])
    } catch (err) {
      setStatus({ type: 'error', text: `Upload failed: ${err.message}` })
    }
  }

  return (
    <div className="upload-panel">
      <span className="sidebar-section-label">Ingest Documents</span>

      {/* Drop zone */}
      <div
        className={`drop-zone ${dragging ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.txt,.md,.docx"
          onChange={(e) => addFiles(e.target.files)}
          style={{ display: 'none' }}
        />
        <div className="drop-zone-icon">📁</div>
        <p className="drop-zone-text">Drop files or click to browse</p>
        <p className="drop-zone-hint">PDF · TXT · MD · DOCX</p>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="file-list">
          {files.map((f, i) => (
            <div key={i} className="file-item">
              <span style={{ fontSize: 14 }}>📄</span>
              <span className="file-item-name" title={f.name}>{f.name}</span>
              <button className="file-remove" onClick={() => removeFile(i)}>×</button>
            </div>
          ))}
        </div>
      )}

      {/* Status */}
      {status && (
        <div className={`ingest-status ${status.type}`} style={{ whiteSpace: 'pre-line' }}>
          {status.text}
        </div>
      )}

      {/* Ingest button */}
      <button
        className="btn-primary"
        onClick={handleIngest}
        disabled={!files.length || status?.type === 'loading'}
      >
        {status?.type === 'loading' ? 'Processing…' : `Ingest${files.length ? ` (${files.length})` : ''}`}
      </button>
    </div>
  )
}
