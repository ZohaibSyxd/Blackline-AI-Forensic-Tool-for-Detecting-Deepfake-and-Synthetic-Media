import React, { useState, useRef } from "react";
import "./UploadCard.css";

const ACCEPT = ".jpg,.jpeg,.png,.gif,.mp4,.avi";

interface AnalysisSummary {
  asset: any;
  validate: any;
  probe: any;
  summary: {
    width?: number;
    height?: number;
    fps?: number;
    duration_s?: number;
    codec?: string;
    format_valid?: boolean;
    decode_valid?: boolean;
    errors?: string[];
    deepfake_likelihood?: number;
    deepfake_label?: string;
    deepfake_method?: string;
  };
}

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8010";

const UploadCard: React.FC = () => {
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisSummary | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  function onFiles(flist: FileList | null) {
    if (!flist || flist.length === 0) return;
    const f = flist[0];
    setFile(f);
    setResult(null);
    setError(null);
    if (f.type.startsWith("image/")) {
      const url = URL.createObjectURL(f);
      setPreview(url);
    } else {
      setPreview(null);
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    onFiles(e.dataTransfer.files);
  };

  const handleBrowse = () => inputRef.current?.click();

  const handleClear = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const resp = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: form,
      });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`Server ${resp.status}: ${txt}`);
      }
      const data = (await resp.json()) as AnalysisSummary;
      setResult(data);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  function pct(v?: number) {
    if (v === undefined || v === null) return '—';
    return (v * 100).toFixed(1) + '%';
  }

  return (
    <div className={`upload-card ${dragOver ? "drag-over" : ""}`}>
      <div
        className="upload-drop"
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <h3>Upload Media for Analysis</h3>
        <p className="muted">Drag & drop image or video here</p>

        {file ? (
          <div className="upload-preview">
            {preview ? <img src={preview} alt={file.name} /> : null}
            <div className="file-meta">
              <div className="file-name">{file.name}</div>
              <div className="file-actions">
                <button className="btn" onClick={handleBrowse} disabled={loading}>Replace</button>
                <button className="btn ghost" onClick={handleClear} disabled={loading}>Remove</button>
                <button className="btn primary" onClick={analyze} disabled={loading}> {loading ? "Analyzing..." : "Analyze"} </button>
              </div>
            </div>
          </div>
        ) : (
          <>
            <button className="btn" onClick={handleBrowse}>Click to browse files</button>
            <div className="accept-text">(Supported formats: JPEG, PNG, GIF, MP4, AVI)</div>
          </>
        )}

        {error && <div className="error-box">{error}</div>}
        {result && (
          <div className="result-box">
            <h4>Analysis Summary</h4>
            <ul>
              <li>Resolution: {result.summary.width}x{result.summary.height}</li>
              <li>FPS: {result.summary.fps ?? '—'}</li>
              <li>Duration: {result.summary.duration_s ? result.summary.duration_s.toFixed(2) + 's' : '—'}</li>
              <li>Codec: {result.summary.codec || '—'}</li>
              <li>Format Valid: {String(result.summary.format_valid)}</li>
              <li>Decode Valid: {String(result.summary.decode_valid)}</li>
              <li>Deepfake Likelihood: {pct(result.summary.deepfake_likelihood)} ({result.summary.deepfake_label || 'n/a'})</li>
            </ul>
            <details>
              <summary>Raw JSON</summary>
              <pre>{JSON.stringify(result, null, 2)}</pre>
            </details>
          </div>
        )}

        <label htmlFor="file-input" className="label-hidden">Upload file</label>
        <input id="file-input" className="upload-hidden" ref={inputRef} type="file" accept={ACCEPT} onChange={(e)=>onFiles(e.target.files as FileList)} />
      </div>
    </div>
  );
};

export default UploadCard;
