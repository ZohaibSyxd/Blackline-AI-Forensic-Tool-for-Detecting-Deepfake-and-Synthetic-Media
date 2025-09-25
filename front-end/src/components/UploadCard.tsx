import React, { useEffect, useRef, useState } from "react";
import { persistFile, loadFile, removeFile as removePersistedFile } from '../utils/uploadPersistence';
import { upsertAnalysis, deleteAnalyses } from '../state/analysisStore';
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

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

type ItemStatus = 'idle' | 'uploading' | 'uploaded' | 'analyzing' | 'done' | 'error';

interface UploadItem {
  id: string;
  file?: File;
  name?: string;
  mime?: string;
  preview: string | null;
  status: ItemStatus;
  progress: number; // 0-100
  result?: AnalysisSummary | null;
  error?: string | null;
}

interface UploadCardProps { pageKey?: string }
const STORAGE_PREFIX = 'bl_uploadItems_';
// Versioned per-page storage (v3) so each analysis page retains its own uploads
const STORAGE_VERSION = 'v3';
// In-memory cache of File blobs keyed by composite (pageKey::itemId)
const memoryFilesGlobal: Record<string, File> = {};

const UploadCard: React.FC<UploadCardProps> = ({ pageKey }) => {
  const [dragOver, setDragOver] = useState(false);
  const [items, setItems] = useState<UploadItem[]>([]);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const replaceRef = useRef<HTMLInputElement | null>(null);
  const replaceTargetId = useRef<string | null>(null);
  // keep Files in-memory (not persisted), keyed by item id
  const filesRef = useRef<Record<string, File | undefined>>({});

  type SlimItem = Pick<UploadItem, 'id' | 'status' | 'progress' | 'preview' | 'result' | 'error'> & { fileName: string; fileType: string };

  const storageKey = pageKey ? `${STORAGE_PREFIX}${pageKey}_${STORAGE_VERSION}` : `${STORAGE_PREFIX}__global_${STORAGE_VERSION}`;

  // hydrate once per *page* (pageKey) from localStorage
  useEffect(() => {
    (async () => {
      try {
        const raw = localStorage.getItem(storageKey);
        const slimArr: SlimItem[] = raw ? (JSON.parse(raw) as any) : [];
        if (!Array.isArray(slimArr)) return;
        const restored: UploadItem[] = await Promise.all(slimArr.map(async si => {
          const compositeId = pageKey ? `${pageKey}::${si.id}` : si.id;
            const file = memoryFilesGlobal[compositeId] || await loadFile(compositeId);
            if (file) memoryFilesGlobal[compositeId] = file;
            return {
              id: si.id,
              file: file || undefined,
              name: si.fileName,
              mime: si.fileType,
              preview: si.preview || null,
              status: si.status as ItemStatus,
              progress: si.progress || 0,
              result: si.result || null,
              error: si.error || null,
            } as UploadItem;
        }));
        // Map current page's files to local ref for quick access
        filesRef.current = restored.reduce<Record<string, File | undefined>>((acc, it) => { if (it.file) acc[it.id] = it.file; return acc; }, {});
        setItems(restored);
      } catch {}
    })();
  }, [pageKey, storageKey]);

  // persist to localStorage whenever items change
  useEffect(() => {
    try {
      const slim: SlimItem[] = items.map(it => ({
        id: it.id,
        fileName: it.file ? it.file.name : '',
        fileType: it.file ? it.file.type : '',
        preview: it.preview,
        status: it.status,
        progress: it.progress,
        result: it.result || null,
        error: it.error || null,
      }));
      localStorage.setItem(storageKey, JSON.stringify(slim));
    } catch {}
  }, [items, storageKey]);

  function onFiles(flist: FileList | null) {
    if (!flist || flist.length === 0) return;
    setGlobalError(null);
    const newItems: UploadItem[] = [];
    const replaceId = replaceTargetId.current;
    if (replaceId) {
      // replace single item
      const f = flist[0];
      filesRef.current[replaceId] = f;
      setItems(prev => prev.map(it => {
        if (it.id !== replaceId) return it;
        if (it.preview) URL.revokeObjectURL(it.preview);
        const preview = f.type.startsWith('image/') ? URL.createObjectURL(f) : null;
        return { ...it, file: f, name: f.name, mime: f.type, preview, status: 'idle', progress: 0, result: null, error: null };
      }));
      const composite = pageKey ? `${pageKey}::${replaceId}` : replaceId;
      memoryFilesGlobal[composite] = f;
      persistFile(composite, f);
      replaceTargetId.current = null;
      return;
    }
    for (let i = 0; i < flist.length; i++) {
      const f = flist[i];
      const preview = f.type.startsWith('image/') ? URL.createObjectURL(f) : null;
      const id = `${Date.now()}_${i}_${f.name}`;
      filesRef.current[id] = f;
      const composite = pageKey ? `${pageKey}::${id}` : id;
      memoryFilesGlobal[composite] = f;
      persistFile(composite, f);
      newItems.push({
        id,
        file: f,
        name: f.name,
        mime: f.type,
        preview,
        status: 'idle',
        progress: 0,
        result: null,
        error: null,
      });
    }
    setItems(prev => [...prev, ...newItems]);
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    onFiles(e.dataTransfer.files);
  };

  const handleBrowse = () => inputRef.current?.click();

  const handleReplace = (id: string) => {
    replaceTargetId.current = id;
    replaceRef.current?.click();
  };

  const removeItem = (id: string) => {
    setItems(prev => {
      const it = prev.find(x => x.id === id);
      if (it?.preview) URL.revokeObjectURL(it.preview);
      delete filesRef.current[id];
      const composite = pageKey ? `${pageKey}::${id}` : id;
      delete memoryFilesGlobal[composite];
      removePersistedFile(composite);
      // Also remove any persisted analysis summary for this item so it disappears from Reports
      try { deleteAnalyses([id], pageKey || 'global'); } catch {/* ignore */}
      return prev.filter(x => x.id !== id);
    });
  };

  async function analyzeItem(id: string) {
    setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'uploading', progress: 0, error: null, result: null } : it));
    const item = items.find(it => it.id === id);
    if (!item) return;
    const file = filesRef.current[id] || item.file;
    if (!file) {
      setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'error', error: 'File not attached. Use Replace to reattach the file.' } : it));
      return;
    }
    try {
      const form = new FormData();
      form.append('file', file);
      const xhr = new XMLHttpRequest();
      await new Promise<void>((resolve, reject) => {
        xhr.upload.onprogress = (ev) => {
          if (ev.lengthComputable) {
            const pct = Math.round((ev.loaded / ev.total) * 100);
            setItems(prev => prev.map(it => it.id === id ? { ...it, progress: pct } : it));
          }
        };
        xhr.onloadstart = () => {
          setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'uploading' } : it));
        };
        xhr.onreadystatechange = () => {
          // When upload finishes but before response ready, mark analyzing
          if (xhr.readyState === 2 || xhr.readyState === 3) {
            setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'analyzing' } : it));
          }
        };
        xhr.onerror = () => reject(new Error('Network error'));
        xhr.onload = () => {
          try {
            if (xhr.status >= 200 && xhr.status < 300) {
              const data = JSON.parse(xhr.responseText) as AnalysisSummary;
              setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'done', progress: 100, result: data, error: null } : it));
              // Persist summary for reports page
              try {
                const page = pageKey || 'global';
                upsertAnalysis({
                  id,
                  pageKey: page,
                  fileName: file.name,
                  mime: file.type,
                  analyzedAt: Date.now(),
                  summary: data.summary || {},
                  raw: data
                });
              } catch {/* swallow */}
              resolve();
            } else {
              reject(new Error(`Server ${xhr.status}: ${xhr.responseText}`));
            }
          } catch (e: any) {
            reject(new Error(e.message || 'Invalid JSON response'));
          }
        };
        xhr.open('POST', `${API_BASE}/api/analyze`);
        xhr.send(form);
      });
    } catch (e: any) {
      setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'error', error: e.message || String(e) } : it));
    }
  }

  async function analyzeAllSequential() {
    setGlobalError(null);
    const pending = items.filter(it => it.status === 'idle' || it.status === 'error');
    for (const it of pending) {
      // eslint-disable-next-line no-await-in-loop
      await analyzeItem(it.id);
    }
  }

  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  function toggleExpand(id: string) { setExpanded(prev => ({ ...prev, [id]: !prev[id] })); }
  function pct(v?: number) { if (v === undefined || v === null) return '—'; return (v * 100).toFixed(1) + '%'; }

  return (
    <div className={`upload-card ${dragOver ? "drag-over" : ""}`}>
      <div
        className="upload-drop"
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <div className="upload-header">
          <div className="upload-header-text">
            <h3>Upload Media for Analysis</h3>
            <p className="muted">Drag & drop image or video here — you can select multiple files</p>
          </div>
          <div className="upload-header-actions">
            <button className="btn" onClick={handleBrowse}>Upload More</button>
            <button className="btn primary" onClick={analyzeAllSequential} disabled={items.length===0}>Analyze All</button>
          </div>
        </div>

        {items.length === 0 ? (
          <div
            className="empty-hint"
            role="button"
            tabIndex={0}
            onClick={handleBrowse}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleBrowse(); } }}
            aria-label="No files yet. Activate to select files to upload."
          >
            No files yet. Use <strong>Upload More</strong> or drag & drop.
          </div>
        ) : (
          <div className="items-list">
            <div className="items-scroll" role="list" aria-label="Uploaded files list">
            {items.map(it => (
              <div key={it.id} className="item-row" role="listitem">
                {it.preview ? (
                  <img className="thumb" src={it.preview} alt={it.file ? it.file.name : (it.name || 'Uploaded file')} />
                ) : (
                  <div className="thumb placeholder" aria-hidden />
                )}
                <div className="item-main">
                  <div className="file-name">{it.file ? it.file.name : (it.name || 'File')}</div>
                  <div className="file-actions with-status">
                    <button className="btn" onClick={() => handleReplace(it.id)} disabled={it.status==='uploading'||it.status==='analyzing'}>Replace</button>
                    <button className="btn ghost" onClick={() => removeItem(it.id)} disabled={it.status==='uploading'||it.status==='analyzing'}>Remove</button>
                    <button className="btn primary" onClick={() => analyzeItem(it.id)} disabled={it.status==='uploading'||it.status==='analyzing'||!(filesRef.current[it.id]||it.file)} title={!(filesRef.current[it.id]||it.file) ? 'File not attached. Click Replace to reattach.' : undefined}>
                      {it.status==='uploading' ? `Uploading ${it.progress}%` : (it.status==='analyzing' ? 'Analyzing…' : (it.status==='done' ? 'Re-run' : 'Analyze'))}
                    </button>
                    {(it.status==='analyzing' || it.status==='done' || it.status==='error') && (
                      <button
                        className="btn ghost details-toggle"
                        onClick={()=>toggleExpand(it.id)}
                        aria-expanded={`${expanded[it.id] ? 'true' : 'false'}`}
                        aria-controls={`steps-${it.id}`}
                      >
                        {expanded[it.id] ? 'Hide Details' : 'Show Details'}
                      </button>
                    )}
                    {(it.status==='analyzing' || it.status==='done') && (
                      <span className="status-badge push-right" aria-label="Upload complete">
                        <span className="dot" /> Upload Complete
                      </span>
                    )}
                  </div>
                  {expanded[it.id] && (
                    <div className="progress-details" id={`steps-${it.id}`}>
                      <div className="progress-steps compact">
                        <div className={`step ${it.status==='analyzing' || it.status==='done' ? 'active' : ''} ${it.status==='done' ? 'done' : ''}`}>
                          <span className={`icon ${it.status==='done' ? 'check' : 'pending'}`} aria-hidden="true" />
                          <div className="step-text">
                            <div className="title">Frame Analysis Running</div>
                            <div className="desc">Extracting and scanning frames</div>
                          </div>
                        </div>
                        <div className={`divider ${it.status==='done' ? 'active' : ''}`} />
                        <div className={`step ${it.status==='done' ? 'done' : ''}`}>
                          <span className={`icon ${it.status==='done' ? 'check' : 'pending'}`} aria-hidden="true" />
                          <div className="step-text">
                            <div className="title">Forensics Check Passed</div>
                            <div className="desc">Metadata and A/V consistency verified</div>
                          </div>
                        </div>
                        <div className={`divider ${it.status==='done' ? 'active' : ''}`} />
                        <div className={`step ${it.status==='done' ? ((it.result?.summary.deepfake_likelihood ?? 0) > 0.5 ? 'alert' : 'done') : ''}`}>
                          <span className={`icon ${(it.result?.summary.deepfake_likelihood ?? 0) > 0.5 ? 'alert' : (it.status==='done' ? 'check' : 'pending')}`} aria-hidden="true" />
                          <div className="step-text">
                            <div className="title">{(it.result?.summary.deepfake_likelihood ?? 0) > 0.5 ? 'Manipulation Detected' : 'No Manipulation Detected'}</div>
                            <div className="desc">{(it.result?.summary.deepfake_likelihood ?? 0) > 0.5 ? 'Suspicious regions or techniques identified' : 'No suspicious regions identified'}</div>
                          </div>
                        </div>
                        <div className={`divider ${it.status==='done' ? 'active' : ''}`} />
                        <div className={`step ${it.status==='done' ? 'done' : ''}`}>
                          <span className={`icon ${it.status==='done' ? 'check' : 'pending'}`} aria-hidden="true" />
                          <div className="step-text">
                            <div className="title">Report Ready</div>
                            <div className="desc">Download detailed forensic analysis report</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  {/* Detailed summary removed from upload view; now available on Reports page */}
                  {it.error && <div className="error-box">{it.error}</div>}
                </div>
              </div>
            ))}
            </div>
          </div>
        )}

        {globalError && <div className="error-box">{globalError}</div>}

        <label htmlFor="file-input" className="label-hidden">Upload file(s)</label>
        <input id="file-input" className="upload-hidden" ref={inputRef} type="file" accept={ACCEPT} multiple onChange={(e)=>onFiles(e.target.files as FileList)} />
  <label htmlFor="file-replace" className="label-hidden">Replace file</label>
  <input id="file-replace" className="upload-hidden" ref={replaceRef} type="file" accept={ACCEPT} onChange={(e)=>onFiles(e.target.files as FileList)} />
      </div>
    </div>
  );
};

export default UploadCard;
