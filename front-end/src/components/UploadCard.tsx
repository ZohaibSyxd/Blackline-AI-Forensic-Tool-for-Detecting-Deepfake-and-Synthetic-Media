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
    // Copy-move specific fields (optional)
    cm_confidence?: number;
    cm_coverage_ratio?: number;
    cm_shift_magnitude?: number;
    cm_num_keypoints?: number;
    cm_num_matches?: number;
    overlay_uri?: string;
    // LBP specific (optional)
    lbp_frames?: number;
    lbp_dim?: number;
  };
}

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8010";

type ItemStatus = 'idle' | 'uploading' | 'uploaded' | 'analyzing' | 'done' | 'error' | 'canceled';

interface UploadItem {
  id: string;
  file?: File;
  name?: string;
  mime?: string;
  preview: string | null;
  status: ItemStatus;
  progress: number; // 0-100
  jobId?: string;
  xhr?: XMLHttpRequest; // in-flight request for cancel
  analyzeStartAt?: number; // persisted start time for timer
  stage?: string; // backend-reported stage key
  stageMsg?: string | null; // backend-reported human message
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
  const [model, setModel] = useState<string>('stub');
  const inputRef = useRef<HTMLInputElement | null>(null);
  const replaceRef = useRef<HTMLInputElement | null>(null);
  const replaceTargetId = useRef<string | null>(null);
  // keep Files in-memory (not persisted), keyed by item id
  const filesRef = useRef<Record<string, File | undefined>>({});
  // per-item poll stop functions (to stop progress polling on cancel)
  const pollStopsRef = useRef<Record<string, () => void>>({});
  // tick to force re-render for elapsed timer
  const [tick, setTick] = useState(0);
  const [undoBuf, setUndoBuf] = useState<{ item: UploadItem; file?: File; index: number } | null>(null);
  const undoTimerRef = useRef<number | null>(null);
  const [doneToast, setDoneToast] = useState<{ id: string; name: string } | null>(null);

  useEffect(() => {
    return () => {
      if (undoTimerRef.current) {
        window.clearTimeout(undoTimerRef.current);
        undoTimerRef.current = null;
      }
    };
  }, []);

  // timer interval for elapsed time while analyzing
  useEffect(() => {
    const iv = window.setInterval(() => setTick(t => t + 1), 500);
    return () => window.clearInterval(iv);
  }, []);

  // remember last used model
  useEffect(() => {
    try {
      const saved = localStorage.getItem('bl_lastModel');
      if (saved) setModel(saved);
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    try { localStorage.setItem('bl_lastModel', model); } catch {}
  }, [model]);

  // stop any active polls on unmount to avoid background loops
  useEffect(() => {
    return () => {
      try {
        const stops = Object.values(pollStopsRef.current || {});
        stops.forEach((fn) => { try { fn(); } catch {} });
        pollStopsRef.current = {};
      } catch {}
    };
  }, []);

  // helper: start progress polling for a job id if not already polling
  function startProgressPolling(id: string, jobId: string) {
    if (!jobId) return;
    if (pollStopsRef.current[id]) return; // already polling
    let stopped = false;
    const stop = () => { stopped = true; };
    pollStopsRef.current[id] = stop;
    const poll = async () => {
      while (!stopped) {
        try {
          const resp = await fetch(`${API_BASE}/api/progress/${jobId}`, { cache: 'no-cache' });
          if (resp.ok) {
            const j = await resp.json();
            const p = Math.max(0, Math.min(100, typeof j.percent === 'number' ? j.percent : 0));
            const stage = j.stage || '';
            const message = typeof j.message === 'string' ? j.message : '';
            setItems(prev => prev.map(it => it.id === id ? { ...it, status: (it.status==='done'||it.status==='error'||it.status==='canceled') ? it.status : (stage==='done' ? 'done' : 'analyzing'), progress: Math.max(it.progress, Math.round(p)), stage, stageMsg: message || it.stageMsg, analyzeStartAt: it.analyzeStartAt || (stage && stage !== 'done' ? Date.now() : it.analyzeStartAt) } : it));
            if (stage === 'done') { stop(); try { delete pollStopsRef.current[id]; } catch {}; break; }
          }
        } catch {}
        await new Promise(r => setTimeout(r, 800));
      }
    };
    poll();
    // safety timeout: stop after 10 minutes
    setTimeout(stop, 10*60*1000);
  }

  // When items are restored or tab is re-opened, reattach polling for in-progress jobs
  useEffect(() => {
    items.forEach(it => {
      if ((it.status === 'analyzing' || it.status === 'uploading') && it.jobId) {
        startProgressPolling(it.id, it.jobId);
      }
    });
  }, [items]);

  // Cross-page sync: when Reports deletes analyses, remove corresponding uploads here
  useEffect(() => {
    function onRemoved(ev: Event) {
      try {
        const ce = ev as CustomEvent<{ pageKey: string; ids: string[] }>;
        const detail = ce.detail;
        if (!detail) return;
        const current = pageKey || 'global';
        if (detail.pageKey !== current) return;
        const idSet = new Set(detail.ids);
        setItems(prev => {
          // revoke previews and drop from memory for removed items
          prev.forEach(it => { if (idSet.has(it.id) && it.preview) URL.revokeObjectURL(it.preview); });
          detail.ids.forEach(id => { delete filesRef.current[id]; });
          return prev.filter(it => !idSet.has(it.id));
        });
      } catch {}
    }
    window.addEventListener('bl:uploads-removed', onRemoved as EventListener);
    return () => { window.removeEventListener('bl:uploads-removed', onRemoved as EventListener); };
  }, [pageKey]);

  type SlimItem = Pick<UploadItem, 'id' | 'status' | 'progress' | 'preview' | 'result' | 'error' | 'stage' | 'stageMsg' | 'analyzeStartAt' | 'jobId'> & { fileName: string; fileType: string };

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
              stage: (si as any).stage,
              stageMsg: (si as any).stageMsg || null,
              result: si.result || null,
              analyzeStartAt: (si as any).analyzeStartAt || undefined,
              jobId: (si as any).jobId || undefined,
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
        stage: it.stage,
        stageMsg: it.stageMsg || null,
        analyzeStartAt: it.analyzeStartAt || undefined,
        jobId: it.jobId || undefined,
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

  // helper: navigate app-wide via custom event
  function navTo(page: string) {
    try { window.dispatchEvent(new CustomEvent('bl:navigate', { detail: { page } })); } catch {}
  }

  const handleReplace = (id: string) => {
    replaceTargetId.current = id;
    replaceRef.current?.click();
  };

  const removeItem = (id: string) => {
    // capture current state for undo
    const idx = items.findIndex(x => x.id === id);
    const it = idx >= 0 ? items[idx] : undefined;
    const file = it ? (filesRef.current[id] || it.file) : undefined;
    if (!it) return;
    if (undoTimerRef.current) { window.clearTimeout(undoTimerRef.current); undoTimerRef.current = null; }
    setUndoBuf({ item: it, file, index: idx });
    // proceed with removal
    setItems(prev => {
      if (it.preview) URL.revokeObjectURL(it.preview);
      delete filesRef.current[id];
      const composite = pageKey ? `${pageKey}::${id}` : id;
      delete memoryFilesGlobal[composite];
      removePersistedFile(composite);
      try { deleteAnalyses([id], pageKey || 'global'); } catch {/* ignore */}
      return prev.filter(x => x.id !== id);
    });
    // start undo window (7s)
    undoTimerRef.current = window.setTimeout(() => { setUndoBuf(null); undoTimerRef.current = null; }, 7000);
  };

  function undoRemove() {
    if (!undoBuf) return;
    const { item, file, index } = undoBuf;
    // restore file refs and persistence
    if (file) {
      filesRef.current[item.id] = file;
      const composite = pageKey ? `${pageKey}::${item.id}` : item.id;
      memoryFilesGlobal[composite] = file;
      try { persistFile(composite, file); } catch { /* ignore */ }
    }
    // regenerate preview if image
    const preview = file && file.type.startsWith('image/') ? URL.createObjectURL(file) : null;
    const restored: UploadItem = { ...item, file: file || item.file, preview, status: item.status };
    setItems(prev => {
      const arr = [...prev];
      const insertAt = Math.min(Math.max(index, 0), arr.length);
      arr.splice(insertAt, 0, restored);
      return arr;
    });
    // re-add analysis summary to reports if exists
    try {
      if (item.result?.summary) {
        const page = pageKey || 'global';
        const fileName = file?.name || item.name || item.result?.asset?.fileName || 'File';
        const mime = file?.type || item.mime;
        upsertAnalysis({ id: item.id, pageKey: page, fileName, mime, analyzedAt: Date.now(), summary: item.result.summary, raw: item.result });
      }
    } catch { /* ignore */ }
    setUndoBuf(null);
    if (undoTimerRef.current) { window.clearTimeout(undoTimerRef.current); undoTimerRef.current = null; }
  }

  async function analyzeItem(id: string) {
  setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'uploading', progress: 0, stage: 'uploading', stageMsg: 'Uploading file', error: null, result: null } : it));
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
  form.append('model', model);
  const jobId = `${id}-${Math.random().toString(36).slice(2,8)}`;
  form.append('job_id', jobId);
      const xhr = new XMLHttpRequest();
      xhr.timeout = 10 * 60 * 1000; // 10 minutes
      // store xhr and jobId so we can cancel later and resume timer across tabs
      setItems(prev => prev.map(it => it.id === id ? { ...it, xhr, jobId } : it));
      // Start polling server-side progress immediately so UI reflects server stages
      startProgressPolling(id, jobId);
      await new Promise<void>((resolve, reject) => {
        xhr.upload.onprogress = (ev) => {
          if (ev.lengthComputable) {
            const pct = Math.round((ev.loaded / ev.total) * 100);
            setItems(prev => prev.map(it => it.id === id ? { ...it, progress: pct } : it));
            if (pct >= 100) {
              // Switch to analyzing as soon as upload finishes from client side
              setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'analyzing', stage: 'processing', stageMsg: 'Processing on server', analyzeStartAt: it.analyzeStartAt || Date.now() } : it));
            }
          }
        };
        xhr.upload.onload = () => {
          // Upload complete; move to analyzing while waiting for server processing/headers
          setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'analyzing', progress: 100, stage: 'processing', stageMsg: 'Processing on server', analyzeStartAt: it.analyzeStartAt || Date.now() } : it));
        };
        xhr.onloadstart = () => {
          setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'uploading' } : it));
        };
        xhr.onreadystatechange = () => {
          // When upload finishes but before response ready, mark analyzing
          if (xhr.readyState === 2 || xhr.readyState === 3) {
            setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'analyzing', stage: 'processing', stageMsg: 'Processing on server', analyzeStartAt: it.analyzeStartAt || Date.now() } : it));
          }
        };
        xhr.ontimeout = () => {
          reject(new Error('Request timed out. The server is taking too long to process the file.'));
        };
        xhr.onerror = () => reject(new Error('Network error'));
        xhr.onabort = () => reject(new Error('canceled'));
        xhr.onload = () => {
          try {
            if (xhr.status >= 200 && xhr.status < 300) {
              const data = JSON.parse(xhr.responseText) as AnalysisSummary;
              setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'done', progress: 100, stage: 'done', stageMsg: 'Completed', result: data, error: null, xhr: undefined } : it));
              try { pollStopsRef.current[id]?.(); delete pollStopsRef.current[id]; } catch {}
              try { setDoneToast({ id, name: file.name }); setTimeout(() => setDoneToast(null), 6000); } catch {}
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
  // Ensure we stop polling when analysis completes or errors via the helper's timeout
    } catch (e: any) {
      // stop polling and clear xhr on error
      try { pollStopsRef.current[id]?.(); delete pollStopsRef.current[id]; } catch {}
      setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'error', error: e.message || String(e), xhr: undefined } : it));
    }
  }

  function cancelItem(id: string) {
    // abort in-flight request and stop polling
    try {
      const it = items.find(x => x.id === id);
      if (it?.xhr) {
        try { it.xhr.abort(); } catch {}
      }
    } catch {}
    try { pollStopsRef.current[id]?.(); delete pollStopsRef.current[id]; } catch {}
    setItems(prev => prev.map(it => it.id === id ? { ...it, status: 'canceled', stage: 'canceled', stageMsg: 'Canceled by user', xhr: undefined, analyzeStartAt: undefined } : it));
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
  // Derive which items can show details (analyzing/done/error)
  const detailEligibleIds = items.filter(it => (it.status === 'analyzing' || it.status === 'done' || it.status === 'error')).map(it => it.id);
  const allEligibleExpanded = detailEligibleIds.length > 0 && detailEligibleIds.every(id => !!expanded[id]);
  const toggleAllDetails = () => {
    setExpanded(prev => {
      const next: Record<string, boolean> = { ...prev };
      const target = !allEligibleExpanded; // if all are expanded, collapse them; otherwise expand
      detailEligibleIds.forEach(id => { next[id] = target; });
      return next;
    });
  };
  function pct(v?: number) { if (v === undefined || v === null) return '—'; return (v * 100).toFixed(1) + '%'; }
  function pctInt(v?: number) { if (v === undefined || v === null) return 0; return Math.min(100, Math.max(0, Math.round(v))); }
  function fmtElapsed(ms?: number) {
    if (!ms || ms < 0) return '0:00';
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, '0')}`;
  }

  return (
    <>
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
            <select
              aria-label="Select analysis model"
              className="btn"
              value={model}
              onChange={(e)=>setModel(e.target.value)}
              title="Choose which technique to run"
            >
              <option value="stub">Hash Stub (baseline)</option>
              <option value="copy_move">Copy-Move (ORB)</option>
              <option value="lbp">LBP (trained)</option>
            </select>
            <button className="btn" onClick={handleBrowse}>Upload More</button>
            <button
              className="btn ghost"
              onClick={toggleAllDetails}
              disabled={detailEligibleIds.length === 0}
              title={allEligibleExpanded ? 'Hide details for all items' : 'Show details for all items'}
            >
              {allEligibleExpanded ? 'Hide All Details' : 'Show All Details'}
            </button>
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
                  {(it.status==='uploading' || it.status==='analyzing') && (
                    <div className={`status-line progress-p${Math.round((pctInt(it.progress))/5)*5}`} aria-live="polite">
                      <div
                        className="bar"
                        role="progressbar"
                        aria-label="Analysis progress"
                      >
                        <div className="bar-fill" />
                      </div>
                      <div className="status-text">
                        <span className="muted">{it.status==='uploading' ? 'Uploading' : 'Analyzing'}</span>
                        <span className="sep">·</span>
                        <span className="muted">{pctInt(it.progress)}%</span>
                        {it.stageMsg ? (
                          <>
                            <span className="sep">·</span>
                            <span className="muted">{it.stageMsg}</span>
                          </>
                        ) : null}
                        {it.analyzeStartAt ? (
                          <>
                            <span className="sep">·</span>
                            <span className="muted">{fmtElapsed(Date.now() - it.analyzeStartAt)}</span>
                          </>
                        ) : null}
                      </div>
                    </div>
                  )}
                  <div className="file-actions with-status">
                    <button className="btn" onClick={() => handleReplace(it.id)} disabled={it.status==='uploading'||it.status==='analyzing'}>Replace</button>
                    <button className="btn ghost" onClick={() => removeItem(it.id)} disabled={it.status==='uploading'||it.status==='analyzing'}>Remove</button>
                    {it.status==='done' ? (
                      <>
                        <button className="btn" onClick={() => analyzeItem(it.id)}>
                          Re-run
                        </button>
                        <button className="btn primary" onClick={() => navTo('reports')}>
                          View Report
                        </button>
                      </>
                    ) : (
                      <button className="btn primary" onClick={() => analyzeItem(it.id)} disabled={it.status==='uploading'||it.status==='analyzing'||!(filesRef.current[it.id]||it.file)} title={!(filesRef.current[it.id]||it.file) ? 'File not attached. Click Replace to reattach.' : undefined}>
                        {it.status==='uploading' ? `Uploading ${it.progress}%` : (it.status==='analyzing' ? 'Analyzing…' : 'Analyze')}
                      </button>
                    )}
                    {(it.status==='uploading' || it.status==='analyzing') && (
                      <button className="btn ghost" onClick={() => cancelItem(it.id)} title="Cancel this analysis">Cancel</button>
                    )}
                    {(it.status==='analyzing' || it.status==='done' || it.status==='error') && (
                      <button
                        className="btn ghost details-toggle"
                        onClick={()=>toggleExpand(it.id)}
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
                      {/* Summary box removed per request */}
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
    {undoBuf ? (
      <div className="toast" role="status" aria-live="polite">
        <span>Removed {undoBuf.item.file?.name || undoBuf.item.name || 'file'}.</span>
        <button className="btn link" onClick={undoRemove}>Undo</button>
      </div>
    ) : null}
    {doneToast ? (
      <div className="toast" role="status" aria-live="polite">
        <span>Analysis complete for {doneToast.name}.</span>
        <button className="btn link" onClick={() => { setDoneToast(null); navTo('reports'); }}>View Report</button>
      </div>
    ) : null}
    </>
  );
};

export default UploadCard;
