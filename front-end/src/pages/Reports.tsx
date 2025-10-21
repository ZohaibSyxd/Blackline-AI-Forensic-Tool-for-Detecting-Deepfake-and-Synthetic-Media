import React, { useEffect, useState, useRef } from "react";
import "./Reports.css";
import { getAnalysesForPage, getAllAnalyses, StoredAnalysisSummary, deleteAnalyses, upsertAnalysis } from '../state/analysisStore';
import { removeFile as removePersistedFile } from '../utils/uploadPersistence';
import { getPlaybackUrl } from '../utils/assetsApi';
import { getAuthState } from '../state/authStore';

// Prefer configured API base; otherwise default to same-origin to work behind proxies
const API_BASE = (import.meta as any).env?.VITE_API_BASE || (typeof window !== 'undefined' ? window.location.origin : "");

const labelFor = (filePage?: string) => {
  if (!filePage) return "FILE ANALYSIS";
  if (filePage === "file1") return "FILE ANALYSIS 1";
  if (filePage === "file2") return "FILE ANALYSIS 2";
  if (filePage === "file3") return "FILE ANALYSIS 3";
  if (filePage === "file$") return "FILE ANALYSIS $";
  return "FILE ANALYSIS";
};

// ---------- Tiny Viz Components (no external deps) ----------
const Sparkline: React.FC<{ values: number[]; width?: number; height?: number; color?: string }>
  = ({ values, width = 260, height = 64, color = 'var(--accent)' }) => {
  if (!values.length) return <div className="sparkline-space" />;
  const w = width, h = height;
  const max = Math.max(0.0001, Math.max(...values));
  const points = values.map((v, i) => {
    const x = (i / Math.max(1, values.length - 1)) * (w - 4) + 2;
    const y = h - 4 - (v / max) * (h - 8);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return (
    <svg className="sparkline" width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-label="Trend sparkline">
      <polyline fill="none" stroke={color} strokeWidth="2" points={points} />
    </svg>
  );
};

const HistogramBars: React.FC<{ bins: number[]; max?: number }>
  = ({ bins, max }) => {
  const m = max ?? Math.max(1, ...bins);
  return (
    <div className="bar-chart" role="img" aria-label="Likelihood distribution">
      {bins.map((c, i) => {
        const pct = Math.max(0, Math.min(100, Math.round((c / m) * 100)));
        const hClass = 'h' + (Math.round(pct/5)*5);
        return (
          <div key={i} className="bar-item" title={`${i*10}-${(i+1)*10}%: ${c}`}>
            <div className="bar-col">
              <div className={`bar ${hClass}`} />
            </div>
            <div className="bar-x">{i*10}</div>
          </div>
        );
      })}
      <div className="bar-x end">100</div>
    </div>
  );
};

const DonutChart: React.FC<{ data: Array<{ label: string; value: number }>; size?: number }>
  = ({ data, size = 160 }) => {
  const total = data.reduce((s,d)=>s+d.value,0) || 1;
  const radius = (size/2) - 10;
  const cx = size/2, cy = size/2;
  const circ = 2 * Math.PI * radius;
  let acc = 0;
  const palette = ['#3D5A80','#10B981','#4F46E5','#8B5CF6','#F43F5E','#F59E0B','#64748B','#0EA5E9'];
  return (
    <div className="donut-wrap">
      <svg className="donut" width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Models used">
        <circle cx={cx} cy={cy} r={radius} fill="none" stroke="#e5e7eb" strokeWidth="14" />
        {data.map((d, i) => {
          const len = (d.value/total) * circ;
          const dasharray = `${len} ${circ-len}`;
          const dashoffset = circ - acc;
          const stroke = palette[i % palette.length];
          acc += len;
          return (
            <circle key={d.label} cx={cx} cy={cy} r={radius} fill="none" stroke={stroke} strokeWidth="14" strokeDasharray={dasharray} strokeDashoffset={dashoffset} />
          );
        })}
      </svg>
      <div className="donut-legend">
        {data.map((d,i)=> (
          <div className="legend-row" key={d.label}><span className={`legend-dot color-${i%8}`} />{d.label}<span className="legend-val">{d.value}</span></div>
        ))}
      </div>
    </div>
  );
};

// --- File Analysis Cards ---
const FileAnalysisCards: React.FC<{ analyses: StoredAnalysisSummary[], onSelect: (id: string) => void, selectedId?: string }> = ({ analyses, onSelect, selectedId }) => {
  if (!analyses.length) return null;
  return (
    <div className="file-analysis-cards">
      {analyses.map(a => (
        <div
          key={a.id}
          className={`file-analysis-card${selectedId === a.id ? ' selected' : ''}`}
          onClick={() => onSelect(a.id)}
        >
          <div className="card-title">{a.fileName}</div>
          <div className="card-info">
            <div>Likelihood: <b>{((a.summary.deepfake_likelihood||0)*100).toFixed(1)}%</b></div>
            <div>Label: <b>{a.summary.deepfake_label || '—'}</b></div>
            <div>Duration: <b>{a.summary.duration_s ? a.summary.duration_s.toFixed(2)+'s' : '—'}</b></div>
            <div>Analyzed: <b>{new Date(a.analyzedAt).toLocaleString()}</b></div>
          </div>
        </div>
      ))}
    </div>
  );
};

const Reports: React.FC<{ filePage?: string }> = ({ filePage }) => {
  const allPageKeys = ["file1", "file2", "file3", "file$"];
  const [activePageKey, setActivePageKey] = useState<string>(filePage || "file1");
  const label = labelFor(activePageKey);
  const [analyses, setAnalyses] = useState<StoredAnalysisSummary[]>([]);
  const [sort, setSort] = useState<{ key: string; dir: 'asc' | 'desc' }>({ key: 'analyzedAt', dir: 'desc' });
  const [selected, setSelected] = useState<StoredAnalysisSummary | null>(null); // single row detail panel
  const [checkedIds, setCheckedIds] = useState<string[]>([]); // multi-select for bulk actions
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [pendingDeleteIds, setPendingDeleteIds] = useState<string[] | null>(null); // for single delete via row "×"
  const [undoBuffer, setUndoBuffer] = useState<StoredAnalysisSummary[] | null>(null);
  const [undoTimer, setUndoTimer] = useState<number | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastCheckIndexRef = useRef<number | null>(null);
  const analysisRegionRef = useRef<HTMLDivElement | null>(null);
  const [videoTime, setVideoTime] = useState(0);
  const [mediaDuration, setMediaDuration] = useState<number | null>(null);
  const [detailVideoSrc, setDetailVideoSrc] = useState<string | undefined>(undefined);
  const [elaLoading, setElaLoading] = useState<boolean>(false);
  const [elaError, setElaError] = useState<string | null>(null);
  const [elaAction, setElaAction] = useState<"generate" | "reload" | null>(null);
  const [elaStatus, setElaStatus] = useState<string | null>(null);
  const [elaFlash, setElaFlash] = useState<boolean>(false);
  const elaDetailsRef = useRef<HTMLDetailsElement | null>(null);
  // ELA viewer state
  const [elaFollowVideo, setElaFollowVideo] = useState<boolean>(true);
  const [elaSelectedIdx, setElaSelectedIdx] = useState<number | null>(null);
  const [elaFramesOverride, setElaFramesOverride] = useState<Array<{ uri: string; time_s?: number|null; index?: number }> | null>(null);
  // LBP viewer state
  const [lbpLoading, setLbpLoading] = useState<boolean>(false);
  const [lbpError, setLbpError] = useState<string | null>(null);
  const [lbpAction, setLbpAction] = useState<"generate" | "reload" | null>(null);
  const [lbpStatus, setLbpStatus] = useState<string | null>(null);
  const [lbpFlash, setLbpFlash] = useState<boolean>(false);
  const lbpDetailsRef = useRef<HTMLDetailsElement | null>(null);
  const [lbpFollowVideo, setLbpFollowVideo] = useState<boolean>(true);
  const [lbpSelectedIdx, setLbpSelectedIdx] = useState<number | null>(null);
  const [lbpFramesOverride, setLbpFramesOverride] = useState<Array<{ uri: string; time_s?: number|null; index?: number }> | null>(null);
  // Noise viewer state
  const [noiseLoading, setNoiseLoading] = useState<boolean>(false);
  const [noiseError, setNoiseError] = useState<string | null>(null);
  const [noiseAction, setNoiseAction] = useState<"generate" | "reload" | null>(null);
  const [noiseStatus, setNoiseStatus] = useState<string | null>(null);
  const [noiseFlash, setNoiseFlash] = useState<boolean>(false);
  const noiseDetailsRef = useRef<HTMLDetailsElement | null>(null);
  const [noiseFollowVideo, setNoiseFollowVideo] = useState<boolean>(true);
  const [noiseSelectedIdx, setNoiseSelectedIdx] = useState<number | null>(null);
  const [noiseFramesOverride, setNoiseFramesOverride] = useState<Array<{ uri: string; fft_uri?: string; time_s?: number|null; index?: number; noise_score?: number }>|null>(null);
  // Add state for selected card
  const [selectedCardId, setSelectedCardId] = useState<string | undefined>(undefined);

  // Jump the source video to a specific time
  const jumpToTime = React.useCallback((t: number, autoPlay = false) => {
    const v = videoRef.current;
    if (!v) return;
    const target = Math.max(0, t);
    try {
      if (Number.isFinite(v.duration) && v.duration > 0) {
        v.currentTime = Math.min(target, v.duration - 0.05);
      } else {
        const onMeta = () => {
          v.currentTime = Math.min(target, Math.max(0, (v.duration || target) - 0.05));
          if (!autoPlay) v.pause(); else void v.play();
          v.removeEventListener('loadedmetadata', onMeta);
        };
        v.addEventListener('loadedmetadata', onMeta);
      }
      if (!autoPlay) v.pause(); else void v.play();
    } catch {}
  }, []);

  // Load list for current page
  useEffect(() => {
    const list = activePageKey ? getAnalysesForPage(activePageKey) : getAllAnalyses();
    setAnalyses(list);
    setCheckedIds([]);
  }, [activePageKey]);

  // On first render after list loads, auto-focus a requested analysis (from Uploads View Report)
  useEffect(() => {
    if (!analyses.length) return;
    let hint: { id?: string; pageKey?: string; ts?: number } | null = null;
    try {
      const raw = sessionStorage.getItem('bl_reports_focus');
      if (raw) hint = JSON.parse(raw);
    } catch {}
    if (!hint || !hint.id) return;
    if (hint.pageKey && filePage && hint.pageKey !== filePage) return; // different filePage; ignore
    const target = analyses.find(a => a.id === hint!.id);
    if (!target) return;
    setSelected(target);
    // Scroll that row into view and add a temporary flash highlight
    setTimeout(() => {
      try {
        let row: HTMLElement | null = null;
        try {
          const esc = (window as any).CSS && typeof (window as any).CSS.escape === 'function' ? (window as any).CSS.escape : (s: string) => s.replace(/"/g, '\\"');
          row = document.querySelector(`.analysis-table .tbody .tr[data-id="${esc(target.id)}"]`) as HTMLElement | null;
        } catch {}
        if (!row) {
          // Fallback: iterate rows and match data-id
          const rows = Array.from(document.querySelectorAll('.analysis-table .tbody .tr')) as HTMLElement[];
          row = rows.find(r => r.dataset && r.dataset.id === target.id) || null;
        }
        if (row) {
          row.scrollIntoView({ behavior: 'smooth', block: 'center' });
          row.classList.add('flash');
          window.setTimeout(() => row && row.classList.remove('flash'), 1500);
        }
      } catch {}
      try { sessionStorage.removeItem('bl_reports_focus'); } catch {}
    }, 50);
  }, [analyses, filePage]);

  // Close detail on click outside or Escape
  useEffect(() => {
    if (!selected) return;
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') setSelected(null); }
    function onClick(e: MouseEvent) {
      if (!analysisRegionRef.current) return;
      const region = analysisRegionRef.current;
      if (region.contains(e.target as Node)) {
        const detail = region.querySelector('.analysis-detail');
        if (detail && detail.contains(e.target as Node)) return;
        const row = (e.target as HTMLElement).closest('.analysis-table .tr');
        if (row) return;
      }
      setSelected(null);
    }
    document.addEventListener('keydown', onKey);
    document.addEventListener('mousedown', onClick);
    return () => { document.removeEventListener('keydown', onKey); document.removeEventListener('mousedown', onClick); };
  }, [selected]);

  // Reset play state when switching selection
  useEffect(() => {
    setVideoTime(0);
    setMediaDuration(null);
    setElaSelectedIdx(null);
    setLbpSelectedIdx(null);
  }, [selected]);

  // Resolve playback URL for the selected item (once per selection)
  useEffect(() => {
    let canceled = false;
    (async () => {
      try {
        const rawAsset = (selected as any)?.raw?.asset as any;
        const stored = rawAsset?.stored_path as string | undefined;
        const assetIdUnknown = (rawAsset as any)?.asset_id;
        if (!selected) { if (!canceled) setDetailVideoSrc(undefined); return; }
        // Only call playback-url for numeric DB IDs; legacy analyze returns a UUID-like string here
        let numericId: number | null = null;
        if (assetIdUnknown !== undefined && assetIdUnknown !== null) {
          const n = Number(assetIdUnknown);
          if (Number.isFinite(n) && String(n) === String(assetIdUnknown).trim()) {
            numericId = n;
          }
        }
        if (numericId !== null) {
          const url = await getPlaybackUrl(numericId);
          // Backend may return a relative path for local storage (e.g. "/assets/<key>").
          // Ensure we resolve that against the configured API base so the browser
          // requests the asset from the API server (not the front-end dev server).
          const resolved = ((): string => {
            try {
              if (!url) return url as string;
              if (/^https?:\/\//i.test(url)) return url;
              const base = String(API_BASE).replace(/\/$/, '');
              if (url.startsWith('/')) return `${base}${url}`;
              return `${base}/${url}`;
            } catch {
              return url as string;
            }
          })();
          if (!canceled) setDetailVideoSrc(resolved);
          return;
        }
        if (stored) {
          const url = `${String(API_BASE).replace(/\/$/, '')}/assets/${String(stored).replace(/^\/+/, '')}`;
          if (!canceled) setDetailVideoSrc(url);
          return;
        }
        if (!canceled) setDetailVideoSrc(undefined);
      } catch {
        if (!canceled) setDetailVideoSrc(undefined);
      }
    })();
    return () => { canceled = true; };
  }, [selected?.id, selected?.analyzedAt]);

  const allChecked = analyses.length>0 && checkedIds.length === analyses.length;
  function toggleCheck(id: string) {
    setCheckedIds(prev => prev.includes(id) ? prev.filter(x=>x!==id) : [...prev, id]);
  }
  function handleCheckboxClick(e: React.MouseEvent<HTMLInputElement>, id: string, index: number) {
    e.stopPropagation();
    const checked = checkedIds.includes(id);
    const shift = e.shiftKey;
    if (shift && lastCheckIndexRef.current !== null && !checked) {
      const start = Math.min(lastCheckIndexRef.current, index);
      const end = Math.max(lastCheckIndexRef.current, index);
      const rangeIds = analyses.slice(start, end+1).map(a=>a.id);
      setCheckedIds(prev => Array.from(new Set([...prev, ...rangeIds])));
    } else {
      toggleCheck(id);
    }
    lastCheckIndexRef.current = index;
  }
  function toggleAll() {
    if (allChecked) setCheckedIds([]); else setCheckedIds(analyses.map(a=>a.id));
  }
  function bulkDelete() {
    if (!checkedIds.length) return;
    // open custom modal instead of native confirm
    setShowDeleteConfirm(true);
  }
  function performDelete(ids: string[]) {
    if (!ids.length) return;
    const idSet = new Set(ids);
    const toDelete = analyses.filter(a => idSet.has(a.id));
    // Also remove from Uploads (Documents) page storage and persisted blobs
    try {
      const storageKey = filePage ? `bl_uploadItems_${filePage}_v3` : `bl_uploadItems__global_v3`;
      const raw = localStorage.getItem(storageKey);
      if (raw) {
        try {
          const arr = JSON.parse(raw);
          if (Array.isArray(arr)) {
            const filtered = arr.filter((x: any) => !idSet.has(x.id));
            localStorage.setItem(storageKey, JSON.stringify(filtered));
          }
        } catch {}
      }
      // remove persisted File blobs in IndexedDB (fire and forget)
      ids.forEach(id => {
        const composite = filePage ? `${filePage}::${id}` : id;
        try { void removePersistedFile(composite); } catch {}
      });
    } catch {}
    deleteAnalyses(ids, filePage);
    // Broadcast deletion so UploadCard can clear in-memory refs if mounted
    try {
      const ev = new CustomEvent('bl:uploads-removed', { detail: { pageKey: filePage || 'global', ids } });
      window.dispatchEvent(ev);
    } catch {}
    const list = filePage ? getAnalysesForPage(filePage) : getAllAnalyses();
    list.sort((a,b)=> b.analyzedAt - a.analyzedAt);
    setAnalyses(list);
    setCheckedIds(prev => prev.filter(id => !idSet.has(id)));
    if (selected && idSet.has(selected.id)) setSelected(null);
    setShowDeleteConfirm(false);
    setPendingDeleteIds(null);
    if (undoTimer) window.clearTimeout(undoTimer);
    setUndoBuffer(toDelete);
    const t = window.setTimeout(()=> { setUndoBuffer(null); setUndoTimer(null); }, 7000);
    setUndoTimer(t);
  }
  function performBulkDelete() { performDelete(checkedIds); }
  function cancelBulkDelete() {
    setShowDeleteConfirm(false);
    setPendingDeleteIds(null);
  }
  function undoDelete() {
    if (!undoBuffer) return;
    // restore items (keep original analyzedAt, id)
    undoBuffer.forEach(item => upsertAnalysis(item));
    const list = filePage ? getAnalysesForPage(filePage) : getAllAnalyses();
    list.sort((a,b) => b.analyzedAt - a.analyzedAt);
    setAnalyses(list);
    setUndoBuffer(null);
    if (undoTimer) { window.clearTimeout(undoTimer); setUndoTimer(null); }
  }

  // Keyboard shortcuts for modal: ESC = cancel, Enter = confirm
  const modalOpen = showDeleteConfirm || !!pendingDeleteIds;
  useEffect(() => {
    if (!modalOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        cancelBulkDelete();
      } else if (e.key === 'Enter') {
        e.preventDefault();
        performDelete(pendingDeleteIds || checkedIds);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [modalOpen, pendingDeleteIds, checkedIds]);

  function handleSingleDelete(item: StoredAnalysisSummary, e?: React.MouseEvent) {
    if (e) e.stopPropagation();
    // open confirmation modal for single delete
    setPendingDeleteIds([item.id]);
  }

  const aggregate = React.useMemo(() => {
    if (!analyses.length) return null;
    const total = analyses.length;
    const avgLikelihood = analyses.reduce((s,a)=> s + (a.summary.deepfake_likelihood || 0),0)/total;
    const maxLikelihood = analyses.reduce((m,a)=> Math.max(m, a.summary.deepfake_likelihood||0),0);
    const suspicious = analyses.filter(a => (a.summary.deepfake_likelihood||0) > 0.5).length;
    // histogram (0..100 step 10)
    const hist = new Array(10).fill(0) as number[];
    analyses.forEach(a => {
      const v = Math.max(0, Math.min(0.9999, a.summary.deepfake_likelihood || 0));
      const b = Math.floor(v*10);
      hist[b] += 1;
    });
    // models used
    const modelMap: Record<string, number> = {};
    analyses.forEach(a => {
      const m = a.summary.deepfake_method || 'unknown';
      modelMap[m] = (modelMap[m]||0)+1;
    });
    // sparkline series by time
    const series = [...analyses].sort((a,b)=> a.analyzedAt - b.analyzedAt).map(a=> a.summary.deepfake_likelihood || 0);
    return { total, avgLikelihood, maxLikelihood, suspicious, hist, modelMap, series };
  }, [analyses]);

  // Apply sorting whenever list or sort state changes
  useEffect(() => {
    if (!analyses.length) return;
    setAnalyses(prev => {
      const arr = [...prev];
      const dir = sort.dir === 'asc' ? 1 : -1;
      arr.sort((a,b) => {
        function cmpNums(x:number,y:number){ return x===y?0:(x<y?-1:1); }
        let res = 0;
        switch (sort.key) {
          case 'file': res = a.fileName.localeCompare(b.fileName); break;
          case 'duration': res = cmpNums(a.summary.duration_s||0, b.summary.duration_s||0); break;
          case 'likelihood': res = cmpNums(a.summary.deepfake_likelihood||0, b.summary.deepfake_likelihood||0); break;
          case 'label': res = (a.summary.deepfake_label||'').localeCompare(b.summary.deepfake_label||''); break;
          case 'analyzedAt': res = cmpNums(a.analyzedAt, b.analyzedAt); break;
          default: res = 0;
        }
        return res * dir;
      });
      return arr;
    });
  }, [sort.key, sort.dir]);

  // Re-run sorting when analyses list itself changes (e.g., new / delete) respecting current sort
  useEffect(() => {
    if (!analyses.length) return;
    setSort(s => ({ ...s })); // trigger effect above by shallow state churn
  }, [analyses.length]);

  function onSort(key: string) {
    setSort(prev => prev.key === key ? { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: key === 'analyzedAt' ? 'desc' : 'asc' });
  }

  function ariaSortAttr(key:string) {
    if (sort.key !== key) return undefined; // omit attribute when unsorted per WAI-ARIA practices
    return sort.dir === 'asc' ? 'ascending' : 'descending';
  }

  // Memoize ELA frames extracted from selected summary
  type ELAFrame = { uri: string; time_s?: number|null; index?: number };
  const elaFrames: ELAFrame[] = React.useMemo(() => {
    if (elaFramesOverride && elaFramesOverride.length) return elaFramesOverride as ELAFrame[];
    if (!selected) return [];
    const s: any = selected;
    const grab = (obj: any) => {
      if (!obj) return { uris: null as string[]|null, frames: null as any[]|null };
      const directUris = Array.isArray(obj.ela_uris) ? obj.ela_uris as string[] : null;
      const directFrames = Array.isArray(obj.ela_frames) ? obj.ela_frames as any[] : null;
      const nestedUris = Array.isArray(obj.ela?.uris) ? obj.ela.uris as string[] : null;
      const nestedFrames = Array.isArray(obj.ela?.frames) ? obj.ela.frames as any[] : null;
      return { uris: directUris || nestedUris, frames: directFrames || nestedFrames };
    };
    const a = grab(s?.summary);
    const b = grab(s?.raw?.summary);
    let frames: ELAFrame[] = [];
    if (a.frames && a.frames.length) frames = a.frames as ELAFrame[];
    else if (a.uris && a.uris.length) frames = (a.uris as string[]).map((u) => ({ uri: u }));
    else if (b.frames && b.frames.length) frames = b.frames as ELAFrame[];
    else if (b.uris && b.uris.length) frames = (b.uris as string[]).map((u) => ({ uri: u }));
    // Normalize and dedupe by URI
    const seen = new Set<string>();
    const list = (frames || []).filter((f: any) => !!f && typeof f.uri === 'string' && f.uri.length>0)
      .filter((f: any) => { if (seen.has(f.uri)) return false; seen.add(f.uri); return true; }) as ELAFrame[];
    return list;
  }, [selected?.id, selected?.analyzedAt, elaFramesOverride]);

  // Memoize LBP frames extracted from selected summary
  type LBPFrame = { uri: string; time_s?: number|null; index?: number };
  const lbpFrames: LBPFrame[] = React.useMemo(() => {
    if (lbpFramesOverride && lbpFramesOverride.length) return lbpFramesOverride as LBPFrame[];
    if (!selected) return [];
    const s: any = selected;
    const grab = (obj: any) => {
      if (!obj) return { uris: null as string[]|null, frames: null as any[]|null };
      const directUris = Array.isArray(obj.lbp_uris) ? obj.lbp_uris as string[] : null;
      const directFrames = Array.isArray(obj.lbp_frames) ? obj.lbp_frames as any[] : null;
      const nestedUris = Array.isArray(obj.lbp?.uris) ? obj.lbp.uris as string[] : null;
      const nestedFrames = Array.isArray(obj.lbp?.frames) ? obj.lbp.frames as any[] : null;
      return { uris: directUris || nestedUris, frames: directFrames || nestedFrames };
    };
    const a = grab(s?.summary);
    const b = grab(s?.raw?.summary);
    let frames: LBPFrame[] = [];
    if (a.frames && a.frames.length) frames = a.frames as LBPFrame[];
    else if (a.uris && a.uris.length) frames = (a.uris as string[]).map((u) => ({ uri: u }));
    else if (b.frames && b.frames.length) frames = b.frames as LBPFrame[];
    else if (b.uris && b.uris.length) frames = (b.uris as string[]).map((u) => ({ uri: u }));
    const seen = new Set<string>();
    const list = (frames || []).filter((f: any) => !!f && typeof f.uri === 'string' && f.uri.length>0)
      .filter((f: any) => { if (seen.has(f.uri)) return false; seen.add(f.uri); return true; }) as LBPFrame[];
    return list;
  }, [selected?.id, selected?.analyzedAt, lbpFramesOverride]);

  // Memoize Noise frames with metrics
  type NoiseFrame = { uri: string; fft_uri?: string; time_s?: number|null; index?: number; residual_abs_mean?: number; residual_std?: number; residual_energy?: number; fft_low_ratio?: number; fft_high_ratio?: number; noise_score?: number };
  const noiseFrames: NoiseFrame[] = React.useMemo(() => {
    if (noiseFramesOverride && noiseFramesOverride.length) return noiseFramesOverride as NoiseFrame[];
    if (!selected) return [];
    const s: any = selected;
    const grab = (obj: any) => {
      if (!obj) return { frames: null as any[]|null };
      const direct = Array.isArray(obj.noise_frames) ? obj.noise_frames as any[] : null;
      const nested = Array.isArray(obj.noise?.frames) ? obj.noise.frames as any[] : null;
      return { frames: direct || nested };
    };
    const a = grab(s?.summary);
    const b = grab(s?.raw?.summary);
    let frames: NoiseFrame[] = [];
    if (a.frames && a.frames.length) frames = a.frames as NoiseFrame[];
    else if (b.frames && b.frames.length) frames = b.frames as NoiseFrame[];
    // Dedup by uri
    const seen = new Set<string>();
    const list = (frames || []).filter((f:any)=> !!f && typeof f.uri === 'string' && f.uri.length>0)
      .filter((f:any)=>{ if (seen.has(f.uri)) return false; seen.add(f.uri); return true; }) as NoiseFrame[];
    return list;
  }, [selected?.id, selected?.analyzedAt, noiseFramesOverride]);

  // Auto-follow: update current ELA selection based on video time
  useEffect(() => {
    if (!elaFollowVideo) return;
    if (!elaFrames.length) return;
    const withTimes = elaFrames.filter(x => typeof x.time_s === 'number' && Number.isFinite(x.time_s as any));
    if (!withTimes.length) return;
    let bestI = 0; let bestD = Infinity;
    elaFrames.forEach((f, i) => {
      const t: any = f.time_s;
      if (typeof t === 'number' && Number.isFinite(t)) {
        const d = Math.abs((t as number) - (videoTime || 0));
        if (d < bestD) { bestD = d; bestI = i; }
      }
    });
    if (elaSelectedIdx !== bestI) setElaSelectedIdx(bestI);
  }, [videoTime, elaFollowVideo, elaFrames]);

  // Auto-follow for LBP
  useEffect(() => {
    if (!lbpFollowVideo) return;
    if (!lbpFrames.length) return;
    const withTimes = lbpFrames.filter(x => typeof x.time_s === 'number' && Number.isFinite(x.time_s as any));
    if (!withTimes.length) return;
    let bestI = 0; let bestD = Infinity;
    lbpFrames.forEach((f, i) => {
      const t: any = f.time_s;
      if (typeof t === 'number' && Number.isFinite(t)) {
        const d = Math.abs((t as number) - (videoTime || 0));
        if (d < bestD) { bestD = d; bestI = i; }
      }
    });
    if (lbpSelectedIdx !== bestI) setLbpSelectedIdx(bestI);
  }, [videoTime, lbpFollowVideo, lbpFrames]);

  // Auto-follow for Noise
  useEffect(() => {
    if (!noiseFollowVideo) return;
    if (!noiseFrames.length) return;
    const withTimes = noiseFrames.filter(x => typeof x.time_s === 'number' && Number.isFinite(x.time_s as any));
    if (!withTimes.length) return;
    let bestI = 0; let bestD = Infinity;
    noiseFrames.forEach((f, i) => {
      const t: any = f.time_s;
      if (typeof t === 'number' && Number.isFinite(t)) {
        const d = Math.abs((t as number) - (videoTime || 0));
        if (d < bestD) { bestD = d; bestI = i; }
      }
    });
    if (noiseSelectedIdx !== bestI) setNoiseSelectedIdx(bestI);
  }, [videoTime, noiseFollowVideo, noiseFrames]);

  return (
    <div className="reports-page">
      <div className="reports-inner">
        <div className="reports-header">
          <div>
            <h2>Reports</h2>
          </div>
        </div>
        {/* Page cards removed from Reports — they appear only on Dashboard */}
        {/* Per-analysis summary cards removed — Reports now shows aggregates and the table only */}
        {aggregate && (
          <>
            <div className="report-stats">
              <div className="stat"><div className="label">Analyses</div><div className="value">{aggregate.total}</div></div>
              <div className="stat"><div className="label">Avg Likelihood</div><div className="value">{(aggregate.avgLikelihood*100).toFixed(1)}%</div></div>
              <div className="stat"><div className="label">Max Likelihood</div><div className="value">{(aggregate.maxLikelihood*100).toFixed(1)}%</div></div>
              <div className="stat"><div className="label">Flagged (&gt;50%)</div><div className="value">{aggregate.suspicious}</div></div>
            </div>
            <div className="viz-grid">
              <div className="viz-card">
                <div className="viz-title">Likelihood Trend</div>
                <Sparkline values={aggregate.series} />
              </div>
              <div className="viz-card">
                <div className="viz-title">Likelihood Distribution</div>
                <HistogramBars bins={aggregate.hist} />
              </div>
              <div className="viz-card">
                <div className="viz-title">Models Used</div>
                <DonutChart data={Object.entries(aggregate.modelMap).map(([label,value])=>({label, value}))} />
              </div>
            </div>
          </>
        )}

        {!analyses.length && (
          <div className="no-analyses">No analyses yet for this page. Run an analysis to populate this report.</div>
        )}

        {analyses.length > 0 && (
          <div className="analysis-table-wrapper" ref={analysisRegionRef}>
            {checkedIds.length>0 && (
              <div className="selection-bar floating">
                <div className="sel-count">{checkedIds.length} selected</div>
                <button className="btn danger" onClick={bulkDelete}>Delete Selected</button>
                <button className="btn ghost" onClick={()=>setCheckedIds([])}>Clear</button>
              </div>
            )}
            <div className="analysis-table" role="table" aria-label="Analysis summaries">
              <div className="thead" role="rowgroup">
                <div className="tr" role="row">
                  <div className="th sel" role="columnheader"><input type="checkbox" aria-label="Select all" checked={allChecked} onChange={toggleAll}/></div>
                  <div className={`th sortable ${sort.key==='file' ? 'sorted '+sort.dir : ''}`} role="columnheader" {...(sort.key==='file' ? { 'aria-sort': (sort.dir==='asc'?'ascending':'descending') } : {})} onClick={()=>onSort('file')} tabIndex={0} onKeyDown={(e)=>{ if(e.key==='Enter' || e.key===' ') { e.preventDefault(); onSort('file'); }}}>File <span className="sort-indicator" aria-hidden="true">{sort.key==='file' ? (sort.dir==='asc'?'▲':'▼') : '↕'}</span></div>
                  <div className={`th sortable ${sort.key==='duration' ? 'sorted '+sort.dir : ''}`} role="columnheader" {...(sort.key==='duration' ? { 'aria-sort': (sort.dir==='asc'?'ascending':'descending') } : {})} onClick={()=>onSort('duration')} tabIndex={0} onKeyDown={(e)=>{ if(e.key==='Enter' || e.key===' ') { e.preventDefault(); onSort('duration'); }}}>Duration <span className="sort-indicator" aria-hidden="true">{sort.key==='duration' ? (sort.dir==='asc'?'▲':'▼') : '↕'}</span></div>
                  <div className={`th sortable ${sort.key==='likelihood' ? 'sorted '+sort.dir : ''}`} role="columnheader" {...(sort.key==='likelihood' ? { 'aria-sort': (sort.dir==='asc'?'ascending':'descending') } : {})} onClick={()=>onSort('likelihood')} tabIndex={0} onKeyDown={(e)=>{ if(e.key==='Enter' || e.key===' ') { e.preventDefault(); onSort('likelihood'); }}}>Deepfake Likelihood <span className="sort-indicator" aria-hidden="true">{sort.key==='likelihood' ? (sort.dir==='asc'?'▲':'▼') : '↕'}</span></div>
                  <div className={`th sortable ${sort.key==='label' ? 'sorted '+sort.dir : ''}`} role="columnheader" {...(sort.key==='label' ? { 'aria-sort': (sort.dir==='asc'?'ascending':'descending') } : {})} onClick={()=>onSort('label')} tabIndex={0} onKeyDown={(e)=>{ if(e.key==='Enter' || e.key===' ') { e.preventDefault(); onSort('label'); }}}>Label <span className="sort-indicator" aria-hidden="true">{sort.key==='label' ? (sort.dir==='asc'?'▲':'▼') : '↕'}</span></div>
                  <div className={`th sortable ${sort.key==='analyzedAt' ? 'sorted '+sort.dir : ''}`} role="columnheader" {...(sort.key==='analyzedAt' ? { 'aria-sort': (sort.dir==='asc'?'ascending':'descending') } : {})} onClick={()=>onSort('analyzedAt')} tabIndex={0} onKeyDown={(e)=>{ if(e.key==='Enter' || e.key===' ') { e.preventDefault(); onSort('analyzedAt'); }}}>Analyzed <span className="sort-indicator" aria-hidden="true">{sort.key==='analyzedAt' ? (sort.dir==='asc'?'▲':'▼') : '↕'}</span></div>
                </div>
              </div>
              <div className="tbody" role="rowgroup">
                {analyses.map((a, idx) => {
                  const likelihood = (a.summary.deepfake_likelihood||0);
                  const pctNum = Math.max(0, Math.min(100, Math.round(likelihood*100)));
                  const pct = pctNum.toFixed(1) + '%';
                  const flagged = likelihood > 0.5;
                  const bucket = Math.round(pctNum/5)*5; // 0,5,...100
                  const pctClass = 'p'+bucket;
                  const isActive = selected?.id===a.id && selected.analyzedAt===a.analyzedAt;
                  const isChecked = checkedIds.includes(a.id);
                  const showDelete = isActive || isChecked; // show when row is active (clicked) or checkbox-selected
                  const modelTag = (() => {
                    const m = (a.summary.deepfake_method || '').toLowerCase();
                    if (!m) return null;
                    if (m.includes('fusion-lr')) return 'fusion-lr';
                    if (m.includes('fusion-blend')) return 'fusion-blend';
                    if (m.includes('stub') || m.includes('hash')) return 'hash-stub';
                    return null;
                  })();
                  return (
                    <React.Fragment key={a.id+"-"+a.analyzedAt}>
                      <div
                        className={`tr ${flagged ? 'flagged' : ''} ${isActive ? 'active' : ''} ${showDelete ? 'has-action' : ''}`}
                        role="row"
                        data-id={a.id}
                        onClick={(e) => {
                          const el = (e.target as HTMLElement);
                          if (el.closest('.sel input') || el.closest('.row-delete')) return; // ignore clicks on checkbox/delete button
                          if (isActive) { setSelected(null); } else { setSelected(a); }
                        }}
                      >
                        <div className="td sel" role="cell"><input type="checkbox" aria-label={`Select ${a.fileName}`} checked={checkedIds.includes(a.id)} readOnly onClick={(e)=>handleCheckboxClick(e, a.id, idx)} /></div>
                        <div className="td filename" role="cell" title={a.fileName}>{a.fileName}</div>
                        <div className="td" role="cell">{a.summary.duration_s ? a.summary.duration_s.toFixed(2)+'s' : '—'}</div>
                        <div className="td likelihood-cell" role="cell">
                          <div className={`likelihood-meter ${flagged? 'alert': ''} ${pctClass}`}> <div className="fill" /> </div>
                          <span className={`pct-text ${flagged? 'alert': ''}`}>{pct}</span>
                        </div>
                        <div className="td" role="cell">{a.summary.deepfake_label || '—'}</div>
                        <div className="td analyzed-cell" role="cell">
                          {modelTag ? <span className={`model-badge ${modelTag}`}>{modelTag}</span> : null}
                          <span className="analyzed-text">{new Date(a.analyzedAt).toLocaleString()}</span>
                          <button className="row-delete" aria-label={`Delete ${a.fileName}`} title="Delete" onClick={(e)=>handleSingleDelete(a, e)}>×</button>
                        </div>
                      </div>
                      {isActive && selected && (
                        <div className="tr detail-row" role="row" key={a.id+"-"+a.analyzedAt+"-detail"}>
                          <div className="td detail-cell" role="cell">
                            {/* Inline analysis detail below the active row */}
                            <div className="analysis-detail embedded" role="region" aria-label="Detailed analysis">
                              <div className="detail-head">
                                <h3 className="detail-title">{selected.fileName}</h3>
                                <button className="detail-close" onClick={()=>setSelected(null)} aria-label="Close details">×</button>
                              </div>
                              <div className="detail-grid">
                                <div><span className="k">Resolution</span><span className="v">{selected.summary.width && selected.summary.height ? `${selected.summary.width}x${selected.summary.height}` : '—'}</span></div>
                                <div><span className="k">FPS</span><span className="v">{selected.summary.fps ?? '—'}</span></div>
                                <div><span className="k">Format Valid</span><span className="v">{String(selected.summary.format_valid)}</span></div>
                                <div><span className="k">Decode Valid</span><span className="v">{String(selected.summary.decode_valid)}</span></div>
                                <div><span className="k">Deepfake %</span><span className="v">{((selected.summary.deepfake_likelihood||0)*100).toFixed(2)}%</span></div>
                                <div><span className="k">Label</span><span className="v">{selected.summary.deepfake_label || '—'}</span></div>
                              </div>
                              {/* Overlay, video, frame timeline, and ELA blocks reuse existing state */}
                              { (selected.summary as any)?.overlay_uri ? (
                                <details className="overlay-block" open>
                                  <summary>Overlay</summary>
                                  <div className="overlay-preview">
                                    <img
                                      src={`${String(API_BASE).replace(/\/$/, '')}/static/${String(((selected.summary as any).overlay_uri || '')).replace(/^\/+/, '')}`}
                                      alt="Model overlay"
                                    />
                                  </div>
                                </details>
                              ) : null}
                              {(() => {
                                const rawAsset = (selected as any)?.raw?.asset as any;
                                const stored = rawAsset?.stored_path as string|undefined;
                                const fs = (selected as any)?.raw?.summary?.frame_scores as Array<{ index:number; time_s:number|null; prob:number }>|undefined;
                                const fps = (selected as any)?.raw?.summary?.frame_fps as number|undefined;
                                const hasAny = !!detailVideoSrc || (fs && fs.length>0);
                                if (!hasAny) return null;
                                const maxH = 56; // px
                                const duration = (selected.summary.duration_s as number|undefined) ?? (typeof fps==='number' && (selected as any)?.raw?.summary?.frame_total ? ((selected as any).raw.summary.frame_total / Math.max(1, fps)) : undefined);
                                const effDuration = (mediaDuration && mediaDuration>0 ? mediaDuration : duration);
                                return (
                                  <details className="overlay-block" open>
                                    <summary>Source video &amp; Frame likelihoods</summary>
                                    <div className="overlay-content">
                                      {detailVideoSrc && (
                                        <div className="overlay-preview resizable">
                                          <video
                                            ref={videoRef}
                                            key={(stored||'')+':'+(selected?.id||'')}
                                            src={detailVideoSrc}
                                            controls
                                            preload="metadata"
                                            playsInline
                                            className="video-fluid"
                                            onLoadedMetadata={(e)=>{ try { const d = (e.currentTarget as HTMLVideoElement).duration; if (Number.isFinite(d) && d>0) setMediaDuration(d); } catch {} }}
                                            onTimeUpdate={(e)=>{ try { setVideoTime((e.currentTarget as HTMLVideoElement).currentTime || 0); } catch {} }}
                                          />
                                        </div>
                                      )}
                                      {fs && Array.isArray(fs) && fs.length>0 && (
                                        <>
                                        <div className="frame-scores" role="list">
                                          {fs.map((f, i) => {
                                            const h = Math.max(2, Math.round((Math.max(0, Math.min(1, f.prob))) * maxH));
                                            const over = f.prob >= 0.5;
                                            const tLabel = (f.time_s!=null ? `${f.time_s.toFixed(2)}s` : (fps ? `${(f.index/Math.max(1,fps)).toFixed(2)}s` : `#${f.index}`));
                                            return (
                                              <div
                                                key={i}
                                                role="listitem"
                                                className={`fs-bar ${over ? 'alert' : ''}`}
                                                title={`Frame ${f.index} • t=${tLabel} • p=${(f.prob*100).toFixed(1)}%`}
                                                aria-label={`Frame ${f.index}, ${tLabel}, probability ${(f.prob*100).toFixed(1)} percent`}
                                                data-h={h}
                                                onClick={()=>{
                                                  const fpsVal = (selected as any)?.raw?.summary?.frame_fps as number | undefined;
                                                  const fpsFallback = (selected?.summary?.fps as number | undefined);
                                                  const effFps = (typeof fpsVal==='number' && fpsVal>0) ? fpsVal : (typeof fpsFallback==='number' && fpsFallback>0 ? fpsFallback : undefined);
                                                  const tt = (f.time_s!=null ? f.time_s : (effFps ? (f.index/effFps) : undefined));
                                                  if (tt!=null) jumpToTime(tt, false);
                                                }}
                                              />
                                            );
                                          })}
                                        </div>
                                        <div className="frame-meta">
                                          <span>samples: {fs.length}</span>
                                          {typeof fps === 'number' && fps>0 ? <span> · fps: {fps.toFixed(2)}</span> : null}
                                          <span> · <strong>blue</strong>=lower, <strong>red</strong>=higher likelihood</span>
                                          <span> · threshold ≥ 50% outlined</span>
                                        </div>
                                        {(() => {
                                          if (!effDuration) return null;
                                          const items = fs.map((f)=>{
                                            const t = f.time_s!=null ? f.time_s : (typeof fps==='number' && fps>0 ? (f.index/Math.max(1,fps)) : undefined);
                                            const leftPct = t!=null ? Math.max(0, Math.min(100, Math.round((t/effDuration)*100))) : (typeof (selected as any)?.raw?.summary?.frame_total==='number' && (selected as any).raw.summary.frame_total>0 ? Math.max(0, Math.min(100, Math.round((f.index/((selected as any).raw.summary.frame_total))*100))) : 0);
                                            const probPct = Math.max(0, Math.min(100, Math.round(f.prob*100)));
                                            const hBucket = Math.round(probPct/5)*5;
                                            return { leftPct, hBucket, idx: f.index, time: t, prob: f.prob };
                                          });
                                          const headLeft = Math.max(0, Math.min(100, Math.round((Math.max(0, videoTime) / effDuration) * 100)));
                                          return (
                                            <div className="frame-timeline" role="region" aria-label="Frame timeline">
                                              <div className="ft-base" aria-hidden="true" />
                                              <div className={`ft-head ft-p${headLeft}`} aria-hidden="true" />
                                              {items.map((it, i) => (
                                                <button
                                                  key={i}
                                                  type="button"
                                                  className={`ft-mark ft-p${it.leftPct} h${it.hBucket} ${it.prob>=0.5?'alert':''}`}
                                                  title={`t=${it.time!=null?it.time.toFixed(2)+'s':('frame #'+it.idx)} • p=${(it.prob*100).toFixed(1)}%`}
                                                  data-tip={`${it.time!=null?it.time.toFixed(2)+'s':('frame #'+it.idx)} • ${(it.prob*100).toFixed(1)}%`}
                                                  aria-label={`Timeline marker at ${it.time!=null?it.time.toFixed(2)+' seconds':'frame '+it.idx}, probability ${(it.prob*100).toFixed(1)} percent`}
                                                  onClick={()=>{ if (it.time!=null) jumpToTime(it.time, false); }}
                                                />
                                              ))}
                                              <div className="ft-axis" aria-hidden="true">
                                                <span>0s</span>
                                                <span>{effDuration.toFixed(2)}s</span>
                                              </div>
                                            </div>
                                          );
                                        })()}
                                        </>
                                      )}
                                    </div>
                                  </details>
                                );
                              })()}
                              {(() => {
                                // Noise frames viewer + grid with metrics
                                const list = noiseFrames;
                                const assetIdUnknown = (selected as any)?.raw?.asset?.asset_id;
                                const numericId = ((): number | null => {
                                  if (assetIdUnknown === undefined || assetIdUnknown === null) return null;
                                  const n = Number(assetIdUnknown);
                                  return (Number.isFinite(n) && String(n) === String(assetIdUnknown).trim()) ? n : null;
                                })();
                                async function handleGenerateNoise() {
                                  setNoiseError(null); setNoiseStatus(null); setNoiseAction('generate'); setNoiseLoading(true);
                                  try {
                                    const tok = getAuthState().token;
                                    const headers: Record<string,string> = { 'Content-Type': 'application/json' };
                                    if (tok) headers['Authorization'] = `Bearer ${tok}`;
                                    let res: Response;
                                    if (numericId !== null) {
                                      res = await fetch(`${API_BASE}/api/noise/generate`, { method: 'POST', headers, body: JSON.stringify({ asset_id: numericId, frames: 12, method: 'both', threshold: 0.6 }) });
                                    } else {
                                      const sp = (selected as any)?.raw?.asset?.stored_path;
                                      const sha = (selected as any)?.raw?.asset?.sha256;
                                      if (!sp) throw new Error('No asset reference for Noise');
                                      res = await fetch(`${API_BASE}/api/noise/generate-by-path`, { method: 'POST', headers, body: JSON.stringify({ stored_path: sp, sha256: sha, frames: 12, method: 'both', threshold: 0.6 }) });
                                    }
                                    if (!res.ok) throw new Error(await res.text());
                                    const data = await res.json();
                                    const listNew = Array.isArray(data.noise_frames) ? data.noise_frames : [];
                                    const updated = { ...selected } as any;
                                    updated.summary = { ...(updated.summary || {}), noise_frames: listNew, noise: { frames: listNew, high_indices: data.high_indices, threshold: data.threshold } };
                                    if ((updated as any).raw && (updated as any).raw.summary) {
                                      (updated as any).raw.summary = { ...((updated as any).raw.summary || {}), noise_frames: listNew, noise: { frames: listNew, high_indices: data.high_indices, threshold: data.threshold } };
                                    }
                                    setNoiseSelectedIdx(0);
                                    setNoiseFramesOverride(listNew);
                                    setSelected(updated);
                                    try {
                                      upsertAnalysis({ ...(updated as any), summary: updated.summary } as any);
                                      const list2 = activePageKey ? getAnalysesForPage(activePageKey) : getAllAnalyses();
                                      setAnalyses(list2);
                                    } catch {}
                                    setNoiseStatus(`Generated ${listNew.length} ${listNew.length===1?'noise frame':'noise frames'}`);
                                    setTimeout(()=> setNoiseStatus(null), 3000);
                                    try { if (noiseDetailsRef.current) { noiseDetailsRef.current.open = true; } } catch {}
                                    setNoiseFlash(true); setTimeout(()=> setNoiseFlash(false), 800);
                                  } catch (e: any) {
                                    setNoiseError(e?.message || 'Failed to generate noise frames');
                                  } finally { setNoiseAction(null); setNoiseLoading(false); }
                                }
                                async function handleReloadNoise() {
                                  setNoiseError(null); setNoiseStatus(null); setNoiseAction('reload'); setNoiseLoading(true);
                                  try {
                                    const tok = getAuthState().token;
                                    const headers: Record<string,string> = { };
                                    if (tok) headers['Authorization'] = `Bearer ${tok}`;
                                    let res: Response;
                                    if (numericId !== null) {
                                      const url = `${API_BASE}/api/noise/list?asset_id=${encodeURIComponent(String(numericId))}`;
                                      res = await fetch(url, { method: 'GET', headers });
                                    } else {
                                      const sp = (selected as any)?.raw?.asset?.stored_path;
                                      const sha = (selected as any)?.raw?.asset?.sha256;
                                      if (!sp) throw new Error('No asset reference for Noise reload');
                                      const qs = new URLSearchParams({ stored_path: String(sp) });
                                      if (sha) qs.set('sha256', String(sha));
                                      const url = `${API_BASE}/api/noise/list-by-path?${qs.toString()}`;
                                      res = await fetch(url, { method: 'GET', headers });
                                    }
                                    if (!res.ok) throw new Error(await res.text());
                                    const data = await res.json();
                                    const listNew = Array.isArray(data.noise_frames) ? data.noise_frames : [];
                                    const updated = { ...selected } as any;
                                    updated.summary = { ...(updated.summary || {}), noise_frames: listNew, noise: { frames: listNew } };
                                    if ((updated as any).raw && (updated as any).raw.summary) {
                                      (updated as any).raw.summary = { ...((updated as any).raw.summary || {}), noise_frames: listNew, noise: { frames: listNew } };
                                    }
                                    setNoiseSelectedIdx(0);
                                    setNoiseFramesOverride(listNew);
                                    setSelected(updated);
                                    try {
                                      upsertAnalysis({ ...(updated as any), summary: updated.summary } as any);
                                      const list2 = activePageKey ? getAnalysesForPage(activePageKey) : getAllAnalyses();
                                      setAnalyses(list2);
                                    } catch {}
                                    setNoiseStatus(listNew.length>0 ? `Loaded ${listNew.length} ${listNew.length===1?'noise frame':'noise frames'}` : 'No noise frames found');
                                    setTimeout(()=> setNoiseStatus(null), 3000);
                                    try { if (noiseDetailsRef.current) { noiseDetailsRef.current.open = true; } } catch {}
                                    if (listNew.length>0) { setNoiseFlash(true); setTimeout(()=> setNoiseFlash(false), 800); }
                                  } catch (e: any) {
                                    setNoiseError(e?.message || 'Failed to reload noise frames');
                                  } finally { setNoiseAction(null); setNoiseLoading(false); }
                                }
                                const resolve = (uri: string) => {
                                  try {
                                    if (!uri) return uri;
                                    if (/^(https?:)?\/\//i.test(uri)) return uri;
                                    if (/^data:/i.test(uri)) return uri;
                                    const base = String(API_BASE).replace(/\/$/, '');
                                    const path = String(uri).replace(/^\/+/, '');
                                    return `${base}/static/${path}`;
                                  } catch { return uri; }
                                };
                                const currentIdx = ((): number => {
                                  if (noiseSelectedIdx != null) return noiseSelectedIdx;
                                  if (!list.length) return 0;
                                  const withTimes = list.filter(x => typeof x.time_s === 'number' && Number.isFinite(x.time_s as any));
                                  if (noiseFollowVideo && withTimes.length) {
                                    let bestI = 0; let bestD = Infinity;
                                    list.forEach((f, i) => {
                                      const t: any = f.time_s;
                                      if (typeof t === 'number' && Number.isFinite(t)) {
                                        const d = Math.abs((t as number) - (videoTime || 0));
                                        if (d < bestD) { bestD = d; bestI = i; }
                                      }
                                    });
                                    return bestI;
                                  }
                                  return 0;
                                })();
                                const setAndMaybeSeek = (i: number) => {
                                  setNoiseSelectedIdx(i);
                                  const f = list[i];
                                  const t = (f && typeof f.time_s === 'number' && Number.isFinite(f.time_s as any)) ? (f.time_s as number) : null;
                                  if (t != null) jumpToTime(t, false);
                                };
                                const fmtPct = (v?: number) => typeof v==='number' ? `${Math.round(Math.max(0, Math.min(100, v*100)))}%` : '—';
                                return (
                                  <details className="overlay-block" ref={noiseDetailsRef as any}>
                                    <summary>Noise frames{list.length ? ` (${list.length})` : ''}</summary>
                                    <div className="ela-toolbar">
                                      <button className="btn" onClick={handleGenerateNoise} disabled={noiseLoading}>
                                        {noiseLoading && noiseAction==='generate' ? 'Generating…' : 'Generate noise frames'}
                                      </button>
                                      <button className="btn ghost" onClick={handleReloadNoise} disabled={noiseLoading}>
                                        {noiseLoading && noiseAction==='reload' ? 'Reloading…' : 'Reload'}
                                      </button>
                                      {noiseStatus ? <span className="status-pill success" role="status" aria-live="polite">{noiseStatus}</span> : null}
                                      {noiseError ? <span className="status-pill error" role="status" aria-live="assertive">{noiseError}</span> : null}
                                    </div>
                                    {list.length ? (
                                      <div className={`ela-grid${noiseFlash ? ' ela-updated-flash' : ''}`}>
                                        {(() => {
                                          const idx = Math.max(0, Math.min(list.length-1, currentIdx));
                                          const f = list[idx];
                                          const src = resolve(f.uri);
                                          const cap = (() => {
                                            const parts: string[] = [];
                                            if (typeof f.time_s === 'number' && Number.isFinite(f.time_s as any)) parts.push(`${(f.time_s as number).toFixed(2)}s`);
                                            if (typeof f.index === 'number' && Number.isFinite(f.index as any)) parts.push(`#${f.index}`);
                                            if (typeof f.noise_score === 'number') parts.push(`score ${(f.noise_score*100).toFixed(0)}%`);
                                            return parts.join(' • ') || `frame ${idx+1}`;
                                          })();
                                          const items = list.map((it, i) => {
                                            let leftPct = 0;
                                            if (typeof it.time_s === 'number' && Number.isFinite(it.time_s as any) && mediaDuration && mediaDuration > 0) {
                                              leftPct = Math.max(0, Math.min(100, Math.round(((it.time_s as number) / mediaDuration) * 100)));
                                            } else {
                                              leftPct = Math.round((i / Math.max(1, list.length - 1)) * 100);
                                            }
                                            return { i, leftPct, score: (typeof it.noise_score==='number'? it.noise_score : undefined) };
                                          });
                                          const headLeft = (() => {
                                            if (mediaDuration && mediaDuration > 0) {
                                              return Math.max(0, Math.min(100, Math.round(((Math.max(0, videoTime)) / mediaDuration) * 100)));
                                            }
                                            return Math.round(((idx) / Math.max(1, list.length - 1)) * 100);
                                          })();
                                          return (
                                            <div className="ela-viewer-block">
                                              <div className="ela-viewer-head">
                                                <div className="left"><span className="muted">Noise preview</span></div>
                                                <div className="right">
                                                  <label className="chk"><input type="checkbox" checked={noiseFollowVideo} onChange={(e)=>setNoiseFollowVideo(e.currentTarget.checked)} /> Follow video</label>
                                                  <button className="btn ghost" onClick={()=>{ if (idx>0) setAndMaybeSeek(idx-1); }} disabled={idx<=0}>Prev</button>
                                                  <button className="btn ghost" onClick={()=>{ if (idx<list.length-1) setAndMaybeSeek(idx+1); }} disabled={idx>=list.length-1}>Next</button>
                                                  {f.fft_uri ? <a className="btn ghost" href={resolve(f.fft_uri)} target="_blank" rel="noreferrer">FFT</a> : null}
                                                  <a className="btn ghost" href={src} target="_blank" rel="noreferrer">Open</a>
                                                </div>
                                              </div>
                                              <div className="ela-view">
                                                <img src={src} alt={`Noise ${cap}`} className="ela-img" />
                                                <div className="ela-cap">{cap}</div>
                                              </div>
                                              <div className="ela-timeline" role="region" aria-label="Noise timeline">
                                                <div className="ft-base" aria-hidden="true" />
                                                <div className={`ft-head ft-p${headLeft}`} aria-hidden="true" />
                                                {items.map(({i, leftPct, score}) => (
                                                  <button key={i} type="button" className={`ft-mark ft-p${leftPct} ${i===idx?'active':''} ${typeof score==='number' && score>=0.6 ? 'alert' : ''}`} onClick={()=> setAndMaybeSeek(i)} title={typeof score==='number'?`score ${(score*100).toFixed(0)}%`:undefined} />
                                                ))}
                                                <div className="ft-axis" aria-hidden="true">
                                                  <span>0s</span>
                                                  <span>{mediaDuration ? mediaDuration.toFixed(2)+'s' : ''}</span>
                                                </div>
                                              </div>
                                            </div>
                                          );
                                        })()}
                                        {list.map((f, i) => {
                                          const cap = ((): string => {
                                            const t = (typeof f.time_s === 'number' && Number.isFinite(f.time_s)) ? `${(f.time_s as number).toFixed(2)}s` : undefined;
                                            const idx2 = (typeof f.index === 'number' && Number.isFinite(f.index)) ? `#${f.index}` : undefined;
                                            const sc = (typeof f.noise_score === 'number') ? `score ${(f.noise_score*100).toFixed(0)}%` : undefined;
                                            return [t, idx2, sc].filter(Boolean).join(' • ') || `frame ${i+1}`;
                                          })();
                                          const src = resolve(f.uri);
                                          return (
                                            <a key={i} className={`ela-item${typeof f.noise_score==='number' && f.noise_score>=0.6 ? ' alert' : ''}`} href={src} target="_blank" rel="noreferrer" title={cap} onClick={(e)=>{ e.preventDefault(); setAndMaybeSeek(i); }}>
                                              <figure>
                                                <img src={src} loading="lazy" alt={cap || 'Noise frame'} />
                                                <figcaption className="cap">{cap}</figcaption>
                                              </figure>
                                            </a>
                                          );
                                        })}
                                      </div>
                                    ) : (
                                      <div className="overlay-content"><div className="muted">No noise frames available for this analysis.</div></div>
                                    )}
                                  </details>
                                );
                              })()}
                              {(() => {
                                // ELA frames viewer + grid
                                const list = elaFrames;
                                const assetIdUnknown = (selected as any)?.raw?.asset?.asset_id;
                                const numericId = ((): number | null => {
                                  if (assetIdUnknown === undefined || assetIdUnknown === null) return null;
                                  const n = Number(assetIdUnknown);
                                  return (Number.isFinite(n) && String(n) === String(assetIdUnknown).trim()) ? n : null;
                                })();
                                async function handleGenerateELA() {
                                  setElaError(null); setElaStatus(null); setElaAction('generate'); setElaLoading(true);
                                  try {
                                    const tok = getAuthState().token;
                                    const headers: Record<string,string> = { 'Content-Type': 'application/json' };
                                    if (tok) headers['Authorization'] = `Bearer ${tok}`;
                                    let res: Response;
                                    if (numericId !== null) {
                                      res = await fetch(`${API_BASE}/api/ela/generate`, {
                                        method: 'POST', headers,
                                        body: JSON.stringify({ asset_id: numericId, frames: 12, quality: 90, scale: 12.0 }),
                                      });
                                    } else {
                                      const sp = (selected as any)?.raw?.asset?.stored_path;
                                      const sha = (selected as any)?.raw?.asset?.sha256;
                                      if (!sp) throw new Error('No asset reference for ELA');
                                      res = await fetch(`${API_BASE}/api/ela/generate-by-path`, {
                                        method: 'POST', headers,
                                        body: JSON.stringify({ stored_path: sp, sha256: sha, frames: 12, quality: 90, scale: 12.0 }),
                                      });
                                    }
                                    if (!res.ok) throw new Error(await res.text());
                                    const data = await res.json();
                                    const listNew = Array.isArray(data.ela_frames) ? data.ela_frames : [];
                                    const updated = { ...selected } as any;
                                    const uris = (listNew || []).map((f:any)=>f && f.uri).filter(Boolean);
                                    updated.summary = { ...(updated.summary || {}), ela_frames: listNew, ela_uris: uris, ela: { frames: listNew, uris } };
                                    if ((updated as any).raw && (updated as any).raw.summary) {
                                      (updated as any).raw.summary = { ...((updated as any).raw.summary || {}), ela_frames: listNew, ela_uris: uris, ela: { frames: listNew, uris } };
                                    }
                                    setElaSelectedIdx(0);
                                    setElaFramesOverride(listNew);
                                    setSelected(updated);
                                    try {
                                      upsertAnalysis({ ...(updated as any), summary: updated.summary } as any);
                                      const list2 = activePageKey ? getAnalysesForPage(activePageKey) : getAllAnalyses();
                                      setAnalyses(list2);
                                    } catch {}
                                    setElaStatus(`Generated ${listNew.length} ${listNew.length===1?'ELA frame':'ELA frames'}`);
                                    setTimeout(()=> setElaStatus(null), 3000);
                                    try { if (elaDetailsRef.current) { elaDetailsRef.current.open = true; } } catch {}
                                    setElaFlash(true); setTimeout(()=> setElaFlash(false), 800);
                                  } catch (e: any) {
                                    setElaError(e?.message || 'Failed to generate ELA frames');
                                  } finally { setElaAction(null); setElaLoading(false); }
                                }
                                async function handleReloadELA() {
                                  setElaError(null); setElaStatus(null); setElaAction('reload'); setElaLoading(true);
                                  try {
                                    const tok = getAuthState().token;
                                    const headers: Record<string,string> = { };
                                    if (tok) headers['Authorization'] = `Bearer ${tok}`;
                                    let res: Response;
                                    if (numericId !== null) {
                                      const url = `${API_BASE}/api/ela/list?asset_id=${encodeURIComponent(String(numericId))}`;
                                      res = await fetch(url, { method: 'GET', headers });
                                    } else {
                                      const sp = (selected as any)?.raw?.asset?.stored_path;
                                      const sha = (selected as any)?.raw?.asset?.sha256;
                                      if (!sp) throw new Error('No asset reference for ELA reload');
                                      const qs = new URLSearchParams({ stored_path: String(sp) });
                                      if (sha) qs.set('sha256', String(sha));
                                      const url = `${API_BASE}/api/ela/list-by-path?${qs.toString()}`;
                                      res = await fetch(url, { method: 'GET', headers });
                                    }
                                    if (!res.ok) throw new Error(await res.text());
                                    const data = await res.json();
                                    const listNew = Array.isArray(data.ela_frames) ? data.ela_frames : [];
                                    const updated = { ...selected } as any;
                                    const uris = (listNew || []).map((f:any)=>f && f.uri).filter(Boolean);
                                    updated.summary = { ...(updated.summary || {}), ela_frames: listNew, ela_uris: uris, ela: { frames: listNew, uris } };
                                    if ((updated as any).raw && (updated as any).raw.summary) {
                                      (updated as any).raw.summary = { ...((updated as any).raw.summary || {}), ela_frames: listNew, ela_uris: uris, ela: { frames: listNew, uris } };
                                    }
                                    setElaSelectedIdx(0);
                                    setElaFramesOverride(listNew);
                                    setSelected(updated);
                                    try {
                                      upsertAnalysis({ ...(updated as any), summary: updated.summary } as any);
                                      const list2 = activePageKey ? getAnalysesForPage(activePageKey) : getAllAnalyses();
                                      setAnalyses(list2);
                                    } catch {}
                                    setElaStatus(listNew.length>0 ? `Loaded ${listNew.length} ${listNew.length===1?'ELA frame':'ELA frames'}` : 'No ELA frames found');
                                    setTimeout(()=> setElaStatus(null), 3000);
                                    try { if (elaDetailsRef.current) { elaDetailsRef.current.open = true; } } catch {}
                                    if (listNew.length>0) { setElaFlash(true); setTimeout(()=> setElaFlash(false), 800); }
                                  } catch (e: any) {
                                    setElaError(e?.message || 'Failed to reload ELA frames');
                                  } finally { setElaAction(null); setElaLoading(false); }
                                }
                                const resolve = (uri: string) => {
                                  try {
                                    if (!uri) return uri;
                                    if (/^(https?:)?\/\//i.test(uri)) return uri; // absolute or protocol-relative
                                    if (/^data:/i.test(uri)) return uri; // data URL
                                    const base = String(API_BASE).replace(/\/$/, '');
                                    const path = String(uri).replace(/^\/+/, '');
                                    return `${base}/static/${path}`;
                                  } catch { return uri; }
                                };
                                const currentIdx = ((): number => {
                                  if (elaSelectedIdx != null) return elaSelectedIdx;
                                  if (!list.length) return 0;
                                  const withTimes = list.filter(x => typeof x.time_s === 'number' && Number.isFinite(x.time_s as any));
                                  if (elaFollowVideo && withTimes.length) {
                                    let bestI = 0; let bestD = Infinity;
                                    list.forEach((f, i) => {
                                      const t: any = f.time_s;
                                      if (typeof t === 'number' && Number.isFinite(t)) {
                                        const d = Math.abs((t as number) - (videoTime || 0));
                                        if (d < bestD) { bestD = d; bestI = i; }
                                      }
                                    });
                                    return bestI;
                                  }
                                  return 0;
                                })();
                                const setAndMaybeSeek = (i: number) => {
                                  setElaSelectedIdx(i);
                                  const f = list[i];
                                  const t = (f && typeof f.time_s === 'number' && Number.isFinite(f.time_s as any)) ? (f.time_s as number) : null;
                                  if (t != null) jumpToTime(t, false);
                                };
                                return (
                                  <details className="overlay-block" ref={elaDetailsRef as any}>
                                    <summary>ELA frames{list.length ? ` (${list.length})` : ''}</summary>
                                    <div className="ela-toolbar">
                                      <button className="btn" onClick={handleGenerateELA} disabled={elaLoading}>
                                        {elaLoading && elaAction==='generate' ? 'Generating…' : 'Generate ELA frames'}
                                      </button>
                                      <button className="btn ghost" onClick={handleReloadELA} disabled={elaLoading}>
                                        {elaLoading && elaAction==='reload' ? 'Reloading…' : 'Reload'}
                                      </button>
                                      {elaStatus ? <span className="status-pill success" role="status" aria-live="polite">{elaStatus}</span> : null}
                                      {elaError ? <span className="status-pill error" role="status" aria-live="assertive">{elaError}</span> : null}
                                    </div>
                                    {list.length ? (
                                      <div className={`ela-grid${elaFlash ? ' ela-updated-flash' : ''}`}>
                                        {(() => {
                                          const idx = Math.max(0, Math.min(list.length-1, currentIdx));
                                          const f = list[idx];
                                          const src = resolve(f.uri);
                                          const cap = (() => {
                                            const parts: string[] = [];
                                            if (typeof f.time_s === 'number' && Number.isFinite(f.time_s as any)) parts.push(`${(f.time_s as number).toFixed(2)}s`);
                                            if (typeof f.index === 'number' && Number.isFinite(f.index as any)) parts.push(`#${f.index}`);
                                            return parts.join(' • ') || `frame ${idx+1}`;
                                          })();
                                          const items = list.map((it, i) => {
                                            let leftPct = 0;
                                            if (typeof it.time_s === 'number' && Number.isFinite(it.time_s as any) && mediaDuration && mediaDuration > 0) {
                                              leftPct = Math.max(0, Math.min(100, Math.round(((it.time_s as number) / mediaDuration) * 100)));
                                            } else {
                                              leftPct = Math.round((i / Math.max(1, list.length - 1)) * 100);
                                            }
                                            return { i, leftPct };
                                          });
                                          const headLeft = (() => {
                                            if (mediaDuration && mediaDuration > 0) {
                                              return Math.max(0, Math.min(100, Math.round(((Math.max(0, videoTime)) / mediaDuration) * 100)));
                                            }
                                            return Math.round(((idx) / Math.max(1, list.length - 1)) * 100);
                                          })();
                                          return (
                                            <div className="ela-viewer-block">
                                              <div className="ela-viewer-head">
                                                <div className="left"><span className="muted">ELA preview</span></div>
                                                <div className="right">
                                                  <label className="chk"><input type="checkbox" checked={elaFollowVideo} onChange={(e)=>setElaFollowVideo(e.currentTarget.checked)} /> Follow video</label>
                                                  <button className="btn ghost" onClick={()=>{ if (idx>0) setAndMaybeSeek(idx-1); }} disabled={idx<=0}>Prev</button>
                                                  <button className="btn ghost" onClick={()=>{ if (idx<list.length-1) setAndMaybeSeek(idx+1); }} disabled={idx>=list.length-1}>Next</button>
                                                  <a className="btn ghost" href={src} target="_blank" rel="noreferrer">Open</a>
                                                </div>
                                              </div>
                                              <div className="ela-view">
                                                <img src={src} alt={`ELA ${cap}`} className="ela-img" />
                                                <div className="ela-cap">{cap}</div>
                                              </div>
                                              <div className="ela-timeline" role="region" aria-label="ELA timeline">
                                                <div className="ft-base" aria-hidden="true" />
                                                <div className={`ft-head ft-p${headLeft}`} aria-hidden="true" />
                                                {items.map(({i, leftPct}) => (
                                                  <button key={i} type="button" className={`ft-mark ft-p${leftPct} ${i===idx?'active':''}`} onClick={()=> setAndMaybeSeek(i)} />
                                                ))}
                                                <div className="ft-axis" aria-hidden="true">
                                                  <span>0s</span>
                                                  <span>{mediaDuration ? mediaDuration.toFixed(2)+'s' : ''}</span>
                                                </div>
                                              </div>
                                            </div>
                                          );
                                        })()}
                                        {list.map((f, i) => {
                                          const cap = ((): string => {
                                            const t = (typeof f.time_s === 'number' && Number.isFinite(f.time_s)) ? `${(f.time_s as number).toFixed(2)}s` : undefined;
                                            const idx2 = (typeof f.index === 'number' && Number.isFinite(f.index)) ? `#${f.index}` : undefined;
                                            return t ? (idx2 ? `${t} • ${idx2}` : t) : (idx2 || `frame ${i+1}`);
                                          })();
                                          const src = resolve(f.uri);
                                          return (
                                            <a key={i} className="ela-item" href={src} target="_blank" rel="noreferrer" title={cap} onClick={(e)=>{ e.preventDefault(); setAndMaybeSeek(i); }}>
                                              <figure>
                                                <img src={src} loading="lazy" alt={cap || 'ELA frame'} />
                                                <figcaption className="cap">{cap}</figcaption>
                                              </figure>
                                            </a>
                                          );
                                        })}
                                      </div>
                                    ) : (
                                      <div className="overlay-content"><div className="muted">No ELA frames available for this analysis.</div></div>
                                    )}
                                  </details>
                                );
                              })()}
                              {(() => {
                                // LBP frames viewer + grid
                                const list = lbpFrames;
                                const assetIdUnknown = (selected as any)?.raw?.asset?.asset_id;
                                const numericId = ((): number | null => {
                                  if (assetIdUnknown === undefined || assetIdUnknown === null) return null;
                                  const n = Number(assetIdUnknown);
                                  return (Number.isFinite(n) && String(n) === String(assetIdUnknown).trim()) ? n : null;
                                })();
                                async function handleGenerateLBP() {
                                  setLbpError(null); setLbpStatus(null); setLbpAction('generate'); setLbpLoading(true);
                                  try {
                                    const tok = getAuthState().token;
                                    const headers: Record<string,string> = { 'Content-Type': 'application/json' };
                                    if (tok) headers['Authorization'] = `Bearer ${tok}`;
                                    let res: Response;
                                    if (numericId !== null) {
                                      res = await fetch(`${API_BASE}/api/lbp/generate`, {
                                        method: 'POST', headers,
                                        body: JSON.stringify({ asset_id: numericId, frames: 12, radius: 1 }),
                                      });
                                    } else {
                                      const sp = (selected as any)?.raw?.asset?.stored_path;
                                      const sha = (selected as any)?.raw?.asset?.sha256;
                                      if (!sp) throw new Error('No asset reference for LBP');
                                      res = await fetch(`${API_BASE}/api/lbp/generate-by-path`, {
                                        method: 'POST', headers,
                                        body: JSON.stringify({ stored_path: sp, sha256: sha, frames: 12, radius: 1 }),
                                      });
                                    }
                                    if (!res.ok) throw new Error(await res.text());
                                    const data = await res.json();
                                    const listNew = Array.isArray(data.lbp_frames) ? data.lbp_frames : [];
                                    const updated = { ...selected } as any;
                                    const uris = (listNew || []).map((f:any)=>f && f.uri).filter(Boolean);
                                    updated.summary = { ...(updated.summary || {}), lbp_frames: listNew, lbp_uris: uris, lbp: { frames: listNew, uris } };
                                    if ((updated as any).raw && (updated as any).raw.summary) {
                                      (updated as any).raw.summary = { ...((updated as any).raw.summary || {}), lbp_frames: listNew, lbp_uris: uris, lbp: { frames: listNew, uris } };
                                    }
                                    setLbpSelectedIdx(0);
                                    setLbpFramesOverride(listNew);
                                    setSelected(updated);
                                    try {
                                      upsertAnalysis({ ...(updated as any), summary: updated.summary } as any);
                                      const list2 = activePageKey ? getAnalysesForPage(activePageKey) : getAllAnalyses();
                                      setAnalyses(list2);
                                    } catch {}
                                    setLbpStatus(`Generated ${listNew.length} ${listNew.length===1?'LBP frame':'LBP frames'}`);
                                    setTimeout(()=> setLbpStatus(null), 3000);
                                    try { if (lbpDetailsRef.current) { lbpDetailsRef.current.open = true; } } catch {}
                                    setLbpFlash(true); setTimeout(()=> setLbpFlash(false), 800);
                                  } catch (e: any) {
                                    setLbpError(e?.message || 'Failed to generate LBP frames');
                                  } finally { setLbpAction(null); setLbpLoading(false); }
                                }
                                async function handleReloadLBP() {
                                  setLbpError(null); setLbpStatus(null); setLbpAction('reload'); setLbpLoading(true);
                                  try {
                                    const tok = getAuthState().token;
                                    const headers: Record<string,string> = { };
                                    if (tok) headers['Authorization'] = `Bearer ${tok}`;
                                    let res: Response;
                                    if (numericId !== null) {
                                      const url = `${API_BASE}/api/lbp/list?asset_id=${encodeURIComponent(String(numericId))}`;
                                      res = await fetch(url, { method: 'GET', headers });
                                    } else {
                                      const sp = (selected as any)?.raw?.asset?.stored_path;
                                      const sha = (selected as any)?.raw?.asset?.sha256;
                                      if (!sp) throw new Error('No asset reference for LBP reload');
                                      const qs = new URLSearchParams({ stored_path: String(sp) });
                                      if (sha) qs.set('sha256', String(sha));
                                      const url = `${API_BASE}/api/lbp/list-by-path?${qs.toString()}`;
                                      res = await fetch(url, { method: 'GET', headers });
                                    }
                                    if (!res.ok) throw new Error(await res.text());
                                    const data = await res.json();
                                    const listNew = Array.isArray(data.lbp_frames) ? data.lbp_frames : [];
                                    const updated = { ...selected } as any;
                                    const uris = (listNew || []).map((f:any)=>f && f.uri).filter(Boolean);
                                    updated.summary = { ...(updated.summary || {}), lbp_frames: listNew, lbp_uris: uris, lbp: { frames: listNew, uris } };
                                    if ((updated as any).raw && (updated as any).raw.summary) {
                                      (updated as any).raw.summary = { ...((updated as any).raw.summary || {}), lbp_frames: listNew, lbp_uris: uris, lbp: { frames: listNew, uris } };
                                    }
                                    setLbpSelectedIdx(0);
                                    setLbpFramesOverride(listNew);
                                    setSelected(updated);
                                    try {
                                      upsertAnalysis({ ...(updated as any), summary: updated.summary } as any);
                                      const list2 = activePageKey ? getAnalysesForPage(activePageKey) : getAllAnalyses();
                                      setAnalyses(list2);
                                    } catch {}
                                    setLbpStatus(listNew.length>0 ? `Loaded ${listNew.length} ${listNew.length===1?'LBP frame':'LBP frames'}` : 'No LBP frames found');
                                    setTimeout(()=> setLbpStatus(null), 3000);
                                    try { if (lbpDetailsRef.current) { lbpDetailsRef.current.open = true; } } catch {}
                                    if (listNew.length>0) { setLbpFlash(true); setTimeout(()=> setLbpFlash(false), 800); }
                                  } catch (e: any) {
                                    setLbpError(e?.message || 'Failed to reload LBP frames');
                                  } finally { setLbpAction(null); setLbpLoading(false); }
                                }
                                const resolve = (uri: string) => {
                                  try {
                                    if (!uri) return uri;
                                    if (/^(https?:)?\/\//i.test(uri)) return uri;
                                    if (/^data:/i.test(uri)) return uri;
                                    const base = String(API_BASE).replace(/\/$/, '');
                                    const path = String(uri).replace(/^\/+/, '');
                                    return `${base}/static/${path}`;
                                  } catch { return uri; }
                                };
                                const currentIdx = ((): number => {
                                  if (lbpSelectedIdx != null) return lbpSelectedIdx;
                                  if (!list.length) return 0;
                                  const withTimes = list.filter(x => typeof x.time_s === 'number' && Number.isFinite(x.time_s as any));
                                  if (lbpFollowVideo && withTimes.length) {
                                    let bestI = 0; let bestD = Infinity;
                                    list.forEach((f, i) => {
                                      const t: any = f.time_s;
                                      if (typeof t === 'number' && Number.isFinite(t)) {
                                        const d = Math.abs((t as number) - (videoTime || 0));
                                        if (d < bestD) { bestD = d; bestI = i; }
                                      }
                                    });
                                    return bestI;
                                  }
                                  return 0;
                                })();
                                const setAndMaybeSeek = (i: number) => {
                                  setLbpSelectedIdx(i);
                                  const f = list[i];
                                  const t = (f && typeof f.time_s === 'number' && Number.isFinite(f.time_s as any)) ? (f.time_s as number) : null;
                                  if (t != null) jumpToTime(t, false);
                                };
                                return (
                                  <details className="overlay-block" ref={lbpDetailsRef as any}>
                                    <summary>LBP frames{list.length ? ` (${list.length})` : ''}</summary>
                                    <div className="ela-toolbar">
                                      <button className="btn" onClick={handleGenerateLBP} disabled={lbpLoading}>
                                        {lbpLoading && lbpAction==='generate' ? 'Generating…' : 'Generate LBP frames'}
                                      </button>
                                      <button className="btn ghost" onClick={handleReloadLBP} disabled={lbpLoading}>
                                        {lbpLoading && lbpAction==='reload' ? 'Reloading…' : 'Reload'}
                                      </button>
                                      {lbpStatus ? <span className="status-pill success" role="status" aria-live="polite">{lbpStatus}</span> : null}
                                      {lbpError ? <span className="status-pill error" role="status" aria-live="assertive">{lbpError}</span> : null}
                                    </div>
                                    {list.length ? (
                                      <div className={`ela-grid${lbpFlash ? ' ela-updated-flash' : ''}`}>
                                        {(() => {
                                          const idx = Math.max(0, Math.min(list.length-1, currentIdx));
                                          const f = list[idx];
                                          const src = resolve(f.uri);
                                          const cap = (() => {
                                            const parts: string[] = [];
                                            if (typeof f.time_s === 'number' && Number.isFinite(f.time_s as any)) parts.push(`${(f.time_s as number).toFixed(2)}s`);
                                            if (typeof f.index === 'number' && Number.isFinite(f.index as any)) parts.push(`#${f.index}`);
                                            return parts.join(' • ') || `frame ${idx+1}`;
                                          })();
                                          const items = list.map((it, i) => {
                                            let leftPct = 0;
                                            if (typeof it.time_s === 'number' && Number.isFinite(it.time_s as any) && mediaDuration && mediaDuration > 0) {
                                              leftPct = Math.max(0, Math.min(100, Math.round(((it.time_s as number) / mediaDuration) * 100)));
                                            } else {
                                              leftPct = Math.round((i / Math.max(1, list.length - 1)) * 100);
                                            }
                                            return { i, leftPct };
                                          });
                                          const headLeft = (() => {
                                            if (mediaDuration && mediaDuration > 0) {
                                              return Math.max(0, Math.min(100, Math.round(((Math.max(0, videoTime)) / mediaDuration) * 100)));
                                            }
                                            return Math.round(((idx) / Math.max(1, list.length - 1)) * 100);
                                          })();
                                          return (
                                            <div className="ela-viewer-block">
                                              <div className="ela-viewer-head">
                                                <div className="left"><span className="muted">LBP preview</span></div>
                                                <div className="right">
                                                  <label className="chk"><input type="checkbox" checked={lbpFollowVideo} onChange={(e)=>setLbpFollowVideo(e.currentTarget.checked)} /> Follow video</label>
                                                  <button className="btn ghost" onClick={()=>{ if (idx>0) setAndMaybeSeek(idx-1); }} disabled={idx<=0}>Prev</button>
                                                  <button className="btn ghost" onClick={()=>{ if (idx<list.length-1) setAndMaybeSeek(idx+1); }} disabled={idx>=list.length-1}>Next</button>
                                                  <a className="btn ghost" href={src} target="_blank" rel="noreferrer">Open</a>
                                                </div>
                                              </div>
                                              <div className="ela-view">
                                                <img src={src} alt={`LBP ${cap}`} className="ela-img" />
                                                <div className="ela-cap">{cap}</div>
                                              </div>
                                              <div className="ela-timeline" role="region" aria-label="LBP timeline">
                                                <div className="ft-base" aria-hidden="true" />
                                                <div className={`ft-head ft-p${headLeft}`} aria-hidden="true" />
                                                {items.map(({i, leftPct}) => (
                                                  <button key={i} type="button" className={`ft-mark ft-p${leftPct} ${i===idx?'active':''}`} onClick={()=> setAndMaybeSeek(i)} />
                                                ))}
                                                <div className="ft-axis" aria-hidden="true">
                                                  <span>0s</span>
                                                  <span>{mediaDuration ? mediaDuration.toFixed(2)+'s' : ''}</span>
                                                </div>
                                              </div>
                                            </div>
                                          );
                                        })()}
                                        {list.map((f, i) => {
                                          const cap = ((): string => {
                                            const t = (typeof f.time_s === 'number' && Number.isFinite(f.time_s)) ? `${(f.time_s as number).toFixed(2)}s` : undefined;
                                            const idx2 = (typeof f.index === 'number' && Number.isFinite(f.index)) ? `#${f.index}` : undefined;
                                            return t ? (idx2 ? `${t} • ${idx2}` : t) : (idx2 || `frame ${i+1}`);
                                          })();
                                          const src = resolve(f.uri);
                                          return (
                                            <a key={i} className="ela-item" href={src} target="_blank" rel="noreferrer" title={cap} onClick={(e)=>{ e.preventDefault(); setAndMaybeSeek(i); }}>
                                              <figure>
                                                <img src={src} loading="lazy" alt={cap || 'LBP frame'} />
                                                <figcaption className="cap">{cap}</figcaption>
                                              </figure>
                                            </a>
                                          );
                                        })}
                                      </div>
                                    ) : (
                                      <div className="overlay-content"><div className="muted">No LBP frames available for this analysis.</div></div>
                                    )}
                                  </details>
                                );
                              })()}
                              {selected.raw && (
                                <details className="raw-block">
                                  <summary>Raw JSON</summary>
                                  <pre>{JSON.stringify(selected.raw, null, 2)}</pre>
                                </details>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </React.Fragment>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
      {(showDeleteConfirm || pendingDeleteIds) && (
        <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="del-title" onMouseDown={(e)=> { if (e.target===e.currentTarget) cancelBulkDelete(); }}>
          <div className="modal" role="document">
            {(() => {
              const ids = pendingDeleteIds || checkedIds;
              const count = ids.length;
              return (
                <>
                  <h3 id="del-title">Delete {count} analysis{count>1?'es':''}?</h3>
                  <p className="modal-text">This action will permanently remove the selected item{count>1?'s':''} from this report. It cannot be undone.</p>
                </>
              );
            })()}
            <div className="modal-actions">
              <button className="btn" onClick={cancelBulkDelete}>Cancel</button>
              <button className="btn danger" onClick={() => performDelete(pendingDeleteIds || checkedIds)} autoFocus>Delete</button>
            </div>
          </div>
        </div>
      )}
      {undoBuffer && (
        <div className="toast" role="status" aria-live="polite">
          <span>Deleted {undoBuffer.length} analysis{undoBuffer.length>1?'es':''}.</span>
          <button className="btn link" onClick={undoDelete}>Undo</button>
        </div>
      )}
    </div>
  );
};

export default Reports;
