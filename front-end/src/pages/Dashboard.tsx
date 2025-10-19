import React, { useEffect, useRef, useState } from "react";
import "./PageStyles.css";
import { Asset, listAssets, analyzeAsset, getUploadUrl, confirmUpload, getPlaybackUrl } from '../utils/assetsApi';
import { getAuthState, subscribe } from '../state/authStore';
import { upsertAnalysis } from '../state/analysisStore';

const API_BASE = (import.meta as any).env?.VITE_API_BASE || 'http://localhost:8000';

const DashboardVideo: React.FC<{ assetId: number }> = ({ assetId }) => {
  const [src, setSrc] = useState<string>('');
  useEffect(() => {
    let canceled = false;
    (async () => {
      try {
        const url = await getPlaybackUrl(assetId);
        if (!canceled) setSrc(url + '#t=0.1');
      } catch {
        if (!canceled) setSrc('');
      }
    })();
    return () => { canceled = true; };
  }, [assetId]);
  if (!src) return <div className="placeholder">video</div>;
  return <video src={src} preload="metadata" playsInline muted controls={false} />;
};

const Dashboard: React.FC = () => {
  const [auth, setAuth] = useState(getAuthState());
  const [assets, setAssets] = useState<Asset[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState<string>(() => {
    try { return localStorage.getItem('bl_lastModel') || 'stub'; } catch { return 'stub'; }
  });
  const fileRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const unsub = subscribe(setAuth);
    return () => { unsub(); };
  }, []);

  async function refresh() {
    if (!auth.token) { setAssets([]); return; }
    setLoading(true); setError(null);
    try {
      const rows = await listAssets();
      setAssets(rows);
    } catch (e: any) {
      setError(e.message || 'Failed to load assets');
    } finally { setLoading(false); }
  }

  useEffect(() => { refresh(); }, [auth.token]);

  function selectFile() { fileRef.current?.click(); }
  async function onPicked(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      setLoading(true); setError(null);
      // Presigned upload flow
      const { key, upload_url, headers } = await getUploadUrl(f.name, f.type || 'application/octet-stream');
      const h = new Headers(headers || {});
      if (!h.get('Content-Type') && f.type) h.set('Content-Type', f.type);
      const putRes = await fetch(upload_url, { method: 'PUT', headers: h, body: f });
      if (!putRes.ok) throw new Error(`Upload failed: ${putRes.status}`);
      await confirmUpload(key, f.name, f.type);
      await refresh();
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setLoading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  }

  async function runAnalysis(a: Asset) {
    try {
      setLoading(true); setError(null);
      const res = await analyzeAsset(a.id, model);
      // Persist summary so it appears in Reports
      const name = a.original_name || (a.stored_path || 'asset');
      const ns = auth.user ? `u_${auth.user.username}` : 'guest';
      const pageKey = localStorage.getItem(`bl_lastFilePage_${ns}`) || 'file1';
      upsertAnalysis({
        id: `asset:${a.id}`,
        pageKey,
        fileName: name,
        mime: a.mime || undefined,
        analyzedAt: Date.now(),
        summary: res.summary || {},
        raw: res,
      });
      // Navigate to Reports page to view details
      try { window.dispatchEvent(new CustomEvent('bl:navigate', { detail: { page: 'reports' } })); } catch {}
    } catch (e: any) {
      setError(e.message || 'Analysis failed');
    } finally { setLoading(false); }
  }

  return (
    <div className="page-content">
      <div className="head-row">
        <h1>My uploads</h1>
        <div className="head-actions">
          <label>
            <span className="visually-hidden">Model</span>
            <select value={model} onChange={(e)=>{ setModel(e.target.value); try{localStorage.setItem('bl_lastModel', e.target.value);}catch{}}}>
              <option value="stub">Stub (fast)</option>
              <option value="cm">Copy-move</option>
              <option value="lbp">LBP (faces)</option>
              <option value="fusion">Deep (Xception+TimeSformer)</option>
            </select>
          </label>
          <button onClick={selectFile} disabled={!auth.token}>Upload</button>
          <input ref={fileRef} type="file" className="visually-hidden" aria-label="Upload file" onChange={onPicked} />
          <button onClick={refresh} disabled={!auth.token || loading}>Refresh</button>
        </div>
      </div>
      {!auth.token ? (
        <div className="notice">Sign in from the top-right profile menu to upload and see your assets.</div>
      ) : null}
      {error && <div className="error-banner">{error}</div>}
      {loading && <div className="loading">Loading…</div>}
      <div className="assets-grid">
        {assets.map(a => {
          const created = new Date((a.created_at || 0) * 1000).toLocaleString();
          const isVideo = (a.mime || '').startsWith('video/');
          return (
            <div key={a.id} className="asset-card">
              <div className="asset-media">
                {isVideo ? (
                  <DashboardVideo assetId={a.id} />
                ) : (
                  <div className="placeholder">{a.mime || 'file'}</div>
                )}
              </div>
              <div className="asset-meta">
                <div className="name" title={a.original_name}>{a.original_name}</div>
                <div className="sub">{created} · {a.size_bytes ? `${(a.size_bytes/1_000_000).toFixed(2)} MB` : ''}</div>
              </div>
              <div className="asset-actions">
                <button onClick={()=>runAnalysis(a)}>Analyze</button>
              </div>
            </div>
          );
        })}
        {assets.length === 0 && auth.token && !loading && (
          <div className="empty">No uploads yet. Use the Upload button to add media.</div>
        )}
      </div>
      <style>{`
        .head-row { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px; }
        .head-actions { display:flex; align-items:center; gap:8px; }
        .visually-hidden { position:absolute !important; height:1px; width:1px; overflow:hidden; clip:rect(1px, 1px, 1px, 1px); white-space:nowrap; }
        .assets-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap:12px; }
        .asset-card { background: var(--panel-bg, #fff); border: 1px solid var(--border, #e5e7eb); border-radius: 8px; overflow: hidden; display:flex; flex-direction:column; }
        .asset-media { aspect-ratio: 16/9; background: #f3f4f6; display:flex; align-items:center; justify-content:center; }
        .asset-media video { width:100%; height:100%; object-fit:cover; display:block; }
        .asset-meta { padding:8px 10px; }
        .asset-meta .name { font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .asset-meta .sub { color:#6b7280; font-size:12px; }
        .asset-actions { padding:8px 10px 12px; display:flex; gap:8px; }
        .notice { margin-top:8px; color:#6b7280; }
        .error-banner { background:#fee2e2; color:#991b1b; border:1px solid #fecaca; padding:8px 10px; border-radius:6px; margin:8px 0; }
        .loading { color:#6b7280; margin:8px 0; }
        .empty { color:#6b7280; margin-top:24px; }
        @media (prefers-color-scheme: dark) {
          .asset-card { background: #0b1720; border-color: #1f2a33; }
          .asset-media { background: #0f1b24; }
          .asset-meta .sub, .notice, .loading, .empty { color: #9aa6af; }
          .error-banner { background:#3f1d1d; color:#fca5a5; border-color:#7f1d1d; }
        }
      `}</style>
    </div>
  );
};

export default Dashboard;
