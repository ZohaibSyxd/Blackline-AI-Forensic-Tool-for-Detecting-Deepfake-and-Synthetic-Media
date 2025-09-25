import React, { useEffect, useState, useRef } from "react";
import "./Reports.css";
import { getAnalysesForPage, getAllAnalyses, StoredAnalysisSummary, deleteAnalyses, upsertAnalysis } from '../state/analysisStore';

const labelFor = (filePage?: string) => {
  if (!filePage) return "FILE ANALYSIS";
  if (filePage === "file1") return "FILE ANALYSIS 1";
  if (filePage === "file2") return "FILE ANALYSIS 2";
  if (filePage === "file3") return "FILE ANALYSIS 3";
  if (filePage === "file$") return "FILE ANALYSIS $";
  return "FILE ANALYSIS";
};

const Reports: React.FC<{ filePage?: string }> = ({ filePage }) => {
  const label = labelFor(filePage);
  const [analyses, setAnalyses] = useState<StoredAnalysisSummary[]>([]);
  const [sort, setSort] = useState<{ key: string; dir: 'asc' | 'desc' }>({ key: 'analyzedAt', dir: 'desc' });
  const [selected, setSelected] = useState<StoredAnalysisSummary | null>(null); // single row detail panel
  const [checkedIds, setCheckedIds] = useState<string[]>([]); // multi-select for bulk actions
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [undoBuffer, setUndoBuffer] = useState<StoredAnalysisSummary[] | null>(null);
  const [undoTimer, setUndoTimer] = useState<number | null>(null);
  const lastCheckIndexRef = useRef<number | null>(null);
  const analysisRegionRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const list = filePage ? getAnalysesForPage(filePage) : getAllAnalyses();
    // default sort (analyzedAt desc) applied via sort state effect below
    setAnalyses(list);
    setCheckedIds([]); // reset selection on page change
  }, [filePage]);

  // Close detail on click outside or Escape
  useEffect(() => {
    if (!selected) return;
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') setSelected(null); }
    function onClick(e: MouseEvent) {
      if (!analysisRegionRef.current) return;
      const region = analysisRegionRef.current;
      // If click is inside the detail or table rows, ignore; else close
      if (region.contains(e.target as Node)) {
        const detail = region.querySelector('.analysis-detail');
        if (detail && detail.contains(e.target as Node)) return; // allow internal interaction
        // If clicked a row, keep (row handles its own selection)
        const row = (e.target as HTMLElement).closest('.analysis-table .tr');
        if (row) return;
      }
      setSelected(null);
    }
    document.addEventListener('keydown', onKey);
    document.addEventListener('mousedown', onClick);
    return () => { document.removeEventListener('keydown', onKey); document.removeEventListener('mousedown', onClick); };
  }, [selected]);

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
  function performBulkDelete() {
    // capture items for undo
    const toDelete = analyses.filter(a => checkedIds.includes(a.id));
    deleteAnalyses(checkedIds, filePage);
    // refresh list
    const list = filePage ? getAnalysesForPage(filePage) : getAllAnalyses();
    list.sort((a,b)=> b.analyzedAt - a.analyzedAt);
    setAnalyses(list);
    setCheckedIds([]);
    if (selected && checkedIds.includes(selected.id)) setSelected(null);
    setShowDeleteConfirm(false);
    // setup undo buffer & timer
    if (undoTimer) window.clearTimeout(undoTimer);
    setUndoBuffer(toDelete);
    const t = window.setTimeout(()=> { setUndoBuffer(null); setUndoTimer(null); }, 7000);
    setUndoTimer(t);
  }
  function cancelBulkDelete() { setShowDeleteConfirm(false); }
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

  const aggregate = React.useMemo(() => {
    if (!analyses.length) return null;
    const total = analyses.length;
    const avgLikelihood = analyses.reduce((s,a)=> s + (a.summary.deepfake_likelihood || 0),0)/total;
    const maxLikelihood = analyses.reduce((m,a)=> Math.max(m, a.summary.deepfake_likelihood||0),0);
    const suspicious = analyses.filter(a => (a.summary.deepfake_likelihood||0) > 0.5).length;
    return { total, avgLikelihood, maxLikelihood, suspicious };
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
          case 'resolution': {
            const aR = (a.summary.width||0) * (a.summary.height||0);
            const bR = (b.summary.width||0) * (b.summary.height||0);
            res = cmpNums(aR,bR); break; }
          case 'duration': res = cmpNums(a.summary.duration_s||0, b.summary.duration_s||0); break;
          case 'codec': res = (a.summary.codec||'').localeCompare(b.summary.codec||''); break;
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

  return (
    <div className="reports-page">
      <div className="reports-inner">
        <div className="reports-header">
          <div>
            <h2>Reports</h2>
          </div>
        </div>

        {aggregate && (
          <div className="report-stats">
            <div className="stat"><div className="label">Analyses</div><div className="value">{aggregate.total}</div></div>
            <div className="stat"><div className="label">Avg Likelihood</div><div className="value">{(aggregate.avgLikelihood*100).toFixed(1)}%</div></div>
            <div className="stat"><div className="label">Max Likelihood</div><div className="value">{(aggregate.maxLikelihood*100).toFixed(1)}%</div></div>
            <div className="stat"><div className="label">Flagged (&gt;50%)</div><div className="value">{aggregate.suspicious}</div></div>
          </div>
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
                  <div className={`th sortable ${sort.key==='file' ? 'sorted '+sort.dir : ''}`} role="columnheader" aria-sort={ariaSortAttr('file')} onClick={()=>onSort('file')} tabIndex={0} onKeyDown={(e)=>{ if(e.key==='Enter' || e.key===' ') { e.preventDefault(); onSort('file'); }}}>File <span className="sort-indicator" aria-hidden="true">{sort.key==='file' ? (sort.dir==='asc'?'▲':'▼') : '↕'}</span></div>
                  <div className={`th sortable ${sort.key==='resolution' ? 'sorted '+sort.dir : ''}`} role="columnheader" aria-sort={ariaSortAttr('resolution')} onClick={()=>onSort('resolution')} tabIndex={0} onKeyDown={(e)=>{ if(e.key==='Enter' || e.key===' ') { e.preventDefault(); onSort('resolution'); }}}>Resolution <span className="sort-indicator" aria-hidden="true">{sort.key==='resolution' ? (sort.dir==='asc'?'▲':'▼') : '↕'}</span></div>
                  <div className={`th sortable ${sort.key==='duration' ? 'sorted '+sort.dir : ''}`} role="columnheader" aria-sort={ariaSortAttr('duration')} onClick={()=>onSort('duration')} tabIndex={0} onKeyDown={(e)=>{ if(e.key==='Enter' || e.key===' ') { e.preventDefault(); onSort('duration'); }}}>Duration <span className="sort-indicator" aria-hidden="true">{sort.key==='duration' ? (sort.dir==='asc'?'▲':'▼') : '↕'}</span></div>
                  <div className={`th sortable ${sort.key==='codec' ? 'sorted '+sort.dir : ''}`} role="columnheader" aria-sort={ariaSortAttr('codec')} onClick={()=>onSort('codec')} tabIndex={0} onKeyDown={(e)=>{ if(e.key==='Enter' || e.key===' ') { e.preventDefault(); onSort('codec'); }}}>Codec <span className="sort-indicator" aria-hidden="true">{sort.key==='codec' ? (sort.dir==='asc'?'▲':'▼') : '↕'}</span></div>
                  <div className={`th sortable ${sort.key==='likelihood' ? 'sorted '+sort.dir : ''}`} role="columnheader" aria-sort={ariaSortAttr('likelihood')} onClick={()=>onSort('likelihood')} tabIndex={0} onKeyDown={(e)=>{ if(e.key==='Enter' || e.key===' ') { e.preventDefault(); onSort('likelihood'); }}}>Deepfake Likelihood <span className="sort-indicator" aria-hidden="true">{sort.key==='likelihood' ? (sort.dir==='asc'?'▲':'▼') : '↕'}</span></div>
                  <div className={`th sortable ${sort.key==='label' ? 'sorted '+sort.dir : ''}`} role="columnheader" aria-sort={ariaSortAttr('label')} onClick={()=>onSort('label')} tabIndex={0} onKeyDown={(e)=>{ if(e.key==='Enter' || e.key===' ') { e.preventDefault(); onSort('label'); }}}>Label <span className="sort-indicator" aria-hidden="true">{sort.key==='label' ? (sort.dir==='asc'?'▲':'▼') : '↕'}</span></div>
                  <div className={`th sortable ${sort.key==='analyzedAt' ? 'sorted '+sort.dir : ''}`} role="columnheader" aria-sort={ariaSortAttr('analyzedAt')} onClick={()=>onSort('analyzedAt')} tabIndex={0} onKeyDown={(e)=>{ if(e.key==='Enter' || e.key===' ') { e.preventDefault(); onSort('analyzedAt'); }}}>Analyzed <span className="sort-indicator" aria-hidden="true">{sort.key==='analyzedAt' ? (sort.dir==='asc'?'▲':'▼') : '↕'}</span></div>
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
                  return (
                    <div className={`tr ${flagged ? 'flagged' : ''} ${selected?.id===a.id && selected.analyzedAt===a.analyzedAt ? 'active' : ''}`} role="row" key={a.id+"-"+a.analyzedAt} onClick={(e) => { if(!(e.target as HTMLElement).closest('.sel input')) setSelected(a); }}>
                      <div className="td sel" role="cell"><input type="checkbox" aria-label={`Select ${a.fileName}`} checked={checkedIds.includes(a.id)} readOnly onClick={(e)=>handleCheckboxClick(e, a.id, idx)} /></div>
                      <div className="td filename" role="cell" title={a.fileName}>{a.fileName}</div>
                      <div className="td" role="cell">{a.summary.width && a.summary.height ? `${a.summary.width}x${a.summary.height}` : '—'}</div>
                      <div className="td" role="cell">{a.summary.duration_s ? a.summary.duration_s.toFixed(2)+'s' : '—'}</div>
                      <div className="td" role="cell">{a.summary.codec || '—'}</div>
                      <div className="td likelihood-cell" role="cell">
                        <div className={`likelihood-meter ${flagged? 'alert': ''} ${pctClass}`}> <div className="fill" /> </div>
                        <span className={`pct-text ${flagged? 'alert': ''}`}>{pct}</span>
                      </div>
                      <div className="td" role="cell">{a.summary.deepfake_label || '—'}</div>
                      <div className="td" role="cell">{new Date(a.analyzedAt).toLocaleString()}</div>
                    </div>
                  );
                })}
              </div>
            </div>
            {selected && (
              <div className="analysis-detail" role="region" aria-label="Detailed analysis">
                <div className="detail-head">
                  <h3 className="detail-title">{selected.fileName}</h3>
                  <button className="detail-close" onClick={()=>setSelected(null)} aria-label="Close details">×</button>
                </div>
                <div className="detail-grid">
                  <div><span className="k">Resolution</span><span className="v">{selected.summary.width && selected.summary.height ? `${selected.summary.width}x${selected.summary.height}` : '—'}</span></div>
                  <div><span className="k">FPS</span><span className="v">{selected.summary.fps ?? '—'}</span></div>
                  <div><span className="k">Duration</span><span className="v">{selected.summary.duration_s ? selected.summary.duration_s.toFixed(2)+'s' : '—'}</span></div>
                  <div><span className="k">Codec</span><span className="v">{selected.summary.codec || '—'}</span></div>
                  <div><span className="k">Format Valid</span><span className="v">{String(selected.summary.format_valid)}</span></div>
                  <div><span className="k">Decode Valid</span><span className="v">{String(selected.summary.decode_valid)}</span></div>
                  <div><span className="k">Deepfake %</span><span className="v">{((selected.summary.deepfake_likelihood||0)*100).toFixed(2)}%</span></div>
                  <div><span className="k">Label</span><span className="v">{selected.summary.deepfake_label || '—'}</span></div>
                </div>
                {selected.raw && (
                  <details className="raw-block">
                    <summary>Raw JSON</summary>
                    <pre>{JSON.stringify(selected.raw, null, 2)}</pre>
                  </details>
                )}
              </div>
            )}
          </div>
        )}
      </div>
      {showDeleteConfirm && (
        <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="del-title" onMouseDown={(e)=> { if (e.target===e.currentTarget) cancelBulkDelete(); }}>
          <div className="modal" role="document">
            <h3 id="del-title">Delete {checkedIds.length} analysis{checkedIds.length>1?'es':''}?</h3>
            <p className="modal-text">This action will permanently remove the selected item{checkedIds.length>1?'s':''} from this report. It cannot be undone.</p>
            <div className="modal-actions">
              <button className="btn" onClick={cancelBulkDelete}>Cancel</button>
              <button className="btn danger" onClick={performBulkDelete} autoFocus>Delete</button>
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
