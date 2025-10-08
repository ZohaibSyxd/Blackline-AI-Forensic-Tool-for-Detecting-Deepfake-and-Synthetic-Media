// Central client-side store for analysis summaries per page & uploaded item
// Persisted in localStorage so the Reports page can aggregate.

export interface StoredAnalysisSummary {
  id: string;               // upload item id
  pageKey: string;          // analysis page
  fileName: string;
  mime?: string;
  analyzedAt: number;
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
  raw?: any; // full raw server response for detailed report view
}

const LS_KEY = 'bl_analysis_summaries_v1';

function loadAll(): StoredAnalysisSummary[] {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    if (Array.isArray(arr)) return arr;
  } catch {}
  return [];
}

function saveAll(list: StoredAnalysisSummary[]) {
  try { localStorage.setItem(LS_KEY, JSON.stringify(list)); } catch {}
}

export function upsertAnalysis(summary: StoredAnalysisSummary) {
  const all = loadAll();
  const idx = all.findIndex(s => s.id === summary.id && s.pageKey === summary.pageKey);
  if (idx >= 0) all[idx] = summary; else all.push(summary);
  saveAll(all);
}

export function getAnalysesForPage(pageKey?: string): StoredAnalysisSummary[] {
  const all = loadAll();
  if (!pageKey) return all;
  return all.filter(a => a.pageKey === pageKey);
}

export function getAllAnalyses(): StoredAnalysisSummary[] { return loadAll(); }

export function clearAnalysesForPage(pageKey: string) {
  const all = loadAll().filter(a => a.pageKey !== pageKey);
  saveAll(all);
}

// Delete a set of analyses by id (within a specific page if provided)
export function deleteAnalyses(ids: string[], pageKey?: string) {
  if (!ids.length) return;
  const idSet = new Set(ids);
  const all = loadAll();
  const filtered = all.filter(a => {
    if (!idSet.has(a.id)) return true; // keep if not selected
    if (pageKey && a.pageKey !== pageKey) return true; // keep if different page
    return false; // remove
  });
  saveAll(filtered);
}
