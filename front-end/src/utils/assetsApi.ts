// Minimal client for Assets API
import { getAuthState } from '../state/authStore';

export interface Asset {
  id: number;
  original_name: string;
  mime?: string | null;
  size_bytes?: number | null;
  sha256?: string | null;
  stored_path?: string | null;
  remote_key?: string | null;
  visibility: string;
  created_at: number; // epoch seconds
}

const API_BASE: string = (import.meta as any).env?.VITE_API_BASE || (typeof window !== 'undefined' ? window.location.origin : '');

function authHeaders(extra?: Record<string, string>) {
  const tok = getAuthState().token;
  const h: Record<string, string> = { ...(extra || {}) };
  if (tok) h['Authorization'] = `Bearer ${tok}`;
  return h;
}

export async function listAssets(): Promise<Asset[]> {
  const res = await fetch(`${API_BASE}/api/assets`, { headers: authHeaders() });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadAsset(file: File): Promise<Asset> {
  const fd = new FormData();
  fd.append('file', file, file.name);
  const res = await fetch(`${API_BASE}/api/assets`, { method: 'POST', body: fd, headers: authHeaders() });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface AnalyzeResponse {
  asset: any;
  validate: any;
  probe: any;
  summary: any;
}

export async function analyzeAsset(assetId: number, model: string = 'stub', jobId?: string): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/api/analyze/asset`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ asset_id: assetId, model, job_id: jobId || null }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getUploadUrl(fileName: string, contentType?: string): Promise<{ key: string; upload_url: string; headers?: Record<string,string> }>{
  const res = await fetch(`${API_BASE}/api/assets/upload-url`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ file_name: fileName, content_type: contentType || null }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function confirmUpload(key: string, originalName: string, mime?: string): Promise<Asset> {
  const res = await fetch(`${API_BASE}/api/assets/confirm`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ key, original_name: originalName, mime: mime || null }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getPlaybackUrl(assetId: number): Promise<string> {
  const res = await fetch(`${API_BASE}/api/assets/${assetId}/playback-url`, { headers: authHeaders() });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.url as string;
}
