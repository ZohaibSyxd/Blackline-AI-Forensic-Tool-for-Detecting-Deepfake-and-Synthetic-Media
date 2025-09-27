// Simple client-side auth store using localStorage for prototype.
// NOT secure for production – tokens are just stored in localStorage.
// Provides subscribe() to reactively update components.

export interface User {
  username: string;
  email: string;
  plan: string;
  created_at: number;
}

interface AuthState {
  user: User | null;
  token: string | null;
  loading: boolean;
  error: string | null;
}

const LS_KEY = 'bl_auth_v1';

let state: AuthState = loadState();
const listeners = new Set<(s: AuthState) => void>();

function loadState(): AuthState {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return { user: null, token: null, loading: false, error: null };
    const parsed = JSON.parse(raw);
    return { user: parsed.user || null, token: parsed.token || null, loading: false, error: null };
  } catch {
    return { user: null, token: null, loading: false, error: null };
  }
}

function persist() {
  try { localStorage.setItem(LS_KEY, JSON.stringify({ user: state.user, token: state.token })); } catch {}
}

function set(partial: Partial<AuthState>) {
  state = { ...state, ...partial };
  persist();
  listeners.forEach(l => l(state));
}

export function subscribe(fn: (s: AuthState) => void) { listeners.add(fn); fn(state); return () => listeners.delete(fn); }
export function getAuthState() { return state; }

const API_BASE = (import.meta as any).env?.VITE_API_BASE || 'http://localhost:8010';

export async function login(username: string, password: string) {
  set({ loading: true, error: null });
  try {
    const body = new URLSearchParams();
    body.append('username', username);
    body.append('password', password);
    body.append('grant_type', 'password');
    const res = await fetch(`${API_BASE}/api/auth/login`, { method: 'POST', body, headers: { 'Content-Type': 'application/x-www-form-urlencoded' } });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const token = data.access_token;
    const meRes = await fetch(`${API_BASE}/api/auth/me`, { headers: { Authorization: `Bearer ${token}` } });
    if (!meRes.ok) throw new Error(await meRes.text());
    const me = await meRes.json();
    set({ user: me.user, token, loading: false, error: null });
  } catch (e: any) {
    set({ loading: false, error: e.message || 'Login failed' });
  }
}

export async function signup(username: string, email: string, password: string) {
  set({ loading: true, error: null });
  try {
    const res = await fetch(`${API_BASE}/api/auth/signup`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, email, password }) });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    set({ user: data.user, token: data.token, loading: false, error: null });
  } catch (e: any) {
    set({ loading: false, error: e.message || 'Signup failed' });
  }
}

export function logout() { set({ user: null, token: null }); }

// Optional: refresh /me (for page reload with stored token)
export async function refreshUser() {
  if (!state.token) return;
  try {
    const res = await fetch(`${API_BASE}/api/auth/me`, { headers: { Authorization: `Bearer ${state.token}` } });
    if (res.ok) {
      const data = await res.json();
      set({ user: data.user });
    } else {
      set({ user: null, token: null });
    }
  } catch { set({ user: null, token: null }); }
}
