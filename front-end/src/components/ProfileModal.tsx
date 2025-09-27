import React, { useEffect, useRef, useState } from 'react';
import { subscribe, getAuthState, login as authLogin, signup as authSignup } from '../state/authStore';
import { createPortal } from 'react-dom';
import './ProfileModal.css';

interface ProfileModalProps {
  open: boolean;
  onClose: () => void;
  user: { name?: string; email?: string; plan?: 'Guest' | 'Premium' };
  onLogin: (username?: string, email?: string, password?: string) => void;
  onSignup: (username?: string, email?: string, password?: string) => void;
  onSignOut: () => void;
}

const ProfileModal: React.FC<ProfileModalProps> = ({ open, onClose, user, onLogin, onSignup, onSignOut }) => {
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    function handleKey(e: KeyboardEvent) { if (e.key === 'Escape') onClose(); }
    function handleClick(e: MouseEvent) {
      if (dialogRef.current && !dialogRef.current.contains(e.target as Node)) onClose();
    }
    document.addEventListener('keydown', handleKey);
    document.addEventListener('mousedown', handleClick);
    return () => {
      document.removeEventListener('keydown', handleKey);
      document.removeEventListener('mousedown', handleClick);
    };
  }, [open, onClose]);

  if (!open) return null;

  // Ensure we render at document.body so positioning is centered relative to viewport
  const target = typeof document !== 'undefined' ? document.body : null;
  if (!target) return null;

  const [auth, setAuth] = useState(getAuthState());
  useEffect(() => { const unsub = subscribe(setAuth); return () => { unsub(); }; }, []);
  const isGuest = (auth.user ? auth.user.plan : user.plan) === 'Guest';
  const [mode, setMode] = useState<'login' | 'signup'>('login');
  const [form, setForm] = useState({ username: '', email: '', password: '' });
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Auto-close when authenticated (non-guest) or user object present
    if (auth.user) {
      // give a tiny delay for visual feedback
      const t = setTimeout(() => { onClose(); }, 400);
      return () => clearTimeout(t);
    }
  }, [auth.user, onClose]);

  async function handleAuth(action: 'login' | 'signup') {
    setPending(true); setError(null);
    try {
      if (!form.username || !form.password || (action==='signup' && !form.email)) {
        setError('Please fill all required fields');
        setPending(false);
        return;
      }
      if (action === 'login') {
        await authLogin(form.username, form.password);
      } else {
        await authSignup(form.username, form.email, form.password);
      }
      // auth subscription effect will close modal on success
    } catch (e: any) {
      setError(e.message || 'Failed');
    } finally { setPending(false); }
  }

  return createPortal(
    <div className="profile-modal-overlay" role="dialog" aria-modal="true" aria-label="Profile details">
      <div className="profile-modal" ref={dialogRef}>
        <div className="pm-header">
          <div className="pm-avatar">{(user.name||'G').charAt(0).toUpperCase()}</div>
          <div className="pm-meta">
            <h2 className="pm-name">{user.name || 'Guest User'}</h2>
            <div className="pm-email">{user.email || 'guest@example.com'}</div>
            <div className={`pm-plan ${isGuest ? 'guest' : 'premium'}`}>{user.plan || 'Guest'}</div>
          </div>
          <button className="pm-close" aria-label="Close" onClick={onClose}>×</button>
        </div>
        <div className="pm-body">
          {isGuest ? (
            <>
              <p className="pm-blurb">Create an account to save analyses, sync across devices, and unlock advanced forensic insights.</p>
              <div className="auth-toggle">
                <button className={`pm-btn small ${mode==='login' ? 'primary' : ''}`} onClick={()=>setMode('login')}>Login</button>
                <button className={`pm-btn small ${mode==='signup' ? 'primary' : ''}`} onClick={()=>setMode('signup')}>Sign up</button>
              </div>
              <form className="pm-auth-form" onSubmit={(e)=>{ e.preventDefault(); handleAuth(mode); }}>
                <label className="pm-field">Username
                  <input required value={form.username} onChange={e=>setForm(f=>({...f, username:e.target.value}))} />
                </label>
                {mode==='signup' && (
                  <label className="pm-field">Email
                    <input required type="email" value={form.email} onChange={e=>setForm(f=>({...f, email:e.target.value}))} />
                  </label>
                )}
                <label className="pm-field">Password
                  <input required type="password" value={form.password} onChange={e=>setForm(f=>({...f, password:e.target.value}))} />
                </label>
                {error && <div className="pm-error" role="alert">{error}</div>}
                {auth.error && !error && <div className="pm-error" role="alert">{auth.error}</div>}
                <div className="pm-actions-inline">
                  <button className="pm-btn primary" disabled={pending}>{pending ? (mode==='login' ? 'Logging in…' : 'Creating…') : (mode==='login' ? 'Login' : 'Create Account')}</button>
                </div>
              </form>
            </>
          ) : (
            <>
              <div className="pm-stats-grid">
                <div className="pm-stat"><span className="k">Analyses</span><span className="v">—</span></div>
                <div className="pm-stat"><span className="k">Storage Used</span><span className="v">—</span></div>
                <div className="pm-stat"><span className="k">Member Since</span><span className="v">—</span></div>
              </div>
              <p className="pm-blurb">Profile overview and usage metrics will appear here.</p>
              <div className="pm-actions-inline">
                <button className="pm-btn" onClick={onSignOut}>Sign out</button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>,
    target
  );
};

export default ProfileModal;
