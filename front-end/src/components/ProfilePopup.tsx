import React, { useEffect, useRef, useState } from 'react';
import './ProfilePopup.css';
import ProfileModal from './ProfileModal';
import { subscribe, getAuthState, logout, login, signup } from '../state/authStore';

interface Props {
  open: boolean;
  onClose: () => void;
  user?: { name?: string; email?: string; plan?: 'Guest' | 'Premium' };
}

const ProfilePopup: React.FC<Props> = ({ open, onClose, user }) => {
  const ref = useRef<HTMLDivElement | null>(null);
  const [profileOpen, setProfileOpen] = useState(false);
  const [auth, setAuth] = useState(getAuthState());

  useEffect(() => {
    const unsub = subscribe(setAuth);
    return () => { unsub(); };
  }, []);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open, onClose]);

  // If main popup isn't open and modal isn't open, render nothing.
  if (!open && !profileOpen) return null;

  const u: { name?: string; email?: string; plan?: 'Guest' | 'Premium' } = auth.user
    ? { name: auth.user.username, email: auth.user.email, plan: (auth.user.plan === 'Guest' ? 'Guest' : 'Premium') }
    : (user || { name: 'Guest User', email: 'guest@example.com', plan: 'Guest' });

  // If profile modal is open, only render modal (popup hidden)
  if (profileOpen) {
    return (
        <ProfileModal
          open={profileOpen}
          user={u}
          onClose={() => setProfileOpen(false)}
          onLogin={(username?: string, _email?: string, password?: string) => { if (username && password) login(username, password); }}
          onSignup={(username?: string, email?: string, password?: string) => { if (username && email && password) signup(username, email, password); }}
          onSignOut={() => { logout(); setProfileOpen(false); }}
        />
    );
  }

  return (
    <div className="profile-popup" role="dialog" aria-label="Profile" ref={ref}>
      <div className="pp-row">
        <div className="pp-avatar">{u.name ? u.name[0].toUpperCase() : 'G'}</div>
        <div className="pp-info">
          <div className="pp-name">{u.name}</div>
          <div className="pp-email">{u.email}</div>
        </div>
      </div>
      <div className="pp-badge-wrap">
        <div className={`pp-badge ${u.plan === 'Premium' ? 'premium' : 'guest'}`}>{u.plan}</div>
      </div>
      <div className="pp-actions">
        <button className="pp-btn" onClick={() => { setProfileOpen(true); onClose(); }}>{auth.user ? 'Account' : 'View profile'}</button>
        {auth.user ? (
          <button className="pp-btn ghost" onClick={() => { logout(); onClose(); }}>Sign out</button>
        ) : (
          <button className="pp-btn ghost" onClick={() => { setProfileOpen(true); onClose(); }}>Login</button>
        )}
      </div>
    </div>
  );
};

export default ProfilePopup;
