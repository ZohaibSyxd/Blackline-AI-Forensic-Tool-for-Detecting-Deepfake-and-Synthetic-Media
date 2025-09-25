import React, { useEffect, useRef, useState } from 'react';
import './ProfilePopup.css';
import ProfileModal from './ProfileModal';

interface Props {
  open: boolean;
  onClose: () => void;
  user?: { name?: string; email?: string; plan?: 'Guest' | 'Premium' };
}

const ProfilePopup: React.FC<Props> = ({ open, onClose, user }) => {
  const ref = useRef<HTMLDivElement | null>(null);
  const [profileOpen, setProfileOpen] = useState(false);

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

  const u = user || { name: 'Guest User', email: 'guest@example.com', plan: 'Guest' };

  // If profile modal is open, only render modal (popup hidden)
  if (profileOpen) {
    return (
      <ProfileModal
        open={profileOpen}
        user={u}
        onClose={() => setProfileOpen(false)}
        onLogin={() => {/* placeholder */}}
        onSignup={() => {/* placeholder */}}
        onSignOut={() => {/* placeholder */}}
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
        <button className="pp-btn" onClick={() => { setProfileOpen(true); onClose(); }}>View profile</button>
        <button className="pp-btn ghost">{u.plan === 'Guest' ? 'Login' : 'Sign out'}</button>
      </div>
    </div>
  );
};

export default ProfilePopup;
