import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import './ProfileModal.css';

interface ProfileModalProps {
  open: boolean;
  onClose: () => void;
  user: { name?: string; email?: string; plan?: 'Guest' | 'Premium' };
  onLogin: () => void;
  onSignup: () => void;
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

  const isGuest = user.plan === 'Guest';

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
              <div className="pm-actions-inline">
                <button className="pm-btn primary" onClick={onSignup}>Sign up</button>
                <button className="pm-btn" onClick={onLogin}>Login</button>
              </div>
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
