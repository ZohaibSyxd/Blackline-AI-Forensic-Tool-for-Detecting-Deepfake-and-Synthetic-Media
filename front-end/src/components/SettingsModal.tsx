import React, { useEffect, useState } from 'react';
import './SettingsModal.css';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const DARK_KEY = 'bl_dark';

const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  const [dark, setDark] = useState<boolean>(false);

  useEffect(() => {
    const saved = localStorage.getItem(DARK_Key_Fallback());
    const initial = saved ? saved === '1' : false;
    setDark(initial);
  }, [isOpen]);

  const applyDark = (enabled: boolean) => {
    setDark(enabled);
    try { localStorage.setItem(DARK_KEY, enabled ? '1' : '0'); } catch {}
    const root = document.documentElement;
    if (enabled) root.classList.add('dark');
    else root.classList.remove('dark');
  };

  if (!isOpen) return null;

  return (
    <div className="sm-overlay" role="dialog" aria-modal="true">
      <div className="sm-panel">
        <h3 className="sm-title">Settings</h3>
        <div className="sm-section">
          <label className="sm-row">
            <input type="checkbox" checked={dark} onChange={(e) => applyDark(e.target.checked)} />
            <span>Enable dark mode</span>
          </label>
        </div>
        <div className="sm-actions">
          <button className="sm-btn" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
};

// Backward/typo-safe getter
function DARK_Key_Fallback() { return DARK_KEY; }

export default SettingsModal;
