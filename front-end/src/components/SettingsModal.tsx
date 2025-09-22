import React, { useEffect, useState } from 'react';
import './SettingsModal.css';
import { ThemeMode, getStoredTheme, storeTheme, applyTheme, watchSystemTheme } from '../theme';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  const [mode, setMode] = useState<ThemeMode>('system');
  const [dispose, setDispose] = useState<undefined | (() => void)>();

  useEffect(() => {
    if (!isOpen) return;
    const current = getStoredTheme();
    setMode(current);
    // If system mode, watch for changes
    const d = current === 'system' ? watchSystemTheme(() => applyTheme('system')) : undefined;
    setDispose(() => d);
    return () => { if (d) d(); };
  }, [isOpen]);

  const choose = (m: ThemeMode) => {
    setMode(m);
    storeTheme(m);
    applyTheme(m);
    if (dispose) { dispose(); setDispose(undefined); }
    if (m === 'system') {
      const d = watchSystemTheme(() => applyTheme('system'));
      setDispose(() => d);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="sm-overlay" role="dialog" aria-modal="true">
      <div className="sm-panel">
        <h3 className="sm-title">Settings</h3>
        <div className="sm-section">
          <div className="sm-row" role="radiogroup" aria-label="Theme">
            <label className="sm-row sm-radio">
              <input type="radio" name="theme" checked={mode==='light'} onChange={() => choose('light')} />
              <span>Light</span>
            </label>
            <label className="sm-row sm-radio">
              <input type="radio" name="theme" checked={mode==='dark'} onChange={() => choose('dark')} />
              <span>Dark</span>
            </label>
            <label className="sm-row sm-radio">
              <input type="radio" name="theme" checked={mode==='system'} onChange={() => choose('system')} />
              <span>System</span>
            </label>
          </div>
        </div>
        <div className="sm-actions">
          <button className="sm-btn" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
