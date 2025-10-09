import React, { useEffect, useState } from 'react';
import './SettingsModal.css';
import {
  ThemeMode,
  getStoredTheme,
  storeTheme,
  applyTheme,
  watchSystemTheme,
  ColorScheme,
  listColorSchemes,
  getStoredColorScheme,
  storeColorScheme,
  applyColorScheme,
} from '../theme';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  const [mode, setMode] = useState<ThemeMode>('system');
  const [scheme, setScheme] = useState<ColorScheme>('ocean');
  const [dispose, setDispose] = useState<undefined | (() => void)>();

  useEffect(() => {
    if (!isOpen) return;
  const current = getStoredTheme();
  setMode(current);
  const currentScheme = getStoredColorScheme();
  setScheme(currentScheme);
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

  const chooseScheme = (s: ColorScheme) => {
    setScheme(s);
    storeColorScheme(s);
    applyColorScheme(s);
  };

  if (!isOpen) return null;

  return (
    <div className="sm-overlay" role="dialog" aria-modal="true">
      <div className="sm-panel">
        <h3 className="sm-title">Settings</h3>

        <div className="sm-section">
          <div className="sm-row sm-justify-between">
            <div>
              <div className="sm-section-title">Appearance</div>
              <div className="sm-section-desc">Choose light or dark, or follow your system.</div>
            </div>
          </div>
          <div className="sm-row sm-wrap" role="radiogroup" aria-label="Theme Mode">
            <label className="sm-radio-tile">
              <input type="radio" name="theme-mode" checked={mode==='light'} onChange={() => choose('light')} />
              <span className="sm-radio-label">Light</span>
            </label>
            <label className="sm-radio-tile">
              <input type="radio" name="theme-mode" checked={mode==='dark'} onChange={() => choose('dark')} />
              <span className="sm-radio-label">Dark</span>
            </label>
            <label className="sm-radio-tile">
              <input type="radio" name="theme-mode" checked={mode==='system'} onChange={() => choose('system')} />
              <span className="sm-radio-label">System</span>
            </label>
          </div>
        </div>

        <div className="sm-section">
          <div className="sm-row sm-justify-between">
            <div>
              <div className="sm-section-title">Color scheme</div>
              <div className="sm-section-desc">Pick an accent color used for highlights and buttons.</div>
            </div>
          </div>
          <div className="sm-schemes-grid" role="list" aria-label="Color schemes">
            {listColorSchemes().map(({ key, label }) => (
              <button
                key={key}
                role="listitem"
                className={`sm-scheme sm-scheme-${key}${scheme===key ? ' selected' : ''}`}
                data-selected={scheme===key ? 'true' : 'false'}
                onClick={() => chooseScheme(key)}
                title={label}
              >
                <span className="sm-swatch" />
                <span className="sm-scheme-label">{label}</span>
              </button>
            ))}
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
