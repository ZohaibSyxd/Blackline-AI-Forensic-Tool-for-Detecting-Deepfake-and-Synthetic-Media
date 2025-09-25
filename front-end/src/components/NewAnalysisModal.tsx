import React, { useState, useEffect, useRef } from 'react';
import './NewAnalysisModal.css';
import { ICONS } from './icons';

interface Props {
  isOpen: boolean;
  defaultName?: string;
  onClose: () => void;
  onCreate: (name: string, icon?: string) => void;
}

const NewAnalysisModal: React.FC<Props> = ({ isOpen, defaultName = '', onClose, onCreate }) => {
  // Start with an empty input; show the suggestion as a placeholder instead of a prefilled value
  const [name, setName] = useState('');
  const [icon, setIcon] = useState<string | undefined>(ICONS[0]);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    // Clear the name field each time the modal opens; keep icon reset
    setName('');
    setIcon(ICONS[0]);
    setError(null);
  }, [defaultName, isOpen]);

  // Submit handler used by the form and Create button
  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const typed = name && name.trim();
    if (!typed) {
      setError('Please Enter This Field');
      if (inputRef.current) inputRef.current.focus();
      return;
    }
    onCreate(typed, icon);
  };

  // Close on Escape for convenience
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') {
        ev.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  // Auto-focus and select the input when the modal opens
  useEffect(() => {
    if (!isOpen) return;
    // wait a frame to ensure the input is in the DOM and visible
    const id = requestAnimationFrame(() => {
      if (inputRef.current) {
        inputRef.current.focus({ preventScroll: true });
        // Select any existing text so it's highlighted
        try { inputRef.current.select(); } catch {}
      }
    });
    return () => cancelAnimationFrame(id);
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="nm-overlay" role="dialog" aria-modal="true">
      <div className="nm-panel">
        <h3 className="nm-title">Create New Analysis</h3>
        <form onSubmit={handleSubmit}>
        <label className="nm-label">Name</label>
        <input
          className={`nm-input${error ? ' error' : ''}`}
          value={name}
          onChange={(e) => { setName(e.target.value); if (e.target.value.trim()) setError(null); }}
          ref={inputRef}
          placeholder={error ? 'Please Enter a Name' : 'Enter a Name'}
        />
        {/* no inline error message; use placeholder + red highlight instead */}

        <div id="nm-icon-label" className="nm-label">Choose an icon</div>
        <div className="nm-icons">
          {ICONS.map((ic, idx) => {
            const id = `nm-icon-${idx}`;
            const selected = icon === ic;
            return (
              <span key={ic} className="nm-icon-wrap">
                <input
                  type="radio"
                  id={id}
                  name="nm-icon"
                  className="nm-radio"
                  checked={selected}
                  onChange={() => setIcon(ic)}
                  value={ic}
                />
                <label htmlFor={id} className={`nm-icon-btn ${selected ? 'selected' : ''}`}>
                  <span className="nm-emoji">{ic}</span>
                </label>
              </span>
            );
          })}
  </div>

        <div className="nm-actions">
          <button type="button" className="nm-btn nm-btn-secondary" onClick={onClose}>Cancel</button>
          <button type="submit" className="nm-btn nm-btn-primary">Create</button>
        </div>
        </form>
      </div>
    </div>
  );
};

export default NewAnalysisModal;
