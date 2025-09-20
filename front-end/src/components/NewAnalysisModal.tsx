import React, { useState, useEffect } from 'react';
import './NewAnalysisModal.css';

// Emoji options for users to choose from
const ICONS = ['📁', '📂', '🗂️', '🧪', '🎬', '🧠', '🔍', '📊'];

interface Props {
  isOpen: boolean;
  defaultName?: string;
  onClose: () => void;
  onCreate: (name: string, icon?: string) => void;
}

const NewAnalysisModal: React.FC<Props> = ({ isOpen, defaultName = '', onClose, onCreate }) => {
  const [name, setName] = useState(defaultName);
  const [icon, setIcon] = useState<string | undefined>(ICONS[0]);

  useEffect(() => {
    setName(defaultName);
    setIcon(ICONS[0]);
  }, [defaultName, isOpen]);

  if (!isOpen) return null;

  return (
    <div className="nm-overlay" role="dialog" aria-modal="true">
      <div className="nm-panel">
        <h3 className="nm-title">Create New Analysis</h3>
        <label className="nm-label">Name</label>
        <input className="nm-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="Enter a name" />

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
          <button className="nm-btn nm-btn-secondary" onClick={onClose}>Cancel</button>
          <button
            className="nm-btn nm-btn-primary"
            onClick={() => {
              const finalName = name && name.trim() ? name.trim() : 'New Analysis';
              onCreate(finalName, icon);
            }}
          >Create</button>
        </div>
      </div>
    </div>
  );
};

export default NewAnalysisModal;
