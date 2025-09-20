import React from 'react';
import './IconPickerModal.css';
import { ICONS } from './icons';

interface IconPickerModalProps {
  isOpen: boolean;
  currentIcon?: string;
  onClose: () => void;
  onSelect: (icon?: string) => void;
}

const IconPickerModal: React.FC<IconPickerModalProps> = ({ isOpen, currentIcon, onClose, onSelect }) => {
  if (!isOpen) return null;
  return (
    <div className="ip-overlay" role="dialog" aria-modal="true">
      <div className="ip-panel">
        <h3 className="ip-title">Choose an icon</h3>
        <div className="ip-icons">
          {ICONS.map((ic) => (
            <button
              key={ic}
              className={`ip-icon-btn ${currentIcon === ic ? 'selected' : ''}`}
              onClick={() => onSelect(ic)}
              aria-label={`Select ${ic}`}
            >
              <span className="ip-emoji">{ic}</span>
            </button>
          ))}
          <button className="ip-icon-btn none" onClick={() => onSelect(undefined)} aria-label="No icon">
            <span className="ip-emoji">∅</span>
          </button>
        </div>
        <div className="ip-actions">
          <button className="ip-btn ip-btn-secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
};

export default IconPickerModal;
