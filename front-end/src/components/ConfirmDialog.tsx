import React from 'react';
import './ConfirmDialog.css';

interface ConfirmDialogProps {
  open: boolean;
  title?: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  open,
  title = 'Confirm',
  message,
  confirmText = 'Delete',
  cancelText = 'Cancel',
  danger = false,
  onConfirm,
  onCancel,
}) => {
  if (!open) return null;

  return (
    <div
      className="cd-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="cd-title"
      onKeyDown={(e) => {
        if (e.key === 'Escape') { e.stopPropagation(); onCancel(); }
      }}
    >
      <div className="cd-panel">
        <h3 id="cd-title" className="cd-title">{title}</h3>
        <form
          onSubmit={(e) => { e.preventDefault(); onConfirm(); }}
        >
          <div className="cd-message">{message}</div>
          <div className="cd-actions">
            <button type="button" className="cd-btn" onClick={onCancel}> {cancelText} </button>
            <button
              type="submit"
              className={`cd-btn ${danger ? 'cd-btn-danger' : 'cd-btn-primary'}`}
              autoFocus
            >
              {confirmText}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ConfirmDialog;
