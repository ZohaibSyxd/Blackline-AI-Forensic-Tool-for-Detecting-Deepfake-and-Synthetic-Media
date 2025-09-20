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
    <div className="cd-overlay" role="dialog" aria-modal="true" aria-labelledby="cd-title">
      <div className="cd-panel">
        <h3 id="cd-title" className="cd-title">{title}</h3>
        <div className="cd-message">{message}</div>
        <div className="cd-actions">
          <button className="cd-btn" onClick={onCancel}> {cancelText} </button>
          <button className={`cd-btn ${danger ? 'cd-btn-danger' : 'cd-btn-primary'}`} onClick={onConfirm}>{confirmText}</button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmDialog;
