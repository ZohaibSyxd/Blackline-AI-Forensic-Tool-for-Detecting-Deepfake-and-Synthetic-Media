import React, { useEffect, useRef, useState } from "react";
import "../pages/FileAnalysis.css";
import "./Sidebar.css"; // reuse item-menu styles
import ConfirmDialog from "./ConfirmDialog";

interface Props {
  currentPage: string;
  onNavigate: (p: string) => void;
  lastFilePage?: string;
  title?: string;
  onDeleteCurrent?: (key: string) => void;
}

const SectionHeader: React.FC<Props> = ({ currentPage, onNavigate, lastFilePage, title, onDeleteCurrent }) => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  const activeKey = currentPage && currentPage.startsWith('file') ? currentPage : (lastFilePage || 'file1');

  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, []);

  const requestDelete = () => {
    setMenuOpen(false);
    setConfirmOpen(true);
  };

  return (
    <div className="page-content">
      <div className="file-analysis-header">
        <div className="file-analysis-title">{title || 'Dashboard'}</div>
        <div className="file-analysis-tabs">
          <div className={`file-analysis-tab ${currentPage && currentPage.startsWith('file') ? 'active' : ''}`} onClick={() => onNavigate(lastFilePage || 'file1')}>Documents</div>
          <div className={`file-analysis-tab ${currentPage === 'reports' ? 'active' : ''}`} onClick={() => onNavigate('reports')}>Reports</div>
          <div className="file-analysis-tab sidebar-actions-wrap" ref={menuRef}>
            <button
              className="header-more-btn"
              title="More"
              aria-label="More actions"
              onClick={() => setMenuOpen(o => !o)}
              aria-haspopup="menu"
            >…</button>
            {menuOpen && (
              <div className="item-menu" role="menu">
                <button className="item-menu-btn danger" role="menuitem" onClick={requestDelete}>Delete</button>
              </div>
            )}
          </div>
        </div>
      </div>
      <ConfirmDialog
        open={confirmOpen}
        title="Confirm deletion"
        message={`Delete "${title || 'this analysis'}"? This action cannot be undone.`}
        confirmText="Delete"
        cancelText="Cancel"
        danger
        onConfirm={() => { onDeleteCurrent && onDeleteCurrent(activeKey); setConfirmOpen(false); }}
        onCancel={() => setConfirmOpen(false)}
      />
    </div>
  );
};

export default SectionHeader;
