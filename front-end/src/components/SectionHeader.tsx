import React from "react";
import "../pages/FileAnalysis.css";

const SectionHeader: React.FC<{ currentPage: string; onNavigate: (p: string) => void; lastFilePage?: string; title?: string }> = ({ currentPage, onNavigate, lastFilePage, title }) => {
  return (
    <div className="page-content">
      <div className="file-analysis-header">
        <div className="file-analysis-title">{title || 'Dashboard'}</div>
        <div className="file-analysis-tabs">
          <div className={`file-analysis-tab ${currentPage && currentPage.startsWith('file') ? 'active' : ''}`} onClick={() => onNavigate(lastFilePage || 'file1')}>Documents</div>
          <div className={`file-analysis-tab ${currentPage === 'reports' ? 'active' : ''}`} onClick={() => onNavigate('reports')}>Reports</div>
          <div className="file-analysis-tab">...</div>
        </div>
      </div>
    </div>
  );
};

export default SectionHeader;
