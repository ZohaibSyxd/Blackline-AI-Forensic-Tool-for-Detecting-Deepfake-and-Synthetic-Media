import React from "react";
import "./PageStyles.css";
import "./Reports.css"; // reuse card styles for page-level cards
import { getAnalysesForPage } from '../state/analysisStore';

type PageEntry = { key: string; label: string; icon?: string };

const Dashboard: React.FC<{ pages: PageEntry[] }> = ({ pages }) => {
  const filePages = pages.filter(p => p.key !== 'dashboard');
  return (
    <div className="page-content">
      {/* File analysis page cards - appear on Dashboard for quick navigation */}
      <div className="dashboard-page-cards">
        <div className="file-analysis-pages">
          {filePages.map(p => (
            <div key={p.key} className="file-page-card" onClick={() => { try { window.dispatchEvent(new CustomEvent('bl:navigate',{ detail: { page: p.key } })); } catch {} }} role="button" tabIndex={0}>
              <div className="file-page-card-title">{p.label}</div>
              <div className="file-page-card-body"><div className="file-page-count">{getAnalysesForPage(p.key).length} analyses</div><div className="file-page-cta">Open</div></div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
