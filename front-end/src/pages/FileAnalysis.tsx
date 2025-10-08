
import React from "react";
import "./PageStyles.css";
import "./FileAnalysis.css";
import UploadCard from "../components/UploadCard";

const FileAnalysis: React.FC<{ label: string; onNavigate?: (p: string) => void; currentPage?: string }> = ({ label, onNavigate, currentPage }) => {
  // label is like "FILE ANALYSIS 1" or "FILE ANALYSIS 2"
  return (
    <div>
      <div className="file-analysis-container">
        <div className="file-analysis-inner">
          <div className="file-analysis-content">
            <UploadCard pageKey={currentPage} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default FileAnalysis;
