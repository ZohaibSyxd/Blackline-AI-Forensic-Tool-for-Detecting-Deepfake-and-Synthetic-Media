
import React from "react";
import "./PageStyles.css";

const FileAnalysis: React.FC<{ label: string }> = ({ label }) => {
  return (
    <div className="page-content">
      <h1>{label}</h1>
      <p>This is the {label} view. Add your analysis content here.</p>
    </div>
  );
};

export default FileAnalysis;
