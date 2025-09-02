import React from "react";
import "./Reports.css";

const labelFor = (filePage?: string) => {
  if (!filePage) return "FILE ANALYSIS";
  if (filePage === "file1") return "FILE ANALYSIS 1";
  if (filePage === "file2") return "FILE ANALYSIS 2";
  if (filePage === "file3") return "FILE ANALYSIS 3";
  if (filePage === "file$") return "FILE ANALYSIS $";
  return "FILE ANALYSIS";
};

const Reports: React.FC<{ filePage?: string }> = ({ filePage }) => {
  const label = labelFor(filePage);

  return (
    <div className="reports-page">
      <h2>Reports</h2>
      <p className="muted">Basic heatmap template (placeholder) — {label}</p>

      <div className="heatmap">
        {/* simple grid heatmap squares as placeholder */}
        {Array.from({ length: 5 }).map((_, row) => (
          <div className="heatmap-row" key={row}>
            {Array.from({ length: 12 }).map((__, col) => {
              const intensity = Math.floor(Math.random() * 5);
              return <div className={`heatcell intensity-${intensity}`} key={col} />;
            })}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Reports;
