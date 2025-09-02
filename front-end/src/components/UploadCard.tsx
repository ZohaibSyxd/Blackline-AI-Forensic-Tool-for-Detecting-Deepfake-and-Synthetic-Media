import React, { useState, useRef } from "react";
import "./UploadCard.css";

const ACCEPT = ".jpg,.jpeg,.png,.gif,.mp4,.avi";

const UploadCard: React.FC = () => {
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  function onFiles(flist: FileList | null) {
    if (!flist || flist.length === 0) return;
    const f = flist[0];
    setFile(f);
    if (f.type.startsWith("image/")) {
      const url = URL.createObjectURL(f);
      setPreview(url);
    } else {
      setPreview(null);
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    onFiles(e.dataTransfer.files);
  };

  const handleBrowse = () => inputRef.current?.click();

  const handleClear = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className={`upload-card ${dragOver ? "drag-over" : ""}`}>
      <div
        className="upload-drop"
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <h3>Upload Media for Analysis</h3>
        <p className="muted">Drag & drop image or video here</p>

        {file ? (
          <div className="upload-preview">
            {preview ? <img src={preview} alt={file.name} /> : <div className="file-icon">📄</div>}
            <div className="file-meta">
              <div className="file-name">{file.name}</div>
              <div className="file-actions">
                <button className="btn" onClick={handleBrowse}>Replace</button>
                <button className="btn ghost" onClick={handleClear}>Remove</button>
                <button className="btn primary">Analyze</button>
              </div>
            </div>
          </div>
        ) : (
          <>
            <button className="btn" onClick={handleBrowse}>Click to browse files</button>
            <div className="accept-text">(Supported formats: JPEG, PNG, GIF, MP4, AVI)</div>
          </>
        )}

  <label htmlFor="file-input" className="label-hidden">Upload file</label>
  <input id="file-input" className="upload-hidden" ref={inputRef} type="file" accept={ACCEPT} onChange={(e)=>onFiles(e.target.files as FileList)} />
      </div>
    </div>
  );
};

export default UploadCard;
