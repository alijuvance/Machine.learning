import React from 'react';
import { UploadCloud, Zap } from 'lucide-react';

export default function Generator() {
  return (
    <div className="page-animate">
      <h1>New Synthetic Generation</h1>
      <p style={{ color: 'var(--text-secondary)' }}>Upload a source dataset to profile and generate synthetic data.</p>
      
      <div style={{ marginTop: '2rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        <div className="glass-panel">
          <h2>1. Source Data</h2>
          <div className="file-upload-zone">
            <UploadCloud size={48} color="var(--accent-primary)" style={{ marginBottom: '1rem' }} />
            <h3>Drag & Drop CSV File</h3>
            <p style={{ color: 'var(--text-secondary)', marginTop: '8px' }}>or click to browse</p>
          </div>
        </div>
        
        <div className="glass-panel">
          <h2>2. Configuration</h2>
          <div className="form-group">
            <label className="form-label">Model Type</label>
            <select className="form-select">
              <option value="gaussian_copula">Gaussian Copula (Fast)</option>
              <option value="ctgan">CTGAN (High Fidelity)</option>
              <option value="tvae">TVAE (Balanced)</option>
            </select>
          </div>
          
          <div className="form-group">
            <label className="form-label">Target Column (For ML Benchmark)</label>
            <input type="text" className="form-input" placeholder="e.g. isFraud, default" defaultValue="isFraud" />
          </div>
          
          <div className="form-group">
            <label className="form-label">Rows to Generate</label>
            <input type="number" className="form-input" defaultValue={10000} />
          </div>
          
          <button className="btn btn-accent" style={{ width: '100%', marginTop: '1rem' }}>
            <Zap size={18} />
            Start Generation Pipeline
          </button>
        </div>
      </div>
    </div>
  );
}
