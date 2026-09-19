import React from 'react';
import { Database, FileText, CheckCircle } from 'lucide-react';

export default function Dashboard() {
  return (
    <div className="page-animate">
      <h1>Synthara Datasets</h1>
      <p style={{ color: 'var(--text-secondary)' }}>Overview of your synthetic datasets for African financial AI.</p>
      
      <div className="metrics-grid">
        <div className="glass-panel metric-card">
          <div className="metric-label">Generated Datasets</div>
          <div className="metric-value pulse">2</div>
          <div style={{ color: 'var(--accent-success)' }}>+1 this week</div>
        </div>
        <div className="glass-panel metric-card">
          <div className="metric-label">Total Rows Generated</div>
          <div className="metric-value">15,000</div>
        </div>
        <div className="glass-panel metric-card">
          <div className="metric-label">Avg Fidelity Score</div>
          <div className="metric-value" style={{ color: 'var(--accent-primary)' }}>82%</div>
        </div>
      </div>
      
      <h2 style={{ marginTop: '3rem' }}>Recent Generations</h2>
      <div className="glass-panel" style={{ padding: '0' }}>
        <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
              <th style={{ padding: '1rem' }}>Dataset Name</th>
              <th style={{ padding: '1rem' }}>Model</th>
              <th style={{ padding: '1rem' }}>Rows</th>
              <th style={{ padding: '1rem' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <td style={{ padding: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FileText size={16} color="var(--accent-primary)" />
                synthetic_gaussian_copula (Credit)
              </td>
              <td style={{ padding: '1rem' }}>Gaussian Copula</td>
              <td style={{ padding: '1rem' }}>5,000</td>
              <td style={{ padding: '1rem', color: 'var(--accent-success)' }}>
                <CheckCircle size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                Complete
              </td>
            </tr>
            <tr>
              <td style={{ padding: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FileText size={16} color="var(--accent-primary)" />
                synthetic_ctgan (Fraud)
              </td>
              <td style={{ padding: '1rem' }}>CTGAN</td>
              <td style={{ padding: '1rem' }}>10,000</td>
              <td style={{ padding: '1rem', color: 'var(--accent-success)' }}>
                <CheckCircle size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                Complete
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
