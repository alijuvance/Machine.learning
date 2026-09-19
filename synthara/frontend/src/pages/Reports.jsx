import React from 'react';

export default function Reports() {
  return (
    <div className="page-animate">
      <h1>Quality & Benchmark Reports</h1>
      <p style={{ color: 'var(--text-secondary)' }}>Detailed analysis of fidelity, privacy, and downstream ML performance.</p>
      
      <div className="glass-panel" style={{ marginTop: '2rem', textAlign: 'center', padding: '4rem' }}>
        <h2 style={{ color: 'var(--accent-primary)' }}>Report Viewer Coming Soon</h2>
        <p style={{ color: 'var(--text-secondary)' }}>
          This page will integrate Recharts to display interactive distributions, 
          correlation matrices, and the TSTR (Train Synthetic, Test Real) benchmark comparisons.
        </p>
      </div>
    </div>
  );
}
