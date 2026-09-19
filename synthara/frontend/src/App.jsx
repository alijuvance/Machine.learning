import React from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Database, Zap, Activity, Settings, Cpu } from 'lucide-react';
import './index.css';

import Dashboard from './pages/Dashboard';
import Generator from './pages/Generator';
import Reports from './pages/Reports';

function Sidebar() {
  const location = useLocation();
  
  return (
    <div className="sidebar">
      <div className="brand-logo">
        <div className="brand-icon">
          <Zap size={20} />
        </div>
        <span>Synthara</span>
      </div>
      
      <nav>
        <Link to="/" className={`nav-item ${location.pathname === '/' ? 'active' : ''}`}>
          <Database size={18} />
          <span>Dashboard</span>
        </Link>
        <Link to="/generate" className={`nav-item ${location.pathname === '/generate' ? 'active' : ''}`}>
          <Cpu size={18} />
          <span>Generator</span>
        </Link>
        <Link to="/reports" className={`nav-item ${location.pathname.includes('/reports') ? 'active' : ''}`}>
          <Activity size={18} />
          <span>Reports</span>
        </Link>
        <Link to="#" className="nav-item">
          <Settings size={18} />
          <span>Settings</span>
        </Link>
      </nav>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="app-container">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/generate" element={<Generator />} />
            <Route path="/reports" element={<Reports />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
