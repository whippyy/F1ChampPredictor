import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import Dashboard from './components/Dashboard';
import Predictions from './components/Predictions';
import RacePoints from './components/RacePoints';
import './App.css';

const App = () => {
  const [loading, setLoading] = useState(false);

  return (
    <Router>
      <div className="app-container">
        <Header />
        
        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/" element={
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <Dashboard />
              </motion.div>
            } />
            <Route path="/predict" element={<Predictions />} />
            <Route path="/race-points" element={<RacePoints />} />
          </Routes>
        </AnimatePresence>

        <Footer />
      </div>
    </Router>
  );
};

const Header = () => (
  <header className="app-header">
    <div className="header-content">
      <div className="logo-container">
        <motion.img 
          src="/f1-logo.png" 
          alt="F1 Logo"
          className="logo"
          whileHover={{ scale: 1.05 }}
        />
        <h1 className="app-title">F1 INSIGHTS</h1>
      </div>
      
      <nav className="main-nav">
        <NavLink 
          to="/" 
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          DASHBOARD
        </NavLink>
        <NavLink 
          to="/predict" 
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          PREDICTIONS
        </NavLink>
        <NavLink 
          to="/race-points" 
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          RACE POINTS
        </NavLink>
      </nav>
    </div>
  </header>
);

const Footer = () => (
  <footer className="app-footer">
    <div className="footer-content">
      <p>© {new Date().getFullYear()} F1 INSIGHTS | Powered by <a href="https://ergast.com/mrd/" target="_blank" rel="noopener noreferrer">Ergast API</a></p>
      <div className="footer-links">
        <a href="#">Terms</a>
        <a href="#">Privacy</a>
        <a href="#">Contact</a>
      </div>
    </div>
  </footer>
);

export default App;