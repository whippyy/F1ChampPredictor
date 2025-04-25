import React, { useState } from "react";
import { BrowserRouter as Router, Route, Routes, NavLink, useLocation } from "react-router-dom";
import Dashboard from "./components/Dashboard";
import Prediction from "./components/Predictions";
import RacePoints from "./components/Racepoints";
import "./App.css";

const AppContent = () => {
  const location = useLocation();
  const [loading, setLoading] = useState(false);

  const pageBackground =
    location.pathname === "/"
      ? "from-gray-900 to-gray-800"
      : location.pathname === "/predict"
      ? "from-black to-red-900"
      : "from-gray-800 to-gray-700";

  return (
    <div className={`min-h-screen bg-gradient-to-b ${pageBackground} text-white`}>
      <header className="bg-gray-900 shadow-lg sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex flex-col md:flex-row items-center justify-between">
            <div className="flex items-center space-x-4 mb-4 md:mb-0">
              <img 
                src="/f1-logo.png" 
                alt="F1 Logo" 
                className="h-10 md:h-12 logo-img"
              />
              <h1 className="text-xl md:text-2xl font-bold">F1 Dashboard</h1>
            </div>
            <nav className="flex space-x-1 md:space-x-6">
              <NavLink 
                to="/" 
                className={({ isActive }) => 
                  `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive ? 'bg-red-600 text-white nav-link-active' : 'text-gray-300 hover:text-white hover:bg-gray-800'
                  }`
                }
              >
                Dashboard
              </NavLink>
              <NavLink 
                to="/predict" 
                className={({ isActive }) => 
                  `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive ? 'bg-red-600 text-white nav-link-active' : 'text-gray-300 hover:text-white hover:bg-gray-800'
                  }`
                }
              >
                Predict Race
              </NavLink>
              <NavLink 
                to="/race-points" 
                className={({ isActive }) => 
                  `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive ? 'bg-red-600 text-white nav-link-active' : 'text-gray-300 hover:text-white hover:bg-gray-800'
                  }`
                }
              >
                Race Points
              </NavLink>
            </nav>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        {loading && <div className="loading-spinner mx-auto my-10" />}
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Prediction />} />
          <Route path="/race-points" element={<RacePoints />} />
        </Routes>
      </main>

      <footer className="bg-gray-900 py-6 mt-8">
        <div className="container mx-auto px-6 text-center text-gray-400 text-sm footer-text">
          <p>© {new Date().getFullYear()} F1 Dashboard. Data sourced from <a href="https://ergast.com/mrd/" target="_blank" rel="noreferrer">public APIs</a>.</p>
        </div>
      </footer>
    </div>
  );
};

const App = () => (
  <Router>
    <AppContent />
  </Router>
);

export default App;
