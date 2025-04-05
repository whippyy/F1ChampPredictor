import React from "react";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import Dashboard from "./components/Dashboard";
import Prediction from "./components/Predictions";
import RacePoints from "./components/Racepoints";
import "./App.css";

const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white">
        <header className="bg-gray-900 shadow-lg">
          <div className="container mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <img src="/f1-logo.png" alt="F1 Logo" className="h-12" />
                <h1 className="text-2xl font-bold">F1 Dashboard</h1>
              </div>
              <nav className="flex space-x-6">
                <Link to="/" className="hover:text-red-500 font-medium transition-colors">Dashboard</Link>
                <Link to="/predict" className="hover:text-red-500 font-medium transition-colors">Predict Race</Link>
                <Link to="/race-points" className="hover:text-red-500 font-medium transition-colors">Race Points</Link>
              </nav>
            </div>
          </div>
        </header>

        <main className="container mx-auto px-6 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Prediction />} />
            <Route path="/race-points" element={<RacePoints />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;

