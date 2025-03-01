import React, { useEffect, useState } from "react";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import Dashboard from "./Dashboard";
import Prediction from "./Prediction";
import RacePoints from "./RacePoints";

const App = () => {
  return (
    <Router>
      <div className="p-6 bg-gray-100 min-h-screen">
        <nav className="flex space-x-6 mb-6 bg-white shadow-md p-4 rounded-lg">
          <Link to="/" className="text-blue-600 font-semibold hover:underline">Dashboard</Link>
          <Link to="/predict" className="text-blue-600 font-semibold hover:underline">Predict Race</Link>
          <Link to="/race-points" className="text-blue-600 font-semibold hover:underline">Race Points</Link>
        </nav>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Prediction />} />
          <Route path="/race-points" element={<RacePoints />} />
        </Routes>
      </div>
    </Router>
  );
};

