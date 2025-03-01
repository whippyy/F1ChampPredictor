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

export default App;



const Prediction = () => {
  const [prediction, setPrediction] = useState(null);
  const [raceId, setRaceId] = useState("");

  const handlePredict = () => {
    fetch(`http://127.0.0.1:8000/predict-race/${raceId}`)
      .then((res) => res.json())
      .then((data) => setPrediction(data))
      .catch((err) => console.error("Error predicting race:", err));
  };

  return (
    <div className="bg-white shadow-lg p-6 rounded-lg">
      <h1 className="text-3xl font-bold text-gray-800">🔮 Predict Race</h1>
      <input
        type="text"
        className="border p-2 mt-4 w-full rounded-lg"
        placeholder="Enter Race ID"
        value={raceId}
        onChange={(e) => setRaceId(e.target.value)}
      />
      <button
        onClick={handlePredict}
        className="bg-blue-500 text-white px-6 py-2 mt-4 rounded-lg hover:bg-blue-600 transition"
      >
        Predict
      </button>

      {prediction && (
        <div className="mt-6 bg-gray-50 p-4 rounded-lg">
          <h2 className="text-xl font-semibold">Predicted Winner: {prediction.driver}</h2>
          <p>Track: {prediction.track}</p>
          <p>Position: {prediction.predicted_race_position}</p>
        </div>
      )}
    </div>
  );
};

