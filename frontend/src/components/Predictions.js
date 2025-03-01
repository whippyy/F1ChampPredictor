import React, { useState, useEffect } from "react";
import "./Predictions.css";

const teamColors = {
  Mercedes: "#00D2BE",
  RedBull: "#1E41FF",
  Ferrari: "#DC0000",
  McLaren: "#FF8700",
  AstonMartin: "#006F62",
  Alpine: "#0090FF",
  Haas: "#FFFFFF",
  AlphaTauri: "#2B4562",
  AlfaRomeo: "#900000",
  Williams: "#005AFF",
};

const Prediction = () => {
  const [races, setRaces] = useState([]);
  const [selectedRace, setSelectedRace] = useState("");
  const [predictions, setPredictions] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/races")
      .then((res) => res.json())
      .then((data) => setRaces(data))
      .catch((err) => console.error("Error fetching races:", err));
  }, []);

  const handlePredict = () => {
    if (!selectedRace) return;
    fetch("http://127.0.0.1:8000/predict-race", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ circuit_id: selectedRace }),
    })
      .then((res) => res.json())
      .then((data) => setPredictions(data.predictions))
      .catch((err) => console.error("Error predicting race:", err));
  };

  return (
    <div className="prediction-container">
      <h1 className="title">🏁 Race Predictions</h1>

      {/* Race Selection Dropdown */}
      <select
        value={selectedRace}
        onChange={(e) => setSelectedRace(e.target.value)}
        className="race-select"
      >
        <option value="">Select a Race</option>
        {races.map((race) => (
          <option key={race.circuit_id} value={race.circuit_id}>
            {race.name}
          </option>
        ))}
      </select>

      <button onClick={handlePredict} className="predict-btn">
        Predict Race
      </button>

      {/* Display Predictions */}
      <div className="predictions-list">
        {predictions.map((driver, index) => (
          <div
            key={driver.driver_id || index}
            className="driver-card"
            style={{ backgroundColor: teamColors[driver.team] || "#444" }}
          >
            <img src={driver.image} alt={driver.name} className="driver-image" />
            <div className="driver-info">
              <h2>{index + 1}. {driver.name}</h2>
              <p>{driver.team}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Prediction;

