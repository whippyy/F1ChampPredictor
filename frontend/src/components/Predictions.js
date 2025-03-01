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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Fetch races when component mounts
  useEffect(() => {
    fetch("http://127.0.0.1:8000/races")
      .then((res) => res.json())
      .then((data) => setRaces(data))
      .catch((err) => console.error("Error fetching races:", err));
  }, []);

  // Handle race selection change
  const handleRaceChange = (e) => {
    setSelectedRace(e.target.value);
  };

  // Handle race prediction request
  const handlePredict = () => {
    if (!selectedRace) {
      setError("Please select a race first!");
      return;
    }

    setLoading(true);
    setError("");

    fetch("http://127.0.0.1:8000/predict-race", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ circuit_id: selectedRace }),
    })
      .then((res) => res.json())
      .then((data) => {
        setPredictions(data.predictions || []);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error predicting race:", err);
        setError("Failed to fetch predictions.");
        setLoading(false);
      });
  };

  return (
    <div className="prediction-container">
      <h1 className="title">🏁 Race Predictions</h1>

      {/* Race Selection Dropdown */}
      <select
        value={selectedRace}
        onChange={handleRaceChange}
        className="race-select"
      >
        <option value="">Select a Race</option>
        {races.map((race) => (
          <option key={race.circuit_id} value={race.circuit_id}>
            {race.name}
          </option>
        ))}
      </select>

      <button onClick={handlePredict} className="predict-btn" disabled={loading}>
        {loading ? "Predicting..." : "Predict Race"}
      </button>

      {error && <p className="error-msg">{error}</p>}

      {/* Display Predictions */}
      <div className="predictions-list">
        {predictions.length > 0 ? (
          predictions.map((driver, index) => (
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
          ))
        ) : (
          !loading && <p className="no-predictions">No predictions available.</p>
        )}
      </div>
    </div>
  );
};

export default Prediction;


