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

const driverImages = (driverCode) => driverCode ? `https://media.formula1.com/image/upload/f_auto,c_limit,q_auto,w_1320/fom-website/drivers/2024Drivers/${driverCode}` : "https://via.placeholder.com/80";
const teamLogos = (teamCode) => teamCode ? `https://media.formula1.com/image/upload/f_auto,c_limit,q_auto,w_1320/fom-website/teams/2024Teams/${teamCode}` : "https://via.placeholder.com/50";

const Prediction = () => {
  const [races, setRaces] = useState([]);
  const [selectedRace, setSelectedRace] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/races?season=2024")
      .then((res) => res.json())
      .then((data) => {
        console.log("Fetched races:", data);
        setRaces([...data.data]);
      })
      .catch((err) => console.error("Error fetching races:", err));
  }, []);

  const handleRaceChange = (e) => {
    setSelectedRace(e.target.value);
  };

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
        console.log("Fetched predictions:", data);
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

      <select
        value={selectedRace}
        onChange={(e) => setSelectedRace(e.target.value)}
        className="race-select"
      >
        <option value="">Select a Race</option>
        {races.map((race) => (
          <option key={race.raceId} value={race.circuitId}>
            {race.name} ({race.year})
          </option>
        ))}
      </select>

      <button onClick={handlePredict} className="predict-btn" disabled={loading}>
        {loading ? "Predicting..." : "Predict Race"}
      </button>

      {error && <p className="error-msg">{error}</p>}

      <div className="predictions-list">
        {predictions.length > 0 ? (
          predictions.map((driver, index) => (
            <div
              key={driver.driver_id || index}
              className="driver-card"
              style={{ backgroundColor: teamColors[driver.team] || "#444" }}
            >
              <img src={driverImages(driver.driver_code)} alt={driver.name} className="driver-headshot" />
              <div className="driver-info">
                <h2>{driver.name}</h2>
                <p>{driver.team}</p>
              </div>
              <div className="team-logo-container">
                <span className="position-number">{index + 1}</span>
                <img src={teamLogos(driver.team_code)} alt={driver.team} className="team-logo" />
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




