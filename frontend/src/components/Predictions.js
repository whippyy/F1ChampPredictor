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

const driverImages = {
  "Max Verstappen": "https://media.formula1.com/d_driver_max_verstappen.png",
  "Sergio Perez": "https://media.formula1.com/d_driver_sergio_perez.png",
  "Lewis Hamilton": "https://media.formula1.com/d_driver_lewis_hamilton.png",
  "George Russell": "https://media.formula1.com/d_driver_george_russell.png",
  "Charles Leclerc": "https://media.formula1.com/d_driver_charles_leclerc.png",
  "Carlos Sainz": "https://media.formula1.com/d_driver_carlos_sainz.png",
  "Lando Norris": "https://media.formula1.com/d_driver_lando_norris.png",
  "Oscar Piastri": "https://media.formula1.com/d_driver_oscar_piastri.png",
  "Fernando Alonso": "https://media.formula1.com/d_driver_fernando_alonso.png",
  "Lance Stroll": "https://media.formula1.com/d_driver_lance_stroll.png",
  "Pierre Gasly": "https://media.formula1.com/d_driver_pierre_gasly.png",
  "Esteban Ocon": "https://media.formula1.com/d_driver_esteban_ocon.png",
  "Kevin Magnussen": "https://media.formula1.com/d_driver_kevin_magnussen.png",
  "Nico Hulkenberg": "https://media.formula1.com/d_driver_nico_hulkenberg.png",
  "Yuki Tsunoda": "https://media.formula1.com/d_driver_yuki_tsunoda.png",
  "Daniel Ricciardo": "https://media.formula1.com/d_driver_daniel_ricciardo.png",
  "Valtteri Bottas": "https://media.formula1.com/d_driver_valtteri_bottas.png",
  "Zhou Guanyu": "https://media.formula1.com/d_driver_zhou_guanyu.png",
  "Alexander Albon": "https://media.formula1.com/d_driver_alexander_albon.png",
  "Logan Sargeant": "https://media.formula1.com/d_driver_logan_sargeant.png",
};

const teamLogos = {
  Mercedes: "https://media.formula1.com/t_team_mercedes.png",
  RedBull: "https://media.formula1.com/t_team_redbull.png",
  Ferrari: "https://media.formula1.com/t_team_ferrari.png",
  McLaren: "https://media.formula1.com/t_team_mclaren.png",
  AstonMartin: "https://media.formula1.com/t_team_astonmartin.png",
  Alpine: "https://media.formula1.com/t_team_alpine.png",
  Haas: "https://media.formula1.com/t_team_haas.png",
  AlphaTauri: "https://media.formula1.com/t_team_alphatauri.png",
  AlfaRomeo: "https://media.formula1.com/t_team_alfaromeo.png",
  Williams: "https://media.formula1.com/t_team_williams.png",
};

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
              <img src={driverImages[driver.name] || "https://via.placeholder.com/80"} alt={driver.name} className="driver-headshot" />
              <div className="driver-info">
                <h2>{driver.name}</h2>
                <p>{driver.team}</p>
              </div>
              <div className="team-logo-container">
                <span className="position-number">{index + 1}</span>
                <img src={teamLogos[driver.team] || "https://via.placeholder.com/50"} alt={driver.team} className="team-logo" />
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



