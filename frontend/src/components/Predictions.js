import React, { useState, useEffect } from "react";

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
    <div className="p-4">
      <h1 className="text-2xl font-bold">🏁 Predict a Race</h1>
      <select
        className="border p-2 mt-4"
        onChange={(e) => setSelectedRace(e.target.value)}
      >
        <option value="">Select a Race</option>
        {races.map((race) => (
          <option key={race.raceId} value={race.circuitId}>
            {race.name}
          </option>
        ))}
      </select>
      <button onClick={handlePredict} className="bg-blue-500 text-white p-2 mt-4">
        Predict
      </button>

      <h2 className="text-xl mt-4">Predictions</h2>
      <ul>
        {predictions.map((p, index) => (
          <li key={index} className="border p-2 mt-2">
            {p.driver} - Position: {p.predicted_race_position}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Prediction;
