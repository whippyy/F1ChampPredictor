import React, { useEffect, useState } from "react";

const RacePoints = () => {
  const [raceResults, setRaceResults] = useState([]);
  const [selectedRace, setSelectedRace] = useState("");
  const [races, setRaces] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/races")
      .then((res) => res.json())
      .then((data) => setRaces(data))
      .catch((err) => console.error("Error fetching races:", err));
  }, []);

  const handleFetchResults = () => {
    if (!selectedRace) return;
    fetch(`http://127.0.0.1:8000/race-results/${selectedRace}`)
      .then((res) => res.json())
      .then((data) => setRaceResults(data))
      .catch((err) => console.error("Error fetching race results:", err));
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold">🏆 Race Results</h1>
      <select
        className="border p-2 mt-4"
        onChange={(e) => setSelectedRace(e.target.value)}
      >
        <option value="">Select a Race</option>
        {races.map((race) => (
          <option key={race.raceId} value={race.raceId}>
            {race.name}
          </option>
        ))}
      </select>
      <button onClick={handleFetchResults} className="bg-green-500 text-white p-2 mt-4">
        Fetch Results
      </button>

      <h2 className="text-xl mt-4">Race Standings</h2>
      <ul>
        {raceResults.map((result, index) => (
          <li key={index} className="border p-2 mt-2">
            {result.driver} - Points: {result.points}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default RacePoints;