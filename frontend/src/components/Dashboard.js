import React, { useEffect, useState } from "react";

const Dashboard = () => {
  const [races, setRaces] = useState([]);
  const [drivers, setDrivers] = useState([]);

  useEffect(() => {
    // Fetch races
    fetch("http://127.0.0.1:8000/races")
      .then((res) => res.json())
      .then((data) => setRaces(data))
      .catch((err) => console.error("Error fetching races:", err));

    // Fetch drivers
    fetch("http://127.0.0.1:8000/drivers")
      .then((res) => res.json())
      .then((data) => setDrivers(data))
      .catch((err) => console.error("Error fetching drivers:", err));
  }, []);

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold">🏎️ F1 Dashboard</h1>
      <h2 className="text-xl mt-4">Upcoming Races</h2>
      <ul>
        {races.map((race) => (
          <li key={race.raceId} className="border p-2 mt-2">
            {race.name} - {race.date}
          </li>
        ))}
      </ul>

      <h2 className="text-xl mt-4">Drivers</h2>
      <ul>
        {drivers.map((driver) => (
          <li key={driver.driverId} className="border p-2 mt-2">
            {driver.forename} {driver.surname}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Dashboard;
