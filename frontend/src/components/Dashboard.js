import React, { useEffect, useState } from "react";

const Dashboard = () => {
  const [races, setRaces] = useState([]);
  const [drivers, setDrivers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch races
        const racesRes = await fetch("http://127.0.0.1:8000/races?season=2024");
        const racesData = await racesRes.json();
        setRaces(racesData.data || []);

        // Fetch drivers
        const driversRes = await fetch("http://127.0.0.1:8000/drivers");
        const driversData = await driversRes.json();
        setDrivers(driversData.data || []);

      } catch (err) {
        console.error("Error fetching data:", err);
        setError("Failed to load data. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-center mb-6">🏎️ F1 Dashboard</h1>

      {loading && <p className="text-center text-gray-600">Loading...</p>}
      {error && <p className="text-center text-red-500">{error}</p>}

      {/* Upcoming Races Section */}
      <div className="bg-white shadow-md rounded-lg p-4 mb-6">
        <h2 className="text-xl font-semibold border-b pb-2">📅 Upcoming Races</h2>
        <ul className="mt-3 space-y-2">
          {races.length > 0 ? (
            races.map((race) => (
              <li key={race.raceId} className="border p-3 rounded-lg shadow-sm">
                <span className="font-medium">{race.name}</span> - <span className="text-gray-600">{race.date}</span>
              </li>
            ))
          ) : (
            <p className="text-gray-500">No races available.</p>
          )}
        </ul>
      </div>

      {/* Drivers Section */}
      <div className="bg-white shadow-md rounded-lg p-4">
        <h2 className="text-xl font-semibold border-b pb-2">🏁 Drivers</h2>
        <ul className="mt-3 grid grid-cols-2 gap-3">
          {drivers.length > 0 ? (
            drivers.map((driver) => (
              <li key={driver.driverId} className="border p-3 rounded-lg shadow-sm">
                {driver.forename} {driver.surname}
              </li>
            ))
          ) : (
            <p className="text-gray-500">No drivers available.</p>
          )}
        </ul>
      </div>
    </div>
  );
};

export default Dashboard;

