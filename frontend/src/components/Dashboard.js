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

        // Fetch drivers with images
        const driversRes = await fetch("http://127.0.0.1:8000/drivers?with_images=true");
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
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-center mb-8 text-red-500">🏎️ F1 2024 Dashboard</h1>

      {loading && (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-red-500"></div>
        </div>
      )}
      
      {error && <p className="text-center text-red-500 p-4 bg-red-100 rounded-lg">{error}</p>}

      {/* Upcoming Races Section */}
      <div className="bg-gray-800 shadow-lg rounded-xl p-6 mb-8 border border-gray-700">
        <h2 className="text-2xl font-bold mb-4 text-white border-b border-gray-600 pb-2 flex items-center">
          <span className="bg-red-500 text-white p-2 rounded-lg mr-3">📅</span>
          Upcoming Races
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {races.length > 0 ? (
            races.map((race) => (
              <div key={race.raceId} className="bg-gray-700 rounded-lg p-4 hover:bg-gray-600 transition-colors">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="font-bold text-lg text-white">{race.name}</h3>
                    <p className="text-gray-300">{race.circuitName}</p>
                  </div>
                  <span className="bg-red-500 text-white px-3 py-1 rounded-full text-sm">
                    {new Date(race.date).toLocaleDateString()}
                  </span>
                </div>
                <div className="mt-3 pt-3 border-t border-gray-600">
                  <p className="text-gray-400 text-sm">{race.location}, {race.country}</p>
                </div>
              </div>
            ))
          ) : (
            !loading && <p className="text-gray-400 col-span-3 text-center py-8">No races available.</p>
          )}
        </div>
      </div>

      {/* Drivers Section */}
      <div className="bg-gray-800 shadow-lg rounded-xl p-6 border border-gray-700">
        <h2 className="text-2xl font-bold mb-4 text-white border-b border-gray-600 pb-2 flex items-center">
          <span className="bg-red-500 text-white p-2 rounded-lg mr-3">🏁</span>
          Drivers 2024
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {drivers.length > 0 ? (
            drivers.map((driver) => (
              <div key={driver.driverId} className="bg-gray-700 rounded-lg overflow-hidden hover:shadow-lg transition-shadow">
                <div className="relative">
                  {driver.imageUrl ? (
                    <img 
                      src={driver.imageUrl} 
                      alt={`${driver.forename} ${driver.surname}`} 
                      className="w-full h-48 object-cover"
                      onError={(e) => {
                        e.target.onerror = null;
                        e.target.src = '/driver-placeholder.png';
                      }}
                    />
                  ) : (
                    <div className="w-full h-48 bg-gray-600 flex items-center justify-center">
                      <span className="text-gray-400">No image</span>
                    </div>
                  )}
                  {driver.teamLogo && (
                    <div className="absolute bottom-2 right-2 bg-black bg-opacity-70 p-1 rounded-full">
                      <img 
                        src={driver.teamLogo} 
                        alt={driver.team} 
                        className="h-8 w-8 object-contain"
                        onError={(e) => {
                          e.target.onerror = null;
                          e.target.src = '/team-placeholder.png';
                        }}
                      />
                    </div>
                  )}
                </div>
                <div className="p-3">
                  <h3 className="font-bold text-white">
                    {driver.forename} <span className="font-extrabold">{driver.surname}</span>
                  </h3>
                  <p className="text-sm text-gray-300">{driver.team}</p>
                  <p className="text-xs text-gray-400">#{driver.number || '--'}</p>
                </div>
              </div>
            ))
          ) : (
            !loading && <p className="text-gray-400 col-span-5 text-center py-8">No drivers available.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;