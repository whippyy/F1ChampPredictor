import React, { useState, useEffect } from "react";

const RacePoints = () => {
  const [races, setRaces] = useState([]);
  const [selectedRace, setSelectedRace] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchRaces = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8000/races?season=2024");
        const data = await response.json();
        setRaces(data.data || []);
        if (data.data && data.data.length > 0) {
          setSelectedRace(data.data[0].raceId);
        }
      } catch (error) {
        console.error("Error fetching races:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchRaces();
  }, []);

  useEffect(() => {
    if (selectedRace) {
      const fetchResults = async () => {
        try {
          const response = await fetch(`http://127.0.0.1:8000/results?raceId=${selectedRace}`);
          const data = await response.json();
          setResults(data.data || []);
        } catch (error) {
          console.error("Error fetching results:", error);
        }
      };

      fetchResults();
    }
  }, [selectedRace]);

  const handleRaceChange = (e) => {
    setSelectedRace(e.target.value);
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-center mb-8 text-white">🏅 Race Points</h1>
      
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-red-500"></div>
        </div>
      ) : (
        <div className="bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-700">
          <div className="mb-6">
            <label htmlFor="race-select" className="block text-sm font-medium text-gray-300 mb-2">
              Select Race:
            </label>
            <select
              id="race-select"
              value={selectedRace || ''}
              onChange={handleRaceChange}
              className="bg-gray-700 border border-gray-600 text-white rounded-lg focus:ring-red-500 focus:border-red-500 block w-full p-2.5"
            >
              {races.map((race) => (
                <option key={race.raceId} value={race.raceId}>
                  {race.name} ({new Date(race.date).toLocaleDateString()})
                </option>
              ))}
            </select>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left text-gray-400">
              <thead className="text-xs text-gray-300 uppercase bg-gray-700">
                <tr>
                  <th scope="col" className="px-6 py-3">Position</th>
                  <th scope="col" className="px-6 py-3">Driver</th>
                  <th scope="col" className="px-6 py-3">Team</th>
                  <th scope="col" className="px-6 py-3">Points</th>
                </tr>
              </thead>
              <tbody>
                {results.length > 0 ? (
                  results.map((result, index) => (
                    <tr key={index} className="border-b border-gray-700 hover:bg-gray-700">
                      <td className="px-6 py-4 font-medium text-white">
                        {result.position === '1' ? '🥇' : result.position === '2' ? '🥈' : result.position === '3' ? '🥉' : result.position}
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center">
                          {result.driverImage && (
                            <img
                              src={result.driverImage}
                              alt={`${result.forename} ${result.surname}`}
                              className="w-8 h-8 rounded-full mr-3"
                              onError={(e) => {
                                e.target.onerror = null;
                                e.target.src = '/driver-placeholder.png';
                              }}
                            />
                          )}
                          <span className="text-white">
                            {result.forename} {result.surname}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center">
                          {result.teamLogo && (
                            <img
                              src={result.teamLogo}
                              alt={result.team}
                              className="w-6 h-6 mr-2"
                              onError={(e) => {
                                e.target.onerror = null;
                                e.target.src = '/team-placeholder.png';
                              }}
                            />
                          )}
                          {result.team}
                        </div>
                      </td>
                      <td className="px-6 py-4 font-semibold text-red-500">{result.points}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="4" className="px-6 py-4 text-center text-gray-400">
                      No results available for this race.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default RacePoints;