import React, { useState, useEffect } from "react";
import "./Predictions.css";

const Prediction = () => {
  const [drivers, setDrivers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [prediction, setPrediction] = useState([]);

  useEffect(() => {
    const fetchDrivers = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8000/drivers?with_images=true");
        const data = await response.json();
        setDrivers(data.data || []);
      } catch (error) {
        console.error("Error fetching drivers:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchDrivers();
  }, []);

  const handlePrediction = (position) => {
    return (driverId) => {
      setPrediction(prev => {
        const newPrediction = [...prev];
        newPrediction[position - 1] = driverId;
        return newPrediction;
      });
    };
  };

  return (
    <div className="prediction-container p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-center mb-8 text-white">🏆 Race Prediction</h1>
      
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-red-500"></div>
        </div>
      ) : (
        <div className="w-full space-y-4">
          {[1, 2, 3, 4, 5].map((position) => (
            <div key={position} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <h2 className="text-xl font-semibold mb-4 text-white">
                {position === 1 ? '🥇' : position === 2 ? '🥈' : position === 3 ? '🥉' : `P${position}`} Position
              </h2>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
                {drivers.map((driver) => (
                  <div
                    key={driver.driverId}
                    onClick={() => handlePrediction(position)(driver.driverId)}
                    className={`driver-card bg-gray-700 rounded-lg p-2 cursor-pointer transition-all ${
                      prediction[position - 1] === driver.driverId ? 'ring-2 ring-red-500' : 'hover:bg-gray-600'
                    }`}
                  >
                    <div className="flex items-center">
                      <div className="relative">
                        {driver.imageUrl ? (
                          <img
                            src={driver.imageUrl}
                            alt={`${driver.forename} ${driver.surname}`}
                            className="w-12 h-12 rounded-full object-cover"
                            onError={(e) => {
                              e.target.onerror = null;
                              e.target.src = '/driver-placeholder.png';
                            }}
                          />
                        ) : (
                          <div className="w-12 h-12 rounded-full bg-gray-600 flex items-center justify-center">
                            <span className="text-xs text-gray-400">No image</span>
                          </div>
                        )}
                        {driver.teamLogo && (
                          <img
                            src={driver.teamLogo}
                            alt={driver.team}
                            className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full border border-gray-800"
                            onError={(e) => {
                              e.target.onerror = null;
                              e.target.src = '/team-placeholder.png';
                            }}
                          />
                        )}
                      </div>
                      <div className="ml-3">
                        <h3 className="text-sm font-semibold text-white">
                          {driver.forename[0]}. {driver.surname}
                        </h3>
                        <p className="text-xs text-gray-300">{driver.team}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
          
          <button
            className="mt-6 w-full bg-red-500 hover:bg-red-600 text-white font-bold py-3 px-4 rounded-lg transition-colors"
            disabled={prediction.length < 5}
          >
            Submit Prediction
          </button>
        </div>
      )}
    </div>
  );
};

export default Prediction;



