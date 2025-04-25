import React, { useState, useEffect } from "react";
import "./Predictions.css";

const Prediction = () => {
  const [drivers, setDrivers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [prediction, setPrediction] = useState([]);
  const [submitted, setSubmitted] = useState(false);

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

  const handleSubmit = () => {
    if (prediction.length === 5) {
      setSubmitted(true);
      // Here you would typically send the prediction to your backend
      setTimeout(() => setSubmitted(false), 3000);
    }
  };

  const getDriverById = (id) => {
    return drivers.find(driver => driver.driverId === id);
  };

  return (
    <div className="prediction-container">
      <div className="text-center mb-10">
        <h1 className="text-3xl font-bold mb-2">🏆 Race Prediction</h1>
        <p className="text-gray-400 max-w-2xl mx-auto">
          Predict the top 5 finishers for the next race. Select one driver for each position.
        </p>
      </div>
      
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="loading-spinner"></div>
        </div>
      ) : submitted ? (
        <div className="bg-gray-800 rounded-lg p-8 max-w-2xl mx-auto text-center">
          <svg className="w-16 h-16 text-green-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
          </svg>
          <h2 className="text-2xl font-bold mb-2">Prediction Submitted!</h2>
          <p className="text-gray-300 mb-6">Your prediction has been successfully recorded.</p>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {prediction.map((driverId, index) => {
              const driver = getDriverById(driverId);
              if (!driver) return null;
              return (
                <div key={index} className="bg-gray-700 rounded-lg p-4">
                  <div className="text-lg font-bold text-red-500 mb-1">P{index + 1}</div>
                  <div className="flex flex-col items-center">
                    {driver.imageUrl ? (
                      <img
                        src={driver.imageUrl}
                        alt={`${driver.forename} ${driver.surname}`}
                        className="w-16 h-16 rounded-full object-cover mb-2"
                      />
                    ) : (
                      <div className="w-16 h-16 rounded-full bg-gray-600 flex items-center justify-center mb-2">
                        <span className="text-xs text-gray-400">No image</span>
                      </div>
                    )}
                    <div className="text-center">
                      <h3 className="font-semibold">
                        {driver.forename[0]}. {driver.surname}
                      </h3>
                      <p className="text-xs text-gray-400">{driver.team}</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          <button
            onClick={() => setSubmitted(false)}
            className="mt-6 btn btn-primary"
          >
            Make New Prediction
          </button>
        </div>
      ) : (
        <div className="w-full max-w-4xl mx-auto space-y-6">
          {[1, 2, 3, 4, 5].map((position) => (
            <div key={position} className="card">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                {position === 1 ? '🥇' : position === 2 ? '🥈' : position === 3 ? '🥉' : `P${position}`}
                <span className="ml-2">Position</span>
              </h2>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
                {drivers.map((driver) => (
                  <div
                    key={driver.driverId}
                    onClick={() => handlePrediction(position)(driver.driverId)}
                    className={`driver-card ${
                      prediction[position - 1] === driver.driverId 
                        ? 'bg-red-900/30 ring-2 ring-red-500' 
                        : 'bg-gray-700 hover:bg-gray-600'
                    }`}
                  >
                    <div className="flex items-center">
                      <div className="relative">
                        {driver.imageUrl ? (
                          <img
                            src={driver.imageUrl}
                            alt={`${driver.forename} ${driver.surname}`}
                            className="w-12 h-12 rounded-full object-cover"
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
                            className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full border-2 border-gray-800"
                          />
                        )}
                      </div>
                      <div className="ml-3">
                        <h3 className="text-sm font-semibold">
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
            onClick={handleSubmit}
            disabled={prediction.filter(Boolean).length < 5}
            className="btn btn-primary w-full mt-8"
          >
            Submit Prediction
          </button>
        </div>
      )}
    </div>
  );
};

export default Prediction;