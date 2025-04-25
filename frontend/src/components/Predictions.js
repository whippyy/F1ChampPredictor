import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import './Predictions.css';

const Predictions = () => {
  const [drivers, setDrivers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [prediction, setPrediction] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDrivers = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/drivers?with_images=true');
        setDrivers(response.data.data || []);
      } catch (err) {
        setError('Failed to load drivers. Please try again later.');
        console.error('Error fetching drivers:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchDrivers();
  }, []);

  const handlePrediction = (position) => (driverId) => {
    setPrediction(prev => {
      const newPrediction = [...prev];
      newPrediction[position - 1] = driverId;
      return newPrediction;
    });
  };

  const getDriverById = (id) => drivers.find(driver => driver.driverId === id);

  const handleSubmit = async () => {
    if (prediction.length < 5) return;
    
    try {
      // Submit prediction to backend
      console.log('Submitting prediction:', prediction);
      // Show success message
    } catch (err) {
      setError('Failed to submit prediction. Please try again.');
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner" />
        <p>Loading drivers...</p>
      </div>
    );
  }

  if (error) {
    return (
      <motion.div 
        className="error-container"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <p>{error}</p>
        <button 
          className="btn btn-primary"
          onClick={() => window.location.reload()}
        >
          Retry
        </button>
      </motion.div>
    );
  }

  return (
    <motion.div 
      className="predictions-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="predictions-header">
        <h2>Race Predictions</h2>
        <p>Select your top 5 finishers for the next race</p>
      </div>

      <div className="positions-grid">
        {[1, 2, 3, 4, 5].map(position => (
          <div key={position} className="position-card">
            <div className="position-header">
              <span className="position-number">{position}</span>
              <h3>
                {position === 1 ? 'Winner' : 
                 position === 2 ? '2nd Place' : 
                 position === 3 ? '3rd Place' : 
                 `${position}th Place`}
              </h3>
            </div>

            <div className="drivers-grid">
              {drivers.map(driver => (
                <motion.div
                  key={driver.driverId}
                  className={`driver-card ${
                    prediction[position - 1] === driver.driverId ? 'selected' : ''
                  }`}
                  onClick={() => handlePrediction(position)(driver.driverId)}
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="driver-image-container">
                    {driver.imageUrl ? (
                      <img 
                        src={driver.imageUrl} 
                        alt={`${driver.forename} ${driver.surname}`}
                        className="driver-image"
                      />
                    ) : (
                      <div className="driver-placeholder">
                        {driver.forename[0]}{driver.surname[0]}
                      </div>
                    )}
                    {driver.teamLogo && (
                      <img 
                        src={driver.teamLogo} 
                        alt={driver.team} 
                        className="team-logo"
                      />
                    )}
                  </div>
                  <div className="driver-info">
                    <h4>{driver.forename[0]}. {driver.surname}</h4>
                    <p>{driver.team}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <motion.button
        className="btn btn-primary submit-btn"
        onClick={handleSubmit}
        disabled={prediction.length < 5}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        Submit Prediction
      </motion.button>
    </motion.div>
  );
};

export default Predictions;