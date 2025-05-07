import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import './Predictions.css';

const TEAM_COLORS = {
  mclaren: '#FF8000',
  mercedes: '#00D2BE',
  red_bull: '#0600EF',
  ferrari: '#DC0000',
  alpine: '#0090FF',
  aston_martin: '#006F62',
  williams: '#005AFF',
  haas: '#FFFFFF',
  rb: '#6692FF',
  sauber: '#52E252',
};

const Predictions = () => {
  const [tracks, setTracks] = useState([]);
  const [selectedTrack, setSelectedTrack] = useState(null);
  const [drivers, setDrivers] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [userPrediction, setUserPrediction] = useState([]);

  useEffect(() => {
    const fetchTracks = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/races?season=2024');
        setTracks(response.data?.data || []);
      } catch (err) {
        setError('Failed to load tracks. Please try again later.');
        console.error('Error fetching tracks:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchTracks();
  }, []);

  useEffect(() => {
    const fetchDrivers = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/drivers?with_images=true');
        setDrivers(response.data?.data || []);
      } catch (err) {
        console.error('Error fetching drivers:', err);
      }
    };
    fetchDrivers();
  }, []);

  const handleTrackSelect = async (track) => {
    if (!track || !track.raceId) {
      setError('Invalid track selected');
      return;
    }
  
    setSelectedTrack(track);
    setLoading(true);
    
    try {
      const response = await axios.get(`http://127.0.0.1:8000/predictions?raceId=${track.raceId}`);
      setPredictions(response.data?.data || []);
      setError(null);
    } catch (err) {
      setError(`Failed to load predictions for ${track.raceName || 'this race'}. Please try again.`);
      console.error('Error fetching predictions:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDriverSelect = (position) => (driverId) => {
    setUserPrediction(prev => {
      const newPrediction = [...prev];
      newPrediction[position - 1] = driverId;
      return newPrediction;
    });
  };

  const handleSubmitPrediction = async () => {
    if (userPrediction.length < 5) {
      setError('Please select drivers for all top 5 positions');
      return;
    }

    try {
      await axios.post('http://127.0.0.1:8000/predictions', {
        raceId: selectedTrack.raceId,
        prediction: userPrediction
      });
      setError(null);
      alert('Prediction submitted successfully!');
    } catch (err) {
      setError('Failed to submit prediction. Please try again.');
      console.error('Error submitting prediction:', err);
    }
  };

  const getDriverById = (id) => drivers.find(driver => driver.driverId === id);

  if (loading && !selectedTrack) {
    return (
      <div className="loading-container">
        <div className="loading-spinner" />
        <p>Loading tracks...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <p>{error}</p>
        <button className="btn btn-primary" onClick={() => window.location.reload()}>
          Retry
        </button>
      </div>
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
        <h1>F1 RACE PREDICTIONS</h1>
        <p>Select a track and predict the top 5 finishers</p>
      </div>

      <AnimatePresence>
        {!selectedTrack ? (
          <motion.div 
            className="tracks-selection"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <h2>Select a Race</h2>
            <div className="tracks-grid">
              {tracks.map(track => (
                <motion.div
                  key={track.raceId}
                  className="track-card"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => handleTrackSelect(track)}
                >
                  <div className="track-image-container">
                    {track.circuitImage ? (
                      <img src={track.circuitImage} alt={track.circuitName} className="track-image" />
                    ) : (
                      <div className="track-placeholder">
                        {track.circuitName?.split(' ').map(word => word[0]).join('') || 'TR'}
                      </div>
                    )}
                  </div>
                  <div className="track-info">
                    <h3>{track.raceName}</h3>
                    <p>{track.circuitName}</p>
                    <p className="track-date">
                      {new Date(track.date).toLocaleDateString('en-US', {
                        month: 'short', day: 'numeric'
                      })}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        ) : (
          <motion.div 
            className="prediction-interface"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="selected-track">
              <button className="btn btn-back" onClick={() => setSelectedTrack(null)}>
                &larr; Back to all tracks
              </button>
              <h2>{selectedTrack.raceName}</h2>
              <p>{selectedTrack.circuitName} - {new Date(selectedTrack.date).toLocaleDateString()}</p>
            </div>

            {loading ? (
              <div className="loading-container">
                <div className="loading-spinner" />
                <p>Loading predictions...</p>
              </div>
            ) : (
              <>
                <div className="prediction-positions">
                  {[1,2,3,4,5].map(position => (
                    <div key={position} className="position-card">
                      <div className="position-header">
                        <span className="position-number">{position}</span>
                        <h3>
                          {position === 1 ? 'Winner' : 
                           position === 2 ? '2nd Place' : 
                           position === 3 ? '3rd Place' : `${position}th Place`}
                        </h3>
                      </div>
                      <div className="drivers-selection">
                        {drivers.map(driver => (
                          <motion.div
                            key={driver.driverId}
                            className={`driver-option ${userPrediction[position - 1] === driver.driverId ? 'selected' : ''}`}
                            onClick={() => handleDriverSelect(position)(driver.driverId)}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.98 }}
                          >
                            <div className="driver-image-container">
                              {driver.imageUrl ? (
                                <img src={driver.imageUrl} alt={`${driver.forename} ${driver.surname}`} className="driver-image" />
                              ) : (
                                <div className="driver-placeholder">
                                  {driver.forename[0]}{driver.surname[0]}
                                </div>
                              )}
                              {driver.teamLogo && (
                                <img src={driver.teamLogo} alt={driver.team} className="team-logo" />
                              )}
                            </div>
                            <div className="driver-info">
                              <h4>{driver.forename} {driver.surname}</h4>
                              <p>{driver.team}</p>
                            </div>
                          </motion.div>
                        ))}
                      </div>

                      // Add this component inside your prediction-interface section
                      <div className="driver-list">
                        {drivers.map((driver) => {
                          const teamColor = TEAM_COLORS[driver.teamRef] || '#333';
                          const predictedPosition = userPrediction.findIndex(id => id === driver.driverId) + 1;
                          
                          return (
                            <motion.div 
                              key={driver.driverId}
                              className="driver-row"
                              style={{ borderLeftColor: teamColor }}
                              whileHover={{ scale: 1.01 }}
                              onClick={() => handleDriverSelect(predictedPosition > 0 ? predictedPosition : 1)(driver.driverId)}
                            >
                              <div className="driver-image-container">
                                <img 
                                  src={driver.imageUrl || `https://via.placeholder.com/60/333/fff?text=${driver.forename[0]}${driver.surname[0]}`} 
                                  alt={`${driver.forename} ${driver.surname}`}
                                  className="driver-image"
                                />
                              </div>
                              
                              <div className="driver-details">
                                <div className="driver-name">
                                  {driver.forename} {driver.surname}
                                </div>
                                <div className="driver-team">
                                  {driver.teamLogo && (
                                    <img 
                                      src={driver.teamLogo} 
                                      alt={driver.teamName} 
                                      className="team-logo-small"
                                    />
                                  )}
                                  {driver.teamName}
                                </div>
                              </div>
                              
                              <div className="driver-time">
                                {predictedPosition > 0 ? (
                                  <span className="position-tag">P{predictedPosition}</span>
                                ) : (
                                  '--:--.---'
                                )}
                              </div>
                            </motion.div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="prediction-visualization">
                <h3>Your Current Prediction</h3>
                <div className="driver-list">
                  {userPrediction.slice(0, 5).map((driverId, index) => {
                    const driver = getDriverById(driverId);
                    if (!driver) return null;
                    
                    const teamColor = TEAM_COLORS[driver.teamRef] || '#333';
                    
                    return (
                      <div key={index} className="driver-row" style={{ borderLeftColor: teamColor }}>
                        <div className="driver-image-container">
                          <img 
                            src={driver.imageUrl} 
                            alt={`${driver.forename} ${driver.surname}`}
                            className="driver-image"
                          />
                        </div>
                        
                        <div className="driver-details">
                          <div className="driver-name">
                            {driver.forename} {driver.surname}
                          </div>
                          <div className="driver-team">
                            {driver.teamLogo && (
                              <img 
                                src={driver.teamLogo} 
                                alt={driver.teamName} 
                                className="team-logo-small"
                              />
                            )}
                            {driver.teamName}
                          </div>
                        </div>
                        
                        <div className="driver-time">
                          <span className="position-tag">P{index + 1}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

                <div className="prediction-actions">
                  <button
                    className="btn btn-primary submit-btn"
                    onClick={handleSubmitPrediction}
                    disabled={userPrediction.length < 5}
                  >
                    Submit Prediction
                  </button>
                </div>

                {predictions.length > 0 && (
                  <div className="community-predictions">
                    <h3>Community Predictions</h3>
                    <div className="predictions-grid">
                      {predictions.map((pred, idx) => (
                        <div key={idx} className="community-prediction">
                          <h4>Prediction #{idx + 1}</h4>
                          <div className="predicted-drivers">
                            {pred.slice(0, 5).map((driverId, pos) => {
                              const driver = getDriverById(driverId);
                              return driver ? (
                                <div key={pos} className="predicted-driver">
                                  <span className="predicted-position">{pos + 1}</span>
                                  {driver.imageUrl && (
                                    <img src={driver.imageUrl} alt={`${driver.forename} ${driver.surname}`} className="predicted-driver-image" />
                                  )}
                                  <span className="predicted-driver-name">
                                    {driver.forename[0]}. {driver.surname}
                                  </span>
                                </div>
                              ) : null;
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default Predictions;
