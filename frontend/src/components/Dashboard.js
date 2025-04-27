import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import './Dashboard.css';

const Dashboard = () => {
  const [drivers, setDrivers] = useState([]);
  const [races, setRaces] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [driversResponse, racesResponse] = await Promise.all([
          axios.get('http://127.0.0.1:8000/drivers?with_images=true'),
          axios.get('http://127.0.0.1:8000/races?season=2024')
        ]);
        
        setDrivers(driversResponse.data.data || []);
        setRaces(racesResponse.data.data || []);
      } catch (err) {
        setError('Failed to load data. Please try again later.');
        console.error('Error fetching data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner" />
        <p>Loading data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <p>{error}</p>
        <button 
          className="btn btn-primary"
          onClick={() => window.location.reload()}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <motion.div 
      className="dashboard-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Main Title */}
      <div className="dashboard-header">
        <h1>F1 RACE PREDICTOR 2024</h1>
        <p>Predict race outcomes and track the 2024 season</p>
      </div>

      {/* Drivers Grid */}
      <section className="drivers-section">
        <h2>Drivers Championship</h2>
        <div className="drivers-grid">
          {drivers.map((driver, index) => (
            <motion.div 
              key={driver.driverId}
              className="driver-card"
              whileHover={{ y: -5, boxShadow: "0 10px 20px rgba(0,0,0,0.2)" }}
              transition={{ duration: 0.3 }}
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
                <h3>{driver.forename} {driver.surname}</h3>
                <p>{driver.team}</p>
                <div className="driver-number">{index + 1}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Races Timeline */}
      <section className="races-section">
        <h2>2024 Race Calendar</h2>
        <div className="races-timeline">
          {races.map((race) => (
            <motion.div 
              key={race.raceId}
              className="race-card"
              whileHover={{ scale: 1.02 }}
              transition={{ duration: 0.2 }}
            >
              <div className="race-image-container">
                {race.circuitImage ? (
                  <img 
                    src={race.circuitImage} 
                    alt={race.circuitName}
                    className="race-image"
                  />
                ) : (
                  <div className="race-placeholder">
                    {race.circuitName.split(' ').map(word => word[0]).join('')}
                  </div>
                )}
              </div>
              <div className="race-info">
                <h3>{race.raceName}</h3>
                <p className="race-circuit">{race.circuitName}</p>
                <p className="race-date">
                  {new Date(race.date).toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric'
                  })}
                </p>
                <div className="race-round">Round {race.round}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </section>
    </motion.div>
  );
};

export default Dashboard;