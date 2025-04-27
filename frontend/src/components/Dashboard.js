import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import './Dashboard.css';

const Dashboard = () => {
  const [drivers, setDrivers] = useState([]);
  const [teams, setTeams] = useState([]);
  const [races, setRaces] = useState([]);
  const [circuits, setCircuits] = useState([]);
  const [loading, setLoading] = useState({
    drivers: true,
    teams: true,
    races: true,
    circuits: true
  });
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch all necessary data in parallel
        const [teamsResponse, racesResponse, circuitsResponse] = await Promise.all([
          axios.get('http://127.0.0.1:8000/teams'),
          axios.get('http://127.0.0.1:8000/races?season=2024'),
          axios.get('http://127.0.0.1:8000/circuits')
        ]);

        setTeams(teamsResponse.data?.data || []);
        setRaces(racesResponse.data?.data || []);
        setCircuits(circuitsResponse.data?.data || []);

        // Now fetch drivers for each team
        const driversPromises = teamsResponse.data.data.map(team => 
          axios.get(`http://127.0.0.1:8000/drivers?team_id=${team.constructorId}`)
        );

        const driversResponses = await Promise.all(driversPromises);
        
        // Combine drivers with their team info
        const combinedDrivers = [];
        teamsResponse.data.data.forEach((team, index) => {
          const teamDrivers = driversResponses[index]?.data?.data || [];
          teamDrivers.forEach(driver => {
            combinedDrivers.push({
              ...driver,
              teamId: team.constructorId,
              teamName: team.name,
              teamRef: team.constructorRef,
              teamNationality: team.nationality
            });
          });
        });

        setDrivers(combinedDrivers);
        setLoading({
          drivers: false,
          teams: false,
          races: false,
          circuits: false
        });

      } catch (err) {
        setError('Failed to load data. Please try again later.');
        console.error('Error fetching data:', err);
      }
    };

    fetchData();
  }, []);

  const getCircuitById = (circuitId) => {
    return circuits.find(circuit => circuit.circuitId === circuitId) || {};
  };

  const getInitials = (name) => {
    if (!name) return 'TR';
    return name.split(' ')
      .map(word => word[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Date TBD';
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      });
    } catch {
      return 'Date TBD';
    }
  };

  if (loading.drivers || loading.teams || loading.races || loading.circuits) {
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
          {drivers.map((driver) => (
            <motion.div 
              key={`${driver.driverId}-${driver.teamId}`}
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
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = '';
                      e.target.style.display = 'none';
                    }}
                  />
                ) : (
                  <div className="driver-placeholder">
                    {getInitials(`${driver.forename} ${driver.surname}`)}
                  </div>
                )}
                {driver.teamRef && (
                  <img 
                    src={`/team-logos/${driver.teamRef}.png`} 
                    alt={driver.teamName}
                    className="team-logo"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = '';
                      e.target.style.display = 'none';
                    }}
                  />
                )}
              </div>
              <div className="driver-info">
                <h3>{driver.forename} {driver.surname}</h3>
                <p>{driver.teamName}</p>
                <div className="driver-number">{driver.number || 'N/A'}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Races Timeline */}
      <section className="races-section">
        <h2>2024 Race Calendar</h2>
        <div className="races-timeline">
          {races.map((race) => {
            const circuit = getCircuitById(race.circuitId);
            return (
              <motion.div 
                key={race.raceId}
                className="race-card"
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.2 }}
              >
                <div className="race-image-container">
                  {circuit.imageUrl ? (
                    <img 
                      src={circuit.imageUrl} 
                      alt={circuit.name}
                      className="race-image"
                      onError={(e) => {
                        e.target.onerror = null;
                        e.target.src = '';
                        e.target.style.display = 'none';
                      }}
                    />
                  ) : (
                    <div className="race-placeholder">
                      {getInitials(circuit.name)}
                    </div>
                  )}
                </div>
                <div className="race-info">
                  <h3>{race.name}</h3>
                  <p className="race-circuit">
                    {circuit.name || 'Circuit not specified'}
                    {circuit.location && `, ${circuit.location}`}
                  </p>
                  <p className="race-date">{formatDate(race.date)}</p>
                  <div className="race-round">Round {race.round}</div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </section>
    </motion.div>
  );
};

export default Dashboard;