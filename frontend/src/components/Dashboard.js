import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import './Dashboard.css';

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

  const getTeamColor = (teamRef) => {
    return TEAM_COLORS[teamRef] || '#333333';
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [teamsResponse, racesResponse, circuitsResponse] = await Promise.all([
          axios.get('http://127.0.0.1:8000/teams'),
          axios.get('http://127.0.0.1:8000/races?season=2024'),
          axios.get('http://127.0.0.1:8000/circuits')
        ]);

        setTeams(teamsResponse.data?.data || []);
        setRaces(racesResponse.data?.data || []);
        setCircuits(circuitsResponse.data?.data || []);

        const driversPromises = teamsResponse.data.data.map(team => 
          axios.get(`http://127.0.0.1:8000/drivers?team_id=${team.constructorId}`)
        );

        const driversResponses = await Promise.all(driversPromises);
        
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
      <div className="dashboard-container">
        <div className="loading-spinner" />
        <p>Loading data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-container">
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      {/* Header with F1 logo */}
      <div className="dashboard-header">
        <div className="dashboard-header-text">
          <h1>2024 Drivers</h1>
          <p>Dashboard Predictions Points</p>
        </div>
      </div>

      {/* Drivers Section */}
      <section className="drivers-section">
        <h2 className="section-title">Drivers Championship</h2>
        <div className="drivers-grid">
          {drivers.map((driver, index) => {
            const teamColor = getTeamColor(driver.teamRef);
            return (
              <motion.div 
                key={`${driver.driverId}-${driver.teamId}`}
                className="driver-card"
                style={{
                  borderLeft: `4px solid ${teamColor}`
                }}
                whileHover={{ y: -5 }}
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
                        e.target.style.display = 'none';
                      }}
                    />
                  )}
                  <div className="driver-number" style={{ backgroundColor: teamColor }}>
                    #{driver.number || 'N/A'}
                  </div>
                </div>
                <div className="driver-info">
                  <h3>{driver.forename} {driver.surname}</h3>
                  <p>{driver.teamName}</p>
                  <div className="prediction-points">
                    Points: {driver.points || 0}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* Races Section */}
      <section className="races-section">
        <h2 className="section-title">Race Calendar</h2>
        <div className="races-timeline">
          {races.map((race) => {
            const circuit = getCircuitById(race.circuitId);
            return (
              <motion.div 
                key={race.raceId}
                className="race-card"
                whileHover={{ y: -5 }}
                transition={{ duration: 0.3 }}
              >
                <div className="race-image-container">
                  {circuit.imageUrl ? (
                    <img 
                      src={circuit.imageUrl} 
                      alt={circuit.name}
                      className="race-image"
                      onError={(e) => {
                        e.target.onerror = null;
                        e.target.style.display = 'none';
                      }}
                    />
                  ) : (
                    <div className="race-placeholder">
                      {getInitials(circuit.name)}
                    </div>
                  )}
                  <div className="race-round">
                    Round {race.round}
                  </div>
                </div>
                <div className="race-info">
                  <h3>{race.name}</h3>
                  <p>{circuit.name || 'Circuit not specified'}</p>
                  <p>{formatDate(race.date)}</p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </section>
    </div>
  );
};

export default Dashboard;