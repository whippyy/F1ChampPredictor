import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';
import f1Logo from './f1-logo.png'; // Make sure to add an F1 logo image

const Dashboard = () => {
  const [drivers, setDrivers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/drivers');
        // Assuming the API returns data in a format we can use directly
        // If not, you'll need to transform it to match the structure below
        setDrivers(response.data?.data || []);
        setLoading(false);
      } catch (err) {
        setError('Failed to load driver data. Please try again later.');
        console.error('Error fetching data:', err);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="dashboard-container">
        <p>Loading driver data...</p>
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
        <img src={f1Logo} alt="F1 Logo" className="f1-logo" />
        <h1>2024 Drivers</h1>
      </div>

      {/* Drivers Grid */}
      <section className="drivers-section">
        <h2>Dashboard Predictions</h2>
        <div className="drivers-grid">
          {drivers.map((driver) => (
            <div key={driver.driverId} className="driver-card">
              <div className="driver-number">#{driver.number}</div>
              <div className="driver-info">
                <h3>{driver.forename} {driver.surname}</h3>
                <p>{driver.teamName}</p>
                <div className="driver-points">Points: {driver.points || 0}</div>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default Dashboard;