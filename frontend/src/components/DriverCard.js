// Create a new component file DriverCard.js
import React, { useState } from 'react';
import { motion } from 'framer-motion';

const DriverCard = ({ driver, teamColor }) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const driverImageUrl = `https://media.formula1.com/image/upload/f_auto,c_limit,q_auto,w_600/content/dam/fom-website/drivers/2024Drivers/${driver.surname.toLowerCase()}.jpg`;
  const teamlogoImageUrl = `https://media.formula1.com/content/dam/fom-website/teams/2024/${driver.teamName}-logo.png`;

  return (
    <motion.div 
      className="driver-card"
      style={{ borderTopColor: teamColor }}
      whileHover={{ y: -8, boxShadow: "0 8px 24px rgba(0,0,0,0.3)" }}
      transition={{ duration: 0.3 }}
    >
      <div className="driver-image-container">
        <img 
          src={driverImageUrl}
          alt={`${driver.forename} ${driver.surname}`}
          className={`driver-image ${imageLoaded ? 'loaded' : 'loading'}`}
          onLoad={() => setImageLoaded(true)}
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = `https://via.placeholder.com/300x400/1e1e1e/ffffff?text=${driver.forename[0]}${driver.surname[0]}`;
            setImageLoaded(true);
          }}
        />
        
        <div className="driver-number">
          #{driver.number || 'N/A'}
        </div>
        
        {driver.teamRef && (
          <img 
            src={teamlogoImageUrl}
            alt={driver.teamName}
            className="team-logo"
            onError={(e) => e.target.style.display = 'none'}
          />
        )}
      </div>

      <div className="driver-info">
        <h3>{driver.forename} {driver.surname}</h3>
        <p>{driver.teamName}</p>
        <div className="prediction-points">
          {driver.points || 0} PTS
        </div>
      </div>
    </motion.div>
  );
};

export default DriverCard;