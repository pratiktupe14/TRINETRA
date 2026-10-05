import React from 'react';

const AgentCard = ({ name, description, icon: Icon, color, isActive, isCompleted }) => {
  let statusClass = 'inactive';
  if (isActive) statusClass = 'active';
  else if (isCompleted) statusClass = 'completed';

  return (
    <div 
      className={`glass-panel agent-card ${statusClass}`}
      style={{ '--active-color': color }}
    >
      <div className="icon-wrapper">
        <Icon size={32} />
      </div>
      <div className="agent-info">
        <h3 className="agent-name">{name}</h3>
        <p className="agent-desc">{description}</p>
      </div>
    </div>
  );
};

export default AgentCard;
