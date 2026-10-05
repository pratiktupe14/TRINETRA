import React, { useEffect, useRef } from 'react';

const LogConsole = ({ logs }) => {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="console">
      {logs.map((log, i) => (
        <p key={i} className="log-line">
          <span className="log-time">{log.time}</span>
          <span className={`log-level-${log.level.toLowerCase()}`}>{log.level}</span>
          <span className="log-module">{log.module}</span>
          <span className="log-msg">{log.message}</span>
        </p>
      ))}
      <div ref={bottomRef} />
    </div>
  );
};

export default LogConsole;
