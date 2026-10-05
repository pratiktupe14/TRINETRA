import React, { useEffect, useState } from 'react';
import { supabase } from '../supabaseClient';

/**
 * This is a standalone example component demonstrating how to use 
 * Supabase Realtime WebSockets to stream live logs.
 */
function LiveLogsExample() {
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    // 1. Create a channel
    const channel = supabase
      .channel('live-pipeline-updates')
      
      // 2. Listen to INSERT events on the 'logs' table
      .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'logs' }, (payload) => {
        console.log("New log received!", payload.new);
        setLogs((prevLogs) => [...prevLogs, payload.new]);
      })
      
      // 3. Listen to all events on 'incidents' table
      .on('postgres_changes', { event: '*', schema: 'public', table: 'incidents' }, (payload) => {
        console.log("Incident updated:", payload.new);
      })
      
      // 4. Subscribe
      .subscribe();

    // 5. Cleanup on unmount
    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  return (
    <div style={{ padding: '20px', background: '#0f111a', color: '#e2e8f0', borderRadius: '8px' }}>
      <h2>Live Log Stream</h2>
      <p style={{ color: '#94a3b8' }}>Waiting for real-time updates from Supabase...</p>
      
      <div style={{ marginTop: '20px', fontFamily: 'monospace' }}>
        {logs.map((log) => (
          <div key={log.id} style={{ marginBottom: '8px' }}>
            <span style={{ color: '#a855f7' }}>[{log.agent}]</span> 
            <span style={{ marginLeft: '10px' }}>{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default LiveLogsExample;
