import React, { useState, useEffect } from 'react';
import { 
  Activity, GitBranch, Crosshair, ShieldCheck, FileText, 
  AlertTriangle, ShieldAlert, Play, RefreshCw, Cpu, CheckCircle2, Zap
} from 'lucide-react';
import AgentCard from './components/AgentCard';
import LogConsole from './components/LogConsole';
import { supabase } from './supabaseClient';
import './index.css';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const PIPELINE_STEPS = [
  { id: 'sentinel', name: 'Sentinel', desc: 'Anomaly Detection', icon: Activity, color: '#06b6d4' },
  { id: 'causor', name: 'Causor', desc: 'Root-Cause Analysis', icon: GitBranch, color: '#3b82f6' },
  { id: 'planner', name: 'Planner', desc: 'Remediation Plan', icon: Crosshair, color: '#8b5cf6' },
  { id: 'guardian', name: 'Guardian', desc: 'Safety & Policy Check', icon: ShieldCheck, color: '#10b981' },
  { id: 'executor', name: 'Executor', desc: 'Remediation Execution', icon: Zap, color: '#f59e0b' },
  { id: 'verifier', name: 'Verifier', desc: 'Telemetry Verification', icon: CheckCircle2, color: '#06b6d4' },
  { id: 'rlm', name: 'RLM Agent', desc: 'Reflective Learning', icon: Cpu, color: '#ec4899' },
  { id: 'scribe', name: 'Scribe', desc: 'Audit & Blockchain', icon: FileText, color: '#a855f7' }
];

const DEMO_INCIDENT = {
  id: 'cad5fef4-3bb1-4258-9b59-e7667de18218',
  title: 'Real Failed Logon Burst Detected (Windows/Linux)',
  severity: 'HIGH',
  status: 'RESOLVED',
  source_ip: '192.168.1.55'
};

const INITIAL_ANOMALIES = [
  { id: 'anom-1', metric: 'failed_logon_burst', source: 'auth-gateway', value: '4625 events/min' },
  { id: 'anom-2', metric: 'traffic_spike', source: 'edge-proxy', value: '99.9 percentile' }
];

const INITIAL_ROOT_CAUSES = [
  { id: 'rc-1', service: 'auth-gateway', description: 'High-risk brute force attack verified from external IP (Score: 75/100).' }
];

const INITIAL_ACTIONS = [
  { id: 'act-1', action_type: 'block_ip', target: '192.168.1.55', status: 'COMPLETED' },
  { id: 'act-2', action_type: 'blockchain_anchor', target: 'Block #4 (SHA-256 Verified)', status: 'COMPLETED' }
];

const INITIAL_LOGS = [
  { time: '10:00:01', level: 'INFO', module: 'Trigger', message: 'Trigger Event Agent: Threat detected from event log telemetry.' },
  { time: '10:00:02', level: 'INFO', module: 'Alert', message: 'Alert Agent: Converted raw telemetry into structured HIGH alert.' },
  { time: '10:00:03', level: 'INFO', module: 'Ticket', message: 'Ticket Agent: Incident case created with ID #cad5fef4.' },
  { time: '10:00:04', level: 'INFO', module: 'Intake', message: 'Diagnosis Agent: Normalizing telemetry and assessing severity.' },
  { time: '10:00:05', level: 'INFO', module: 'Sentinel', message: 'Sentinel Agent: Anomaly verified on auth-gateway.' },
  { time: '10:00:06', level: 'INFO', module: 'Causor', message: 'RCA Agent: Dynamic dependency graph constructed. Root cause localized.' },
  { time: '10:00:07', level: 'INFO', module: 'Guardian', message: 'Guardian Agent: Blast-radius evaluation passed (Risk: LOW). Approved.' },
  { time: '10:00:08', level: 'INFO', module: 'Executor', message: 'Fix Agent: Executing OS firewall rule. Cryptographic hash mined in Block #2.' },
  { time: '10:00:09', level: 'WARN', module: 'Verifier', message: 'Verification Agent: Threat signature still active on attempt #0.' },
  { time: '10:00:10', level: 'INFO', module: 'RLM', message: 'RLM Agent: Learning from failure. Pivoting strategy to EXPAND_BLOCK_RANGE.' },
  { time: '10:00:11', level: 'INFO', module: 'Verifier', message: 'Verification Agent: Verification PASSED on attempt #1.' },
  { time: '10:00:12', level: 'INFO', module: 'Scribe', message: 'Resolution Agent: Resolution hash anchored to Blockchain. Incident RESOLVED.' }
];

function App() {
  const [activeStep, setActiveStep] = useState(8);
  const [incident, setIncident] = useState(DEMO_INCIDENT);
  const [anomalies, setAnomalies] = useState(INITIAL_ANOMALIES);
  const [rootCauses, setRootCauses] = useState(INITIAL_ROOT_CAUSES);
  const [actions, setActions] = useState(INITIAL_ACTIONS);
  const [logs, setLogs] = useState(INITIAL_LOGS);
  const [backendOnline, setBackendOnline] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);

  // Check Backend Connectivity
  useEffect(() => {
    fetch(`${API_BASE}/`)
      .then(res => res.json())
      .then(() => setBackendOnline(true))
      .catch(() => setBackendOnline(false));
  }, []);

  // Fetch Supabase data if configured
  useEffect(() => {
    const fetchLatestIncident = async () => {
      try {
        if (!supabase) return;
        const { data: incidentData } = await supabase
          .from('incidents')
          .select('*')
          .order('created_at', { ascending: false })
          .limit(1)
          .single();

        if (incidentData) {
          setIncident(incidentData);
          fetchRelatedData(incidentData.id);
        }
      } catch (e) {
        console.warn("Supabase fetch fallback to local state:", e);
      }
    };

    const fetchRelatedData = async (incidentId) => {
      try {
        const { data: logData } = await supabase.from('logs').select('*').eq('incident_id', incidentId).order('timestamp', { ascending: true });
        const { data: anomalyData } = await supabase.from('anomalies').select('*').eq('incident_id', incidentId);
        const { data: rcData } = await supabase.from('root_causes').select('*').eq('incident_id', incidentId);
        const { data: actionData } = await supabase.from('actions').select('*').eq('incident_id', incidentId);

        if (logData && logData.length > 0) setLogs(logData.map(formatLog));
        if (anomalyData && anomalyData.length > 0) setAnomalies(anomalyData);
        if (rcData && rcData.length > 0) setRootCauses(rcData);
        if (actionData && actionData.length > 0) setActions(actionData);
      } catch (e) {
        console.warn("Supabase related data fallback:", e);
      }
    };

    fetchLatestIncident();

    try {
      const channel = supabase
        .channel('schema-db-changes')
        .on('postgres_changes', { event: '*', schema: 'public', table: 'incidents' }, (payload) => {
          if (payload.eventType === 'INSERT' || payload.eventType === 'UPDATE') {
            setIncident(payload.new);
          }
        })
        .subscribe();

      return () => {
        try { supabase.removeChannel(channel); } catch (e) {}
      };
    } catch (e) {
      console.warn("Realtime channel not available:", e);
    }
  }, []);

  const formatLog = (dbLog) => ({
    time: new Date(dbLog.timestamp || Date.now()).toLocaleTimeString(),
    level: 'INFO',
    module: dbLog.agent,
    message: dbLog.message
  });

  const runSimulation = async (simType = 'bruteforce') => {
    setIsSimulating(true);
    const incidentId = 'INC-' + Math.random().toString(36).substring(2, 9).toUpperCase();
    
    // Step 1: Trigger Event Agent & Alert Agent
    setActiveStep(1);
    setIncident({
      id: incidentId,
      title: simType === 'bruteforce' ? 'Real Brute Force Attack Burst' : 'Service Outage on Core API',
      severity: 'HIGH',
      status: 'INVESTIGATING',
      source_ip: '192.168.1.55'
    });
    setAnomalies([]);
    setRootCauses([]);
    setActions([]);
    setLogs([{
      time: new Date().toLocaleTimeString(),
      level: 'INFO',
      module: 'Trigger',
      message: 'Trigger Event Agent: Threshold condition breached (4625 Failed Logon Burst).'
    }]);

    // Step by step simulation progression
    setTimeout(() => {
      setActiveStep(2);
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        level: 'INFO',
        module: 'Alert',
        message: 'Alert Agent: Converted threat telemetry to HIGH severity security alert.'
      }]);
    }, 1000);

    setTimeout(() => {
      setActiveStep(3);
      setAnomalies([
        { id: 'a-sim-1', metric: 'failed_auth_burst', source: 'auth-svc', value: '45 req/sec' }
      ]);
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        level: 'INFO',
        module: 'Ticket',
        message: `Ticket Agent: Incident case #${incidentId} registered. Dispatched to queue.`
      }]);
    }, 2000);

    setTimeout(() => {
      setActiveStep(4);
      setRootCauses([
        { id: 'rc-sim-1', service: 'auth-svc', description: 'Malicious credential stuffing from untrusted range (Score: 85).' }
      ]);
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        level: 'INFO',
        module: 'Causor',
        message: 'RCA Agent: Dynamic dependency graph analyzed. Attack source verified.'
      }]);
    }, 3200);

    setTimeout(() => {
      setActiveStep(5);
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        level: 'INFO',
        module: 'Guardian',
        message: 'Guardian Agent: Blast-radius check passed. Policy compliant (Risk: LOW).'
      }]);
    }, 4400);

    setTimeout(() => {
      setActiveStep(6);
      setActions([
        { id: 'act-sim-1', action_type: 'block_ip', target: '192.168.1.55', status: 'COMPLETED' }
      ]);
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        level: 'INFO',
        module: 'Executor',
        message: 'Fix Agent: Executed network firewall rule. Cryptographic hash recorded in Blockchain Block #2.'
      }]);
    }, 5600);

    setTimeout(() => {
      setActiveStep(7);
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        level: 'WARN',
        module: 'Verifier',
        message: 'Verification Agent: Verification attempt #0 failed (Signature persistent).'
      }, {
        time: new Date().toLocaleTimeString(),
        level: 'INFO',
        module: 'RLM',
        message: 'RLM Agent: Learning from verification failure. Pivoting strategy to EXPAND_BLOCK_RANGE.'
      }]);
    }, 6800);

    setTimeout(() => {
      setActiveStep(8);
      setIncident(prev => ({ ...prev, status: 'RESOLVED' }));
      setActions(prev => [
        ...prev,
        { id: 'act-sim-2', action_type: 'blockchain_anchor', target: 'Resolution Block #4', status: 'COMPLETED' }
      ]);
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        level: 'INFO',
        module: 'Verifier',
        message: 'Verification Agent: Threat mitigated successfully on attempt #1.'
      }, {
        time: new Date().toLocaleTimeString(),
        level: 'INFO',
        module: 'Scribe',
        message: 'Resolution Agent: Final audit bundle compiled. Immutable block mined. Incident RESOLVED.'
      }]);
      setIsSimulating(false);
    }, 8200);

    // Also trigger backend simulation in background if API online
    if (backendOnline) {
      try {
        fetch(`${API_BASE}/simulate/bruteforce`, { method: 'POST' }).catch(() => {});
      } catch (e) {}
    }
  };

  return (
    <div className="app-container">
      {/* Top Bar with System Health Indicator */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            width: '10px', height: '10px', borderRadius: '50%',
            backgroundColor: backendOnline ? '#10b981' : '#f59e0b',
            boxShadow: backendOnline ? '0 0 10px #10b981' : 'none'
          }} />
          <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
            Backend API: <strong style={{ color: backendOnline ? '#10b981' : '#f59e0b' }}>
              {backendOnline ? 'Online (Port 8000)' : 'Standby / Offline Mode'}
            </strong>
          </span>
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button 
            onClick={() => runSimulation('bruteforce')}
            disabled={isSimulating}
            style={{
              padding: '0.5rem 1.25rem',
              backgroundColor: isSimulating ? '#334155' : '#3b82f6',
              color: '#ffffff',
              border: 'none',
              borderRadius: '8px',
              cursor: isSimulating ? 'not-allowed' : 'pointer',
              fontWeight: 600,
              fontSize: '0.9rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)',
              transition: 'all 0.2s ease'
            }}
          >
            {isSimulating ? <RefreshCw className="animate-spin" size={16} /> : <Play size={16} />}
            {isSimulating ? 'Agents Running...' : 'Simulate Live Attack'}
          </button>
        </div>
      </div>

      <div className="header slide-up">
        <h1>SHIELD-RCA Dashboard</h1>
        <p>Autonomous Multi-Agent Incident Response & Self-Healing Pipeline</p>
      </div>

      {/* Incident Header Panel */}
      <div className="incident-panel fade-in glass-panel">
        <div className="incident-info">
          <ShieldAlert size={34} color={incident?.severity === 'CRITICAL' ? '#ef4444' : '#3b82f6'} />
          <div style={{ marginLeft: '1rem' }}>
            <h2 style={{ margin: 0, fontSize: '1.4rem' }}>{incident?.title || "Monitoring System..."}</h2>
            <p style={{ margin: '0.25rem 0 0', color: '#94a3b8', fontSize: '0.9rem' }}>
              Incident ID: <code>{incident?.id || "N/A"}</code> | Target IP: <code>{incident?.source_ip || "192.168.1.55"}</code>
            </p>
          </div>
        </div>
        <div className="incident-status">
          <span className={`badge ${incident?.status?.toLowerCase() || 'standby'}`}>
            {incident?.status || "STANDBY"}
          </span>
        </div>
      </div>

      {/* Agents Pipeline Grid */}
      <div className="pipeline-grid fade-in" style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', 
        gap: '1rem',
        animationDelay: '0.2s', 
        marginTop: '1.5rem' 
      }}>
        {PIPELINE_STEPS.map((step, index) => {
          const stepNum = index + 1;
          const isActive = activeStep === stepNum;
          const isCompleted = activeStep > stepNum;
          
          return (
            <AgentCard 
              key={step.id}
              name={step.name}
              description={step.desc}
              icon={step.icon}
              color={step.color}
              isActive={isActive}
              isCompleted={isCompleted}
            />
          );
        })}
      </div>

      {/* Telemetry & RCA Data Panels */}
      <div className="data-panels fade-in" style={{ animationDelay: '0.4s', marginTop: '1.5rem' }}>
        <div className="data-card glass-panel">
          <h3><Activity size={18} color="#06b6d4" /> Detected Anomalies</h3>
          <ul>
            {anomalies.map(a => (
              <li key={a.id}><strong>{a.metric}</strong> on <code>{a.source}</code> ({a.value})</li>
            ))}
            {anomalies.length === 0 && <li className="empty">No anomalies detected yet</li>}
          </ul>
        </div>
        
        <div className="data-card glass-panel">
          <h3><GitBranch size={18} color="#3b82f6" /> Root Cause Analysis (RCA)</h3>
          <ul>
            {rootCauses.map(rc => (
              <li key={rc.id}><strong>{rc.service}</strong>: {rc.description}</li>
            ))}
            {rootCauses.length === 0 && <li className="empty">Dependency graph analyzing...</li>}
          </ul>
        </div>

        <div className="data-card glass-panel">
          <h3><Crosshair size={18} color="#10b981" /> Remediation & Blockchain Audit</h3>
          <ul>
            {actions.map(act => (
              <li key={act.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span><strong>{act.action_type}</strong> &rarr; <code>{act.target}</code></span>
                <span className={`status-sm ${act.status.toLowerCase()}`}>{act.status}</span>
              </li>
            ))}
            {actions.length === 0 && <li className="empty">Awaiting Guardian approval...</li>}
          </ul>
        </div>
      </div>

      {/* Terminal / Live Communications Console */}
      <div className="glass-panel terminal-wrapper fade-in" style={{ animationDelay: '0.6s', marginTop: '1.5rem' }}>
        <h3 style={{ margin: '0 0 1rem 0', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#10b981', animation: 'pulseGlow 2s infinite' }} />
          Live Agent Communications & Self-Healing Trace Log
        </h3>
        <LogConsole logs={logs} />
      </div>
    </div>
  );
}

export default App;
