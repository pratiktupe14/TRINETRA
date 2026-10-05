import os
import subprocess

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content.strip())

# 1. Setup Vite Project (assume we run this inside the generated vite folder, or we generate it here)
base_dir = "cyberheal-ui"
os.makedirs(base_dir, exist_ok=True)

# Package.json will be handled by npm create vite, but we will overwrite src

# --- Tailwind Config ---
tailwind_config = """
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        indigo: {
          500: '#6366f1',
          600: '#4f46e5',
        }
      }
    },
  },
  plugins: [],
}
"""

index_css = """
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  @apply bg-gray-50 text-gray-900 antialiased;
}
"""

# --- API Layer ---
api_js = """
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const mockTickets = [
  { id: 'INC-1001', title: 'Brute Force Attack on auth-api', issue_type: 'Brute Force', severity: 'HIGH', status: 'RESOLVED', affected_service: 'auth-api', source_ip: '192.168.1.55', created_at: new Date().toISOString() },
  { id: 'INC-1002', title: 'Payment API Service Down', issue_type: 'Service Outage', severity: 'MEDIUM', status: 'IN_PROGRESS', affected_service: 'payment-api', source_ip: 'N/A', created_at: new Date().toISOString() },
  { id: 'INC-1003', title: 'Edge Camera Suspicious Traffic', issue_type: 'Anomaly', severity: 'HIGH', status: 'ESCALATED', affected_service: 'edge-camera-07', source_ip: '10.0.0.99', created_at: new Date().toISOString() }
];

export const getTickets = async () => {
  try {
    const res = await axios.get(`${API_BASE_URL}/incidents`);
    return res.data;
  } catch (e) {
    console.warn("Backend unavailable, using mock data");
    return mockTickets;
  }
};

export const getTicketById = async (id) => {
  try {
    const res = await axios.get(`${API_BASE_URL}/incidents/${id}`);
    return res.data;
  } catch (e) {
    console.warn("Backend unavailable, using mock data");
    const ticket = mockTickets.find(t => t.id === id) || mockTickets[0];
    return {
      ticket,
      agent_logs: [{ agent: 'Sentinel', status: 'SUCCESS', message: 'Anomaly detected', created_at: new Date().toISOString() }],
      remediation_actions: [{ action_type: 'Block IP', target: '192.168.1.55', status: 'COMPLETED' }],
      blocked_ips: [],
      verification_checks: []
    };
  }
};

export const simulateBruteforce = async () => { return { id: 'INC-SIM-1' }; };
export const simulateServiceDown = async () => { return { id: 'INC-SIM-2' }; };
export const simulateEdgeAnomaly = async () => { return { id: 'INC-SIM-3' }; };
"""

# --- Components ---
navbar_jsx = """
import { Link } from 'react-router-dom';
import { ShieldCheck, Activity } from 'lucide-react';

export default function Navbar() {
  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <ShieldCheck className="h-8 w-8 text-indigo-600" />
            <div className="ml-3">
              <span className="text-xl font-bold text-gray-900">CyberHeal AI</span>
              <span className="block text-xs text-gray-500">Autonomous cyber incident response</span>
            </div>
            <div className="ml-10 flex space-x-4">
              <Link to="/" className="text-gray-700 hover:text-indigo-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">Dashboard</Link>
              <Link to="/simulate" className="text-gray-700 hover:text-indigo-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">Simulate</Link>
            </div>
          </div>
          <div className="flex items-center">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800 border border-emerald-200">
              <Activity className="w-3 h-3 mr-1 animate-pulse" /> AI Agents Online
            </span>
          </div>
        </div>
      </div>
    </nav>
  );
}
"""

badges_jsx = """
export const StatusBadge = ({ status }) => {
  const colors = {
    OPEN: 'bg-blue-100 text-blue-800 border-blue-200',
    IN_PROGRESS: 'bg-amber-100 text-amber-800 border-amber-200',
    FIX_EXECUTED: 'bg-purple-100 text-purple-800 border-purple-200',
    VERIFYING: 'bg-indigo-100 text-indigo-800 border-indigo-200',
    RESOLVED: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    ESCALATED: 'bg-red-100 text-red-800 border-red-200',
  };
  return <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium border ${colors[status] || 'bg-gray-100 text-gray-800'}`}>{status}</span>;
};

export const SeverityBadge = ({ severity }) => {
  const colors = {
    LOW: 'bg-gray-100 text-gray-800',
    MEDIUM: 'bg-amber-100 text-amber-800',
    HIGH: 'bg-orange-100 text-orange-800',
    CRITICAL: 'bg-red-100 text-red-800',
  };
  return <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[severity] || 'bg-gray-100'}`}>{severity}</span>;
};
"""

stats_cards_jsx = """
import { Shield, Ticket, Activity, CheckCircle, AlertTriangle } from 'lucide-react';

export default function StatsCards({ tickets }) {
  const stats = [
    { name: 'Total Tickets', value: tickets.length, icon: Shield, color: 'text-indigo-600', bg: 'bg-indigo-50' },
    { name: 'Open', value: tickets.filter(t => t.status === 'OPEN').length, icon: Ticket, color: 'text-blue-600', bg: 'bg-blue-50' },
    { name: 'In Progress', value: tickets.filter(t => t.status === 'IN_PROGRESS').length, icon: Activity, color: 'text-amber-600', bg: 'bg-amber-50' },
    { name: 'Resolved', value: tickets.filter(t => t.status === 'RESOLVED').length, icon: CheckCircle, color: 'text-emerald-600', bg: 'bg-emerald-50' },
    { name: 'Escalated', value: tickets.filter(t => t.status === 'ESCALATED').length, icon: AlertTriangle, color: 'text-red-600', bg: 'bg-red-50' }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
      {stats.map((s) => (
        <div key={s.name} className="bg-white rounded-2xl p-6 border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 truncate">{s.name}</p>
              <p className="mt-1 text-3xl font-semibold text-gray-900">{s.value}</p>
            </div>
            <div className={`p-3 rounded-xl ${s.bg}`}>
              <s.icon className={`w-6 h-6 ${s.color}`} />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
"""

ticket_card_jsx = """
import { Link } from 'react-router-dom';
import { StatusBadge, SeverityBadge } from './Badges';

export default function TicketCard({ ticket }) {
  return (
    <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-bold text-gray-900">{ticket.title}</h3>
          <p className="text-sm text-gray-500 font-mono mt-1">{ticket.id}</p>
        </div>
        <div className="flex space-x-2">
          <SeverityBadge severity={ticket.severity} />
          <StatusBadge status={ticket.status} />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm text-gray-600 mb-6">
        <div><span className="font-medium">Issue Type:</span> {ticket.issue_type}</div>
        <div><span className="font-medium">Service:</span> {ticket.affected_service}</div>
        <div><span className="font-medium">Source IP:</span> {ticket.source_ip}</div>
        <div><span className="font-medium">Created:</span> {new Date(ticket.created_at).toLocaleString()}</div>
      </div>
      <Link to={`/tickets/${ticket.id}`} className="block w-full text-center bg-gray-50 hover:bg-indigo-50 text-indigo-600 font-medium py-2 rounded-lg transition-colors border border-gray-200 hover:border-indigo-200">
        View Details
      </Link>
    </div>
  );
}
"""

stepper_jsx = """
export default function LifecycleStepper({ currentStatus }) {
  const steps = ['OPEN', 'IN_PROGRESS', 'FIX_EXECUTED', 'VERIFYING', 'RESOLVED'];
  const currentIndex = steps.indexOf(currentStatus) === -1 ? (currentStatus === 'ESCALATED' ? 4 : 0) : steps.indexOf(currentStatus);
  
  return (
    <div className="flex items-center justify-between w-full mb-8 relative">
      <div className="absolute left-0 top-1/2 w-full h-1 bg-gray-200 -z-10 transform -translate-y-1/2"></div>
      {steps.map((step, idx) => {
        const isCompleted = idx < currentIndex || currentStatus === 'RESOLVED';
        const isCurrent = idx === currentIndex && currentStatus !== 'RESOLVED' && currentStatus !== 'ESCALATED';
        const isEscalated = step === 'RESOLVED' && currentStatus === 'ESCALATED';
        
        let bgColor = 'bg-white border-gray-300 text-gray-400';
        if (isCompleted) bgColor = 'bg-emerald-500 border-emerald-500 text-white';
        if (isCurrent) bgColor = 'bg-indigo-600 border-indigo-600 text-white ring-4 ring-indigo-100';
        if (isEscalated) bgColor = 'bg-red-500 border-red-500 text-white';

        return (
          <div key={step} className="flex flex-col items-center bg-white px-2">
            <div className={`w-8 h-8 rounded-full border-2 flex items-center justify-center font-bold text-sm transition-all ${bgColor}`}>
              {idx + 1}
            </div>
            <span className={`text-xs mt-2 font-medium ${isCurrent ? 'text-indigo-600' : 'text-gray-500'}`}>{isEscalated ? 'ESCALATED' : step}</span>
          </div>
        );
      })}
    </div>
  );
}
"""

dashboard_jsx = """
import { useState, useEffect } from 'react';
import { getTickets } from '../api';
import StatsCards from '../components/StatsCards';
import TicketCard from '../components/TicketCard';

export default function Dashboard() {
  const [tickets, setTickets] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getTickets().then(data => {
      setTickets(data);
      setLoading(false);
    });
  }, []);

  if (loading) return <div className="p-10 text-center text-gray-500">Loading Dashboard...</div>;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Security Operations Dashboard</h1>
        <p className="text-gray-500 mt-2">Monitor autonomous detection, diagnosis, remediation, and verification.</p>
      </div>
      
      <StatsCards tickets={tickets} />
      
      <h2 className="text-xl font-bold text-gray-900 mb-4">Recent Tickets</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {tickets.map(ticket => (
          <TicketCard key={ticket.id} ticket={ticket} />
        ))}
      </div>
    </div>
  );
}
"""

ticket_detail_jsx = """
import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getTicketById } from '../api';
import { StatusBadge, SeverityBadge } from '../components/Badges';
import LifecycleStepper from '../components/LifecycleStepper';
import { ArrowLeft, Terminal } from 'lucide-react';

export default function TicketDetail() {
  const { id } = useParams();
  const [data, setData] = useState(null);

  useEffect(() => {
    getTicketById(id).then(setData);
  }, [id]);

  if (!data) return <div className="p-10 text-center">Loading Ticket...</div>;
  const { ticket, agent_logs } = data;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Link to="/" className="inline-flex items-center text-sm font-medium text-gray-500 hover:text-indigo-600 mb-6 transition-colors">
        <ArrowLeft className="w-4 h-4 mr-1" /> Back to Dashboard
      </Link>
      
      <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden mb-8">
        <div className="p-6 border-b border-gray-100 bg-gray-50 flex justify-between items-start">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{ticket.title}</h1>
            <p className="text-sm text-gray-500 font-mono mt-1">{ticket.id}</p>
          </div>
          <div className="flex space-x-3">
            <SeverityBadge severity={ticket.severity} />
            <StatusBadge status={ticket.status} />
          </div>
        </div>
        <div className="p-6">
          <LifecycleStepper currentStatus={ticket.status} />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-sm">
            <div><p className="text-gray-500">Issue Type</p><p className="font-medium text-gray-900">{ticket.issue_type}</p></div>
            <div><p className="text-gray-500">Service</p><p className="font-medium text-gray-900">{ticket.affected_service}</p></div>
            <div><p className="text-gray-500">Source IP</p><p className="font-medium text-gray-900">{ticket.source_ip}</p></div>
            <div><p className="text-gray-500">Created</p><p className="font-medium text-gray-900">{new Date(ticket.created_at).toLocaleString()}</p></div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-4 flex items-center"><Terminal className="w-5 h-5 mr-2 text-gray-400"/> Agent Timeline</h2>
            <div className="space-y-4">
              {agent_logs.map((log, i) => (
                <div key={i} className="flex gap-4 p-4 rounded-xl bg-gray-50 border border-gray-100">
                  <div className="w-2 h-2 mt-2 rounded-full bg-indigo-500 flex-shrink-0"></div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-bold text-gray-900">{log.agent}</span>
                      <span className="text-xs text-gray-400">{new Date(log.created_at).toLocaleTimeString()}</span>
                    </div>
                    <p className="text-sm text-gray-600">{log.message}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        
        <div className="space-y-6">
           <div className="bg-indigo-50 rounded-2xl border border-indigo-100 p-6">
            <h2 className="text-lg font-bold text-indigo-900 mb-2">Final Report</h2>
            <p className="text-sm text-indigo-700">{ticket.resolution_summary || "Investigation ongoing..."}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
"""

simulate_jsx = """
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { simulateBruteforce, simulateServiceDown, simulateEdgeAnomaly } from '../api';
import { ShieldAlert, ActivitySquare, Radio } from 'lucide-react';

export default function Simulate() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState(null);

  const runSim = async (fn, name) => {
    setLoading(true);
    try {
      const res = await fn();
      setToast({ type: 'success', msg: `Triggered ${name}! Incident ID: ${res.id}`, id: res.id });
    } catch (e) {
      setToast({ type: 'error', msg: `Failed to trigger ${name}` });
    }
    setLoading(false);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Simulate Cyber Incidents</h1>
        <p className="text-gray-500 mt-2">Trigger demo alerts and watch CyberHeal AI resolve them autonomously.</p>
      </div>

      {toast && (
        <div className={`p-4 mb-6 rounded-xl border ${toast.type === 'success' ? 'bg-emerald-50 border-emerald-200 text-emerald-800' : 'bg-red-50 border-red-200 text-red-800'} flex justify-between items-center`}>
          <span>{toast.msg}</span>
          {toast.id && <button onClick={() => navigate(`/tickets/${toast.id}`)} className="text-sm font-bold underline">View Ticket Details</button>}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm">
          <ShieldAlert className="w-10 h-10 text-red-500 mb-4" />
          <h2 className="text-xl font-bold text-gray-900 mb-2">Brute Force Attack</h2>
          <p className="text-gray-500 text-sm mb-6">Simulates 148 failed login attempts from a suspicious IP against auth-api.</p>
          <button disabled={loading} onClick={() => runSim(simulateBruteforce, 'Brute Force')} className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 rounded-lg transition-colors disabled:opacity-50">Run Simulation</button>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm">
          <ActivitySquare className="w-10 h-10 text-amber-500 mb-4" />
          <h2 className="text-xl font-bold text-gray-900 mb-2">Service Down</h2>
          <p className="text-gray-500 text-sm mb-6">Simulates payment-api health failure with high error rate.</p>
          <button disabled={loading} onClick={() => runSim(simulateServiceDown, 'Service Down')} className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 rounded-lg transition-colors disabled:opacity-50">Run Simulation</button>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm">
          <Radio className="w-10 h-10 text-purple-500 mb-4" />
          <h2 className="text-xl font-bold text-gray-900 mb-2">Edge Device Anomaly</h2>
          <p className="text-gray-500 text-sm mb-6">Simulates suspicious outbound traffic from edge-camera-07.</p>
          <button disabled={loading} onClick={() => runSim(simulateEdgeAnomaly, 'Edge Anomaly')} className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 rounded-lg transition-colors disabled:opacity-50">Run Simulation</button>
        </div>
      </div>
    </div>
  );
}
"""

app_jsx = """
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import TicketDetail from './pages/TicketDetail';
import Simulate from './pages/Simulate';

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/tickets/:id" element={<TicketDetail />} />
            <Route path="/simulate" element={<Simulate />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
"""

main_jsx = """
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
"""

# Write all files
create_file(f"{base_dir}/tailwind.config.js", tailwind_config)
create_file(f"{base_dir}/src/index.css", index_css)
create_file(f"{base_dir}/src/api.js", api_js)
create_file(f"{base_dir}/src/components/Navbar.jsx", navbar_jsx)
create_file(f"{base_dir}/src/components/Badges.jsx", badges_jsx)
create_file(f"{base_dir}/src/components/StatsCards.jsx", stats_cards_jsx)
create_file(f"{base_dir}/src/components/TicketCard.jsx", ticket_card_jsx)
create_file(f"{base_dir}/src/components/LifecycleStepper.jsx", stepper_jsx)
create_file(f"{base_dir}/src/pages/Dashboard.jsx", dashboard_jsx)
create_file(f"{base_dir}/src/pages/TicketDetail.jsx", ticket_detail_jsx)
create_file(f"{base_dir}/src/pages/Simulate.jsx", simulate_jsx)
create_file(f"{base_dir}/src/App.jsx", app_jsx)
create_file(f"{base_dir}/src/main.jsx", main_jsx)

print("UI scaffolding complete.")
