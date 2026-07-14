import { useState, useEffect } from 'react';
import { 
  Shield, 
  UploadCloud, 
  Video, 
  AlertTriangle, 
  Search, 
  BarChart2, 
  Bot,
  Layers,
  Radio,
  Menu,
  X
} from 'lucide-react';
import { VideoUpload } from './components/VideoUpload';
import { LiveCapture } from './components/LiveCapture';
import { FrameAnalysis } from './components/FrameAnalysis';
import { AlertCenter } from './components/AlertCenter';
import { SemanticSearch } from './components/SemanticSearch';
import { SessionSummary } from './components/SessionSummary';
import { SecurityAgent } from './components/SecurityAgent';
import { AgenticOrchestration } from './components/AgenticOrchestration';
import { DebugPanel } from './components/DebugPanel';
import { GuidedTour } from './components/GuidedTour';
import { LandingPage } from './components/LandingPage';

// Dynamically resolve API URL to support both local development and production deployments
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : window.location.origin;

function App() {
  const [activeTab, setActiveTab] = useState<string>('upload');
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [showLanding, setShowLanding] = useState<boolean>(true);
  const [showTour, setShowTour] = useState<boolean>(false);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(false);
  
  // Session states
  const [sessions, setSessions] = useState<any[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [totalAlerts, setTotalAlerts] = useState<number>(0);
  const [totalFrames, setTotalFrames] = useState<number>(0);

  // Monitor API Connection Health
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
          setApiConnected(true);
        } else {
          setApiConnected(false);
        }
      } catch (err) {
        setApiConnected(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fetch Available Video Sessions
  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      if (res.ok) {
        const data = await res.json();
        const sessionList = data.sessions || [];
        setSessions(sessionList);
        
        // Default select the most recent session if none selected
        if (sessionList.length > 0 && !selectedSessionId) {
          // Sort by upload_time descending and pick the most recent completed or processing session
          const sorted = [...sessionList].sort((a: any, b: any) => 
            new Date(b.upload_time).getTime() - new Date(a.upload_time).getTime()
          );
          const activeSession = sorted.find((s: any) => s.status === 'processing') || sorted.find((s: any) => s.status === 'completed');
          if (activeSession) {
            setSelectedSessionId(activeSession.session_id);
          }
        }
      }
    } catch (err) {
      console.error("Error loading sessions:", err);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  // Periodically refresh sessions list in background
  useEffect(() => {
    const interval = setInterval(fetchSessions, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fetch Quick Stats for selected session
  useEffect(() => {
    const fetchStats = async () => {
      if (!selectedSessionId) {
        setTotalAlerts(0);
        setTotalFrames(0);
        return;
      }

      try {
        // Alerts count
        const alertRes = await fetch(`${API_BASE}/sessions/${selectedSessionId}/alerts`);
        if (alertRes.ok) {
          const alertData = await alertRes.json();
          setTotalAlerts(alertData.total_alerts || 0);
        }

        // Frames count
        const frameRes = await fetch(`${API_BASE}/sessions/${selectedSessionId}/frames`);
        if (frameRes.ok) {
          const frameData = await frameRes.json();
          setTotalFrames(frameData.count || 0);
        }
      } catch (err) {
        console.error("Error fetching stats:", err);
      }
    };

    fetchStats();
  }, [selectedSessionId]);

  const handleUploadSuccess = (sessionId: string) => {
    setSelectedSessionId(sessionId);
    // Don't call fetchSessions here — it runs on the 15s interval anyway
    // and calling it immediately can trigger auto-select that overrides our choice
    setActiveTab('upload'); // Remain on upload to monitor process
  };

  const handleSessionDeleted = () => {
    setSelectedSessionId(null);
    fetchSessions();
  };

  const handleInvestigateFrame = (frameName: string) => {
    // Switch to analysis tab
    setActiveTab('frames');
    // Frame selection will be handled by passing it down or triggering selected state
    // To make this work smoothly, we can pass down parameters or let components sync
    // In our FrameAnalysis, we set selectedFrame inside useEffect based on activeSessionId.
    // We will save it in localStorage to share frame state!
    localStorage.setItem('drone_selected_frame', frameName);
    
    // Dispatch custom event to trigger update if FrameAnalysis is mounted
    window.dispatchEvent(new Event('frame_change_event'));
  };

  // Sync Frame selection across components using localstorage event
  const handleSelectFrameGlobal = (frameName: string) => {
    localStorage.setItem('drone_selected_frame', frameName);
    setActiveTab('frames');
    window.dispatchEvent(new Event('frame_change_event'));
  };

  return (
    <>
    {showLanding ? (
      <LandingPage onEnterDashboard={() => setShowLanding(false)} />
    ) : (
    <div className="dashboard-container">
      {/* Mobile backdrop — tap outside the sidebar to close it */}
      {sidebarOpen && <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />}

      {/* 1. Sidebar Navigation */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="logo-container">
          <Shield size={24} style={{ color: 'var(--primary)' }} />
          <h2 className="logo-text">DRONE SECURITY</h2>
          <button className="sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Close menu">
            <X size={20} />
          </button>
        </div>

        <div className="sidebar-section-title">System Status</div>
        <div className="status-indicator-box">
          <div className="status-row">
            <span className="status-label">Network API:</span>
            <span className="status-value" style={{ color: apiConnected ? 'var(--color-clear)' : 'var(--color-critical)' }}>
              <span className={`pulse-dot ${apiConnected === false ? 'offline' : ''}`}></span>
              {apiConnected === null ? 'Connecting...' : apiConnected ? 'Online' : 'Offline'}
            </span>
          </div>
          <div className="status-row">
            <span className="status-label">Environment:</span>
            <span className="status-value" style={{ textTransform: 'uppercase', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
              {window.location.hostname === 'localhost' ? 'Local Dev' : 'Cloud Prod'}
            </span>
          </div>
          {selectedSessionId && (
            <>
              <div className="status-row" style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                <span className="status-label">Frames:</span>
                <span className="status-value">{totalFrames}</span>
              </div>
              <div className="status-row">
                <span className="status-label">Alerts:</span>
                <span className="status-value" style={{ color: totalAlerts > 0 ? 'var(--color-critical)' : 'var(--color-clear)' }}>{totalAlerts}</span>
              </div>
            </>
          )}
        </div>

        <div className="sidebar-section-title">Navigation</div>
        <nav style={{ flexGrow: 1 }} onClick={() => setSidebarOpen(false)}>
          <ul className="nav-menu">
            <li 
              className={`nav-item ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              <UploadCloud size={18} />
              Video Upload
            </li>
            <li
              className={`nav-item ${activeTab === 'live' ? 'active' : ''}`}
              onClick={() => setActiveTab('live')}
            >
              <Radio size={18} />
              Live Capture
            </li>
            <li
              className={`nav-item ${activeTab === 'frames' ? 'active' : ''}`}
              onClick={() => setActiveTab('frames')}
            >
              <Video size={18} />
              Frame Analysis
            </li>
            <li 
              className={`nav-item ${activeTab === 'alerts' ? 'active' : ''}`}
              onClick={() => setActiveTab('alerts')}
            >
              <AlertTriangle size={18} />
              Alert Center
              {totalAlerts > 0 && (
                <span style={{ 
                  marginLeft: 'auto', 
                  backgroundColor: 'var(--color-critical)', 
                  color: 'white', 
                  fontSize: '0.75rem', 
                  fontWeight: 700, 
                  padding: '0.1rem 0.45rem', 
                  borderRadius: '10px' 
                }}>
                  {totalAlerts}
                </span>
              )}
            </li>
            <li 
              className={`nav-item ${activeTab === 'search' ? 'active' : ''}`}
              onClick={() => setActiveTab('search')}
            >
              <Search size={18} />
              Semantic Search
            </li>
            <li 
              className={`nav-item ${activeTab === 'summary' ? 'active' : ''}`}
              onClick={() => setActiveTab('summary')}
            >
              <BarChart2 size={18} />
              Session Summary
            </li>
            <li 
              className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              <Bot size={18} />
              Security Agent
            </li>
            <li 
              className={`nav-item ${activeTab === 'orchestration' ? 'active' : ''}`}
              onClick={() => setActiveTab('orchestration')}
            >
              <Layers size={18} />
              AI Orchestration
            </li>
            <li 
              className={`nav-item ${activeTab === 'debug' ? 'active' : ''}`}
              onClick={() => setActiveTab('debug')}
            >
              <AlertTriangle size={18} />
              Debug
            </li>
          </ul>
        </nav>

        {/* Footer info */}
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-color)', paddingTop: '1rem', marginTop: 'auto' }}>
          {/* Inline Tour Controls */}
          {showTour && (
            <div style={{ marginBottom: '0.75rem' }}>
              <GuidedTour 
                onNavigate={(tab) => setActiveTab(tab)} 
                onClose={() => setShowTour(false)}
                apiBase={API_BASE}
                onSessionCreated={(sid) => setSelectedSessionId(sid)}
              />
            </div>
          )}
          <p>© 2026 Drone Analyst Agent v2.0</p>
        </div>
      </aside>

      {/* 2. Main Workspace */}
      <main className="main-content">
        <header className="top-header">
          <div className="header-title-container">
            <button className="menu-toggle" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
              <Menu size={20} />
            </button>
            <h1>
              <Layers size={18} style={{ color: 'var(--primary)' }} />
              Security Intelligence Dashboard
            </h1>
          </div>

          {/* Session Selector + Tour Button */}
          <div className="session-selector-container">
            <button 
              onClick={() => setShowTour(true)} 
              style={{ padding: '0.35rem 0.75rem', background: 'linear-gradient(135deg, #7c3aed, #a78bfa)', border: 'none', borderRadius: '6px', color: 'white', fontSize: '0.75rem', fontWeight: 700, cursor: 'pointer', whiteSpace: 'nowrap' }}
            >
              Guided Tour
            </button>
            <span className="session-label" style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Active Video Session:</span>
            <select 
              className="custom-select"
              value={selectedSessionId || ''} 
              onChange={(e) => setSelectedSessionId(e.target.value || null)}
            >
              {sessions.length === 0 ? (
                <option value="">No Active Sessions</option>
              ) : (
                sessions.map(s => (
                  <option key={s.session_id} value={s.session_id}>
                    {s.filename} ({s.status})
                  </option>
                ))
              )}
            </select>
          </div>
        </header>

        {/* Dynamic Screen Routing */}
        <div style={{ flexGrow: 1, overflowY: 'auto' }}>
          {activeTab === 'upload' && (
            <VideoUpload 
              apiBase={API_BASE} 
              onUploadSuccess={handleUploadSuccess} 
              activeSessionId={selectedSessionId} 
              onSessionDeleted={handleSessionDeleted}
            />
          )}

          {activeTab === 'live' && (
            <LiveCapture
              apiBase={API_BASE}
              onAnalysisStarted={(sessionId) => {
                setSelectedSessionId(sessionId);
                setActiveTab('upload'); // VideoUpload polls & shows the pipeline progress
              }}
            />
          )}

          {activeTab === 'frames' && (
            <FrameAnalysis 
              apiBase={API_BASE} 
              activeSessionId={selectedSessionId} 
            />
          )}

          {activeTab === 'alerts' && (
            <AlertCenter 
              apiBase={API_BASE} 
              activeSessionId={selectedSessionId} 
              onInvestigateFrame={handleInvestigateFrame}
            />
          )}

          {activeTab === 'search' && (
            <SemanticSearch 
              apiBase={API_BASE} 
              activeSessionId={selectedSessionId} 
              onSelectFrame={handleSelectFrameGlobal}
            />
          )}

          {activeTab === 'summary' && (
            <SessionSummary 
              apiBase={API_BASE} 
              activeSessionId={selectedSessionId} 
            />
          )}

          {activeTab === 'chat' && (
            <SecurityAgent 
              apiBase={API_BASE} 
              activeSessionId={selectedSessionId} 
              onSelectFrame={handleSelectFrameGlobal}
            />
          )}

          {activeTab === 'orchestration' && (
            <AgenticOrchestration 
              apiBase={API_BASE} 
              activeSessionId={selectedSessionId} 
            />
          )}

          {activeTab === 'debug' && (
            <DebugPanel apiBase={API_BASE} />
          )}
        </div>
      </main>
    </div>
    )}
    </>
  );
}

export default App;
