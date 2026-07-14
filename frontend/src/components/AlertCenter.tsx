import React, { useState, useEffect } from 'react';
import { AlertOctagon, AlertTriangle, Info, ShieldCheck, PlayCircle, Filter } from 'lucide-react';

interface AlertCenterProps {
  apiBase: string;
  activeSessionId: string | null;
  onInvestigateFrame: (frameId: string) => void;
}

export const AlertCenter: React.FC<AlertCenterProps> = ({ apiBase, activeSessionId, onInvestigateFrame }) => {
  const [alerts, setAlerts] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [severityFilter, setSeverityFilter] = useState<string>('ALL');

  useEffect(() => {
    const fetchAlerts = async () => {
      if (!activeSessionId) return;

      setLoading(true);
      try {
        const res = await fetch(`${apiBase}/sessions/${activeSessionId}/alerts`);
        if (res.ok) {
          const data = await res.json();
          setAlerts(data.alerts || []);
        }
      } catch (err) {
        console.error("Error loading alerts:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchAlerts();
  }, [activeSessionId, apiBase]);

  if (!activeSessionId) {
    return (
      <div className="view-body">
        <div className="premium-card" style={{ textAlign: 'center', padding: '3rem' }}>
          <AlertTriangle size={48} style={{ color: 'var(--text-muted)', marginBottom: '1rem' }} />
          <h3>No Active Video Session</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Please select or upload a video session to monitor security alerts.
          </p>
        </div>
      </div>
    );
  }

  const getAlertIcon = (severity: string) => {
    const s = (severity || '').toUpperCase();
    if (s === 'CRITICAL') return <AlertOctagon size={20} style={{ color: 'var(--color-critical)' }} />;
    if (s === 'HIGH') return <AlertTriangle size={20} style={{ color: 'var(--color-high)' }} />;
    if (s === 'MEDIUM') return <AlertTriangle size={20} style={{ color: 'var(--color-medium)' }} />;
    return <Info size={20} style={{ color: 'var(--color-clear)' }} />;
  };

  const getRowClass = (severity: string) => {
    const s = (severity || '').toUpperCase();
    if (s === 'CRITICAL') return 'alert-critical';
    if (s === 'HIGH') return 'alert-high';
    if (s === 'MEDIUM') return 'alert-medium';
    return 'alert-clear';
  };

  const filteredAlerts = alerts.filter(a => {
    if (severityFilter === 'ALL') return true;
    return (a.severity || '').toUpperCase() === severityFilter;
  });

  return (
    <div className="view-body">
      <div className="premium-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <h3 className="card-title" style={{ fontSize: '1.1rem', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertOctagon size={20} /> Security Alert Logs ({filteredAlerts.length})
          </h3>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Filter size={14} style={{ color: 'var(--text-muted)' }} />
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Severity:</span>
            <div style={{ display: 'flex', gap: '0.25rem', background: 'rgba(255,255,255,0.02)', padding: '0.2rem', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
              {['ALL', 'CRITICAL', 'HIGH', 'MEDIUM'].map(lvl => (
                <button
                  key={lvl}
                  className="btn"
                  onClick={() => setSeverityFilter(lvl)}
                  style={{
                    padding: '0.25rem 0.65rem',
                    fontSize: '0.75rem',
                    borderRadius: '4px',
                    border: 'none',
                    fontWeight: 700,
                    backgroundColor: severityFilter === lvl ? 'var(--primary)' : 'transparent',
                    color: severityFilter === lvl ? 'var(--text-main)' : 'var(--text-muted)'
                  }}
                >
                  {lvl}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="premium-card" style={{ textAlign: 'center', padding: '2rem' }}>
          <p style={{ color: 'var(--text-muted)' }}>Loading active threats...</p>
        </div>
      ) : filteredAlerts.length === 0 ? (
        <div className="premium-card" style={{ textAlign: 'center', padding: '4rem', borderLeft: '4px solid var(--color-clear)' }}>
          <ShieldCheck size={48} style={{ color: 'var(--color-clear)', marginBottom: '1rem' }} />
          <h3>Security Status Clear</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            No matching threats are registered for this timeframe.
          </p>
        </div>
      ) : (
        <div className="alert-card-grid">
          {filteredAlerts.map((alt, idx) => {
            const frameId = alt.frame_id || alt.frame || '';
            const frameName = frameId.endsWith('.jpg') ? frameId : `${frameId}.jpg`;
            
            return (
              <div key={idx} className={`alert-row ${getRowClass(alt.severity)}`}>
                <div className="alert-main-info">
                  <div className="alert-meta">
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                      {getAlertIcon(alt.severity)}
                      <strong style={{ textTransform: 'capitalize', color: 'var(--text-main)' }}>
                        {alt.severity} Alert
                      </strong>
                    </span>
                    <span>•</span>
                    <span>Frame: <strong style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>{frameId}</strong></span>
                    <span>•</span>
                    <span>Time: {alt.timestamp || 'N/A'}</span>
                    <span>•</span>
                    <span>Zone: {alt.location || 'Interior'}</span>
                  </div>

                  <p className="alert-message">
                    <strong>Event:</strong> {alt.message || 'Suspicious visual triggers detected.'}
                  </p>

                  {alt.recommended_action && (
                    <p className="alert-recommendation">
                      <span><strong>Recommended Action:</strong> {alt.recommended_action}</span>
                    </p>
                  )}
                </div>

                <div className="alert-action-btn">
                  <button 
                    className="btn btn-secondary"
                    onClick={() => onInvestigateFrame(frameName)}
                    style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8rem', padding: '0.5rem 0.85rem' }}
                  >
                    <PlayCircle size={14} /> Investigate
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
