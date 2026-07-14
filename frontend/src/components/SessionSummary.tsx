import React, { useState, useEffect } from 'react';
import { BarChart2, Calendar, Clock, AlertTriangle, FileText, Download } from 'lucide-react';

interface SessionSummaryProps {
  apiBase: string;
  activeSessionId: string | null;
}

export const SessionSummary: React.FC<SessionSummaryProps> = ({ apiBase, activeSessionId }) => {
  const [summaryData, setSummaryData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    const fetchSummary = async () => {
      if (!activeSessionId) return;

      setLoading(true);
      try {
        const res = await fetch(`${apiBase}/sessions/${activeSessionId}/summary`);
        if (res.ok) {
          const data = await res.json();
          // The API sometimes returns { session_summary: {...} } or {...} directly. Let's handle both.
          setSummaryData(data.session_summary || data);
        }
      } catch (err) {
        console.error("Error loading session summary:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchSummary();
  }, [activeSessionId, apiBase]);

  if (!activeSessionId) {
    return (
      <div className="view-body">
        <div className="premium-card" style={{ textAlign: 'center', padding: '3rem' }}>
          <BarChart2 size={48} style={{ color: 'var(--text-muted)', marginBottom: '1rem' }} />
          <h3>No Active Video Session</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Select or upload a video session to compile intelligence summaries.
          </p>
        </div>
      </div>
    );
  }

  // Export handlers
  const handleExportJSON = () => {
    if (!summaryData) return;
    const blob = new Blob([JSON.stringify(summaryData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `session_report_${activeSessionId}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleExportCSV = () => {
    if (!summaryData || !summaryData.key_events) return;
    
    // Create CSV header & rows
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Event ID,Timestamp,Event Description,Severity\n";
    
    summaryData.key_events.forEach((ev: any, index: number) => {
      const row = `${index + 1},"${ev.timestamp || ''}","${(ev.description || ev.event || '').replace(/"/g, '""')}","${ev.severity || 'INFO'}"`;
      csvContent += row + "\n";
    });
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `session_events_${activeSessionId}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="view-body">
      {loading ? (
        <div className="premium-card" style={{ textAlign: 'center', padding: '3rem' }}>
          <p style={{ color: 'var(--text-muted)' }}>Compiling analysis stats...</p>
        </div>
      ) : summaryData ? (
        <div className="summary-container">
          {/* Main Info */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            {/* Overview Stats */}
            <div className="grid-cols-4" style={{ width: '100%' }}>
              <div className="premium-card">
                <span className="card-title">Date Processed</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
                  <Calendar size={18} style={{ color: 'var(--primary)' }} />
                  <span className="card-value" style={{ fontSize: '1.25rem' }}>
                    {summaryData.session_date || 'June 2026'}
                  </span>
                </div>
              </div>

              <div className="premium-card">
                <span className="card-title">Frames Analyzed</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
                  <BarChart2 size={18} style={{ color: 'var(--primary)' }} />
                  <span className="card-value" style={{ fontSize: '1.25rem' }}>
                    {summaryData.total_frames_analyzed || '0'}
                  </span>
                </div>
              </div>

              <div className="premium-card">
                <span className="card-title">Alerts Logged</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
                  <AlertTriangle size={18} style={{ color: 'var(--color-critical)' }} />
                  <span className="card-value" style={{ fontSize: '1.25rem', color: summaryData.total_alerts > 0 ? 'var(--color-critical)' : 'var(--color-clear)' }}>
                    {summaryData.total_alerts || '0'}
                  </span>
                </div>
              </div>

              <div className="premium-card">
                <span className="card-title">Analysis Time</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
                  <Clock size={18} style={{ color: 'var(--primary)' }} />
                  <span className="card-value" style={{ fontSize: '1.25rem' }}>
                    {summaryData.analysis_duration || '0.5 mins'}
                  </span>
                </div>
              </div>
            </div>

            {/* Key Events Timeline */}
            <div className="premium-card">
              <h3 className="card-title" style={{ fontSize: '0.95rem', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Clock size={16} /> Chronological Incident Log & Key Events
              </h3>
              
              <div className="key-events-list">
                {summaryData.key_events && summaryData.key_events.length > 0 ? (
                  summaryData.key_events.map((ev: any, idx: number) => (
                    <div key={idx} className="key-event-item">
                      <div className="key-event-time">{ev.timestamp || ev.frame || `Event ${idx + 1}`}</div>
                      <div className="key-event-desc">
                        {ev.description || ev.event || 'No description recorded'}
                        {ev.severity && ev.severity !== 'INFO' && (
                          <span style={{ marginLeft: '0.5rem', fontSize: '0.7rem', padding: '0.1rem 0.35rem', borderRadius: '4px', background: 'rgba(239, 68, 68, 0.15)', color: 'var(--color-critical)', fontWeight: 700 }}>
                            {ev.severity}
                          </span>
                        )}
                      </div>
                    </div>
                  ))
                ) : (
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>
                    No critical timeline events registered. Overall activity appears clear.
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Export Actions Panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div className="premium-card">
              <h4 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <FileText size={16} /> Export Intel Reports
              </h4>
              
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem', lineHeight: 1.4 }}>
                Compile the threat database, telemetry logs, and LLM analysis outputs of this session into structured files.
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <button className="btn btn-primary" onClick={handleExportJSON} style={{ width: '100%' }}>
                  <Download size={14} /> Export JSON Data
                </button>
                <button className="btn btn-secondary" onClick={handleExportCSV} style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Download size={14} /> Export CSV Timeline
                </button>
              </div>
            </div>

            <div className="premium-card">
              <h4 className="card-title" style={{ marginBottom: '0.75rem' }}>Threat Ratio</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                  <span>Monitored Activity</span>
                  <span>{summaryData.total_frames_analyzed > 0 ? '100' : '0'}%</span>
                </div>
                <div className="progress-track" style={{ height: '8px' }}>
                  <div className="progress-bar" style={{ width: '100%', background: 'var(--color-clear)' }}></div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginTop: '0.5rem' }}>
                  <span>Alarms Triggered</span>
                  <span>{summaryData.total_alerts || '0'} events</span>
                </div>
                <div className="progress-track" style={{ height: '8px' }}>
                  <div className="progress-bar" style={{ width: `${Math.min(100, (summaryData.total_alerts || 0) * 10)}%`, background: 'var(--color-critical)' }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="premium-card" style={{ textAlign: 'center', padding: '2rem' }}>
          <p style={{ color: 'var(--text-muted)' }}>Could not load summary details for this session.</p>
        </div>
      )}
    </div>
  );
};
