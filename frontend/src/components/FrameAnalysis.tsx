import React, { useState, useEffect } from 'react';
import { Shield, MapPin, Eye, Users, FileText } from 'lucide-react';

interface FrameAnalysisProps {
  apiBase: string;
  activeSessionId: string | null;
}

export const FrameAnalysis: React.FC<FrameAnalysisProps> = ({ apiBase, activeSessionId }) => {
  const [frames, setFrames] = useState<string[]>([]);
  const [selectedFrame, setSelectedFrame] = useState<string | null>(null);
  const [loadingFrames, setLoadingFrames] = useState<boolean>(false);
  
  // Frame metadata states
  const [telemetry, setTelemetry] = useState<any>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [alert, setAlert] = useState<any>(null);

  // Fetch frame list when session changes
  useEffect(() => {
    const fetchFrames = async () => {
      if (!activeSessionId) {
        setFrames([]);
        setSelectedFrame(null);
        return;
      }

      setLoadingFrames(true);
      try {
        const res = await fetch(`${apiBase}/sessions/${activeSessionId}/frames`);
        if (res.ok) {
          const data = await res.json();
          const frameList = data.frames || [];
          setFrames(frameList);
          if (frameList.length > 0) {
            setSelectedFrame(frameList[0]);
          } else {
            setSelectedFrame(null);
          }
        }
      } catch (err) {
        console.error("Error loading frames:", err);
      } finally {
        setLoadingFrames(false);
      }
    };

    fetchFrames();
  }, [activeSessionId, apiBase]);

  // Listen for global frame change events (from AlertCenter / SemanticSearch redirections)
  useEffect(() => {
    const handleFrameChange = () => {
      const storedFrame = localStorage.getItem('drone_selected_frame');
      if (storedFrame && frames.includes(storedFrame)) {
        setSelectedFrame(storedFrame);
        localStorage.removeItem('drone_selected_frame');
      }
    };

    window.addEventListener('frame_change_event', handleFrameChange);
    handleFrameChange();

    return () => {
      window.removeEventListener('frame_change_event', handleFrameChange);
    };
  }, [frames]);


  // Fetch frame metadata when selected frame changes
  useEffect(() => {
    const fetchFrameMetadata = async () => {
      if (!activeSessionId || !selectedFrame) {
        setTelemetry(null);
        setAnalysis(null);
        setAlert(null);
        return;
      }

      const frameId = selectedFrame.replace('.jpg', '');

      try {
        // Fetch Telemetry
        const tPromise = fetch(`${apiBase}/sessions/${activeSessionId}/frames/${frameId}/telemetry`)
          .then(res => res.ok ? res.json() : null)
          .catch(() => null);

        // Fetch Analysis
        const aPromise = fetch(`${apiBase}/sessions/${activeSessionId}/frames/${frameId}/analysis`)
          .then(res => res.ok ? res.json() : null)
          .catch(() => null);

        // Fetch Alert
        const alPromise = fetch(`${apiBase}/sessions/${activeSessionId}/frames/${frameId}/alert`)
          .then(res => res.ok ? res.json() : null)
          .catch(() => null);

        const [tData, aData, alData] = await Promise.all([tPromise, aPromise, alPromise]);
        
        setTelemetry(tData);
        setAnalysis(aData);
        setAlert(alData);
      } catch (err) {
        console.error("Error loading frame metadata:", err);
      }
    };

    fetchFrameMetadata();
  }, [selectedFrame, activeSessionId, apiBase]);

  if (!activeSessionId) {
    return (
      <div className="view-body">
        <div className="premium-card" style={{ textAlign: 'center', padding: '3rem' }}>
          <Shield size={48} style={{ color: 'var(--text-muted)', marginBottom: '1rem' }} />
          <h3>No Active Video Session</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Please select a session from the top dropdown or upload a new video.
          </p>
        </div>
      </div>
    );
  }

  const getThreatColorClass = (level: string) => {
    const l = (level || '').toUpperCase();
    if (l === 'CRITICAL') return 'badge-critical';
    if (l === 'HIGH') return 'badge-high';
    if (l === 'MEDIUM') return 'badge-medium';
    return 'badge-clear';
  };

  const getThreatBorderColor = (level: string) => {
    const l = (level || '').toUpperCase();
    if (l === 'CRITICAL') return 'var(--color-critical)';
    if (l === 'HIGH') return 'var(--color-high)';
    if (l === 'MEDIUM') return 'var(--color-medium)';
    return 'var(--color-clear)';
  };

  return (
    <div className="view-body" style={{ paddingBottom: '1rem' }}>
      <div className="split-screen">
        {/* Left Side Frame List */}
        <div className="frame-list-sidebar">
          <h4>Extracted Frames ({frames.length})</h4>
          {loadingFrames ? (
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Loading frames...</p>
          ) : frames.length === 0 ? (
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No frames extracted yet.</p>
          ) : (
            frames.map(f => (
              <button
                key={f}
                className={`frame-item-btn ${selectedFrame === f ? 'active' : ''}`}
                onClick={() => setSelectedFrame(f)}
              >
                {f.replace('.jpg', '')}
              </button>
            ))
          )}
        </div>

        {/* Right Side Frame View and Metadata */}
        <div className="frame-analysis-main">
          {selectedFrame ? (
            <>
              {/* Frame Media display and Alert Panel */}
              <div className="frame-display-card">
                {/* 1. Large Image Frame */}
                <div className="frame-img-box">
                  <img 
                    src={`${apiBase}/sessions/${activeSessionId}/frame-image/${selectedFrame}`} 
                    alt={`Frame ${selectedFrame}`}
                  />
                  {analysis && (analysis.threat_level || analysis.overall_threat_level) && (
                    <div className="frame-img-overlay-alert">
                      <span className={`alert-badge ${getThreatColorClass(analysis.threat_level || analysis.overall_threat_level)}`}>
                        Threat: {analysis.threat_level || analysis.overall_threat_level}
                      </span>
                    </div>
                  )}
                </div>

                {/* 2. Top-Level Alarm Card */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div 
                    className="premium-card" 
                    style={{ 
                      height: '100%', 
                      borderLeft: `4px solid ${getThreatBorderColor(analysis?.threat_level || analysis?.overall_threat_level || 'CLEAR')}` 
                    }}
                  >
                    <h4 className="card-title">Threat Level Summary</h4>
                    <div style={{ marginTop: '0.5rem', display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                      <span className="card-value" style={{ color: getThreatBorderColor(analysis?.threat_level || analysis?.overall_threat_level || 'CLEAR') }}>
                        {analysis?.threat_level || analysis?.overall_threat_level || 'CLEAR'}
                      </span>
                    </div>
                    
                    {analysis?.threat_type && analysis.threat_type !== 'clear' && (
                      <p style={{ fontSize: '0.85rem', marginTop: '0.5rem', fontWeight: 700 }}>
                        Incident: <span style={{ textTransform: 'capitalize', color: 'var(--color-high)' }}>
                          {analysis.threat_type.replace(/_/g, ' ')}
                        </span>
                      </p>
                    )}

                    <div style={{ marginTop: '0.75rem', fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                      <strong>Reasoning: </strong>
                      {analysis?.alert_reasoning || analysis?.reasoning || 'No threat indicators identified in this frame.'}
                    </div>

                    {analysis?.security_signals && analysis.security_signals.length > 0 && (
                      <div style={{ marginTop: '0.75rem' }}>
                        <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>Signals:</span>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', marginTop: '0.25rem' }}>
                          {analysis.security_signals.map((sig: string) => (
                            <span key={sig} style={{ fontSize: '0.75rem', background: 'rgba(255,255,255,0.05)', padding: '0.15rem 0.5rem', borderRadius: '4px' }}>
                              {sig}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Telemetry and Vision Analysis Details */}
              <div className="resp-grid-2" style={{ gap: '1.5rem' }}>
                {/* Telemetry Data Card */}
                <div className="premium-card" style={{ height: 'fit-content' }}>
                  <h4 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <MapPin size={16} /> Drone Telemetry Data
                  </h4>
                  <div style={{ marginTop: '1rem' }} className="metadata-grid">
                    <div className="metadata-pair">
                      <span className="metadata-label">GPS Coordinate</span>
                      <span className="metadata-value-mono">
                        {telemetry?.location?.latitude?.toFixed(5) || '37.7749'}N,{' '}
                        {telemetry?.location?.longitude?.toFixed(5) || '-122.4194'}W
                      </span>
                    </div>
                    <div className="metadata-pair">
                      <span className="metadata-label">Altitude</span>
                      <span className="metadata-value-mono">
                        {telemetry?.altitude_m?.toFixed(1) || telemetry?.altitude || '45.2'} meters
                      </span>
                    </div>
                    <div className="metadata-pair">
                      <span className="metadata-label">Gimbal Heading</span>
                      <span className="metadata-value-mono">
                        {telemetry?.gimbal_yaw_deg?.toFixed(1) || telemetry?.heading || '180.0'}° (South)
                      </span>
                    </div>
                    <div className="metadata-pair">
                      <span className="metadata-label">Drone Speed</span>
                      <span className="metadata-value-mono">
                        {telemetry?.speed_mps?.toFixed(1) || telemetry?.speed || '5.4'} m/s
                      </span>
                    </div>
                    <div className="metadata-pair">
                      <span className="metadata-label">Timestamp</span>
                      <span className="metadata-value-mono">
                        {telemetry?.timestamp || 'N/A'}
                      </span>
                    </div>
                    <div className="metadata-pair">
                      <span className="metadata-label">Battery Level</span>
                      <span className="metadata-value" style={{ color: 'var(--color-clear)' }}>
                        {telemetry?.battery_percentage || '92'}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* AI Vision Analysis Card */}
                <div className="premium-card" style={{ height: 'fit-content' }}>
                  <h4 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Eye size={16} /> AI Scene Vision
                  </h4>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
                    {analysis?.image_quality && (
                      <div style={{ fontSize: '0.8rem', padding: '0.6rem 0.75rem', borderRadius: '6px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)' }}>
                        <strong>Frame Quality:</strong>{' '}
                        <span style={{ color: analysis.image_quality.quality_score >= 60 ? 'var(--color-clear)' : analysis.image_quality.quality_score >= 40 ? 'var(--color-medium)' : 'var(--color-high)' }}>
                          {analysis.image_quality.quality_score ?? 'N/A'}/100
                        </span>
                        {analysis.image_quality.preprocessed === 1 && (
                          <span style={{ marginLeft: '0.5rem', color: 'var(--primary)', fontSize: '0.75rem' }}>(enhanced for VLM)</span>
                        )}
                        <div style={{ marginTop: '0.35rem', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                          Blur {analysis.image_quality.blur_score?.toFixed?.(0) ?? '—'} ·
                          Brightness {analysis.image_quality.brightness?.toFixed?.(0) ?? '—'} ·
                          Contrast {analysis.image_quality.contrast?.toFixed?.(0) ?? '—'}
                        </div>
                      </div>
                    )}
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                      <span><strong>Scene Location:</strong> {analysis?.scene_type || 'Retail Store/Aisle'}</span>
                      <span><strong>People Present:</strong> {analysis?.people_count ?? 0}</span>
                    </div>

                    <div style={{ fontSize: '0.9rem', lineHeight: '1.4', background: 'rgba(255,255,255,0.02)', padding: '0.75rem', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                      <strong>VLM Scene Description:</strong>
                      <p style={{ marginTop: '0.25rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                        {analysis?.vlm_description || 'A neutral scan of shelves with products. Customers are walking.'}
                      </p>
                    </div>

                    <div>
                      <strong style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Detected Objects:</strong>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', marginTop: '0.25rem' }}>
                        {analysis?.objects_detected && analysis.objects_detected.length > 0 ? (
                          analysis.objects_detected.map((obj: string) => (
                            <span key={obj} style={{ fontSize: '0.75rem', background: 'rgba(59, 130, 246, 0.1)', color: 'var(--primary)', padding: '0.15rem 0.5rem', borderRadius: '4px', border: '1px solid rgba(59,130,246,0.2)' }}>
                              {obj}
                            </span>
                          ))
                        ) : (
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>None listed</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Cross-Frame Person Tracker */}
              {analysis?.person_features && analysis.person_features.length > 0 && (
                <div className="premium-card">
                  <h4 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                    <Users size={16} /> Suspect Tracker (Cross-Frame Re-Identification)
                    <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)', fontWeight: 400, marginLeft: 'auto', fontStyle: 'italic' }}>Only works when suspect is detected</span>
                  </h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '1rem' }}>
                    {analysis.person_features.map((person: any, idx: number) => (
                      <div key={idx} style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '0.75rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem', fontWeight: 700 }}>
                          <span style={{ color: 'var(--primary)' }}>ID: {person.id || `suspect_${idx + 1}`}</span>
                          <span style={{ textTransform: 'capitalize', color: 'var(--text-muted)' }}>{person.clothing_color || 'Unknown'} Clothing</span>
                        </div>
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                          <strong>Actions: </strong>{person.actions?.join(', ') || 'Neutral posture'}
                        </p>
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                          <strong>Physical: </strong>{person.physical_attributes || 'Slim build, average height'}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Developer Logs & Raw JSON */}
              <div className="premium-card" style={{ overflow: 'visible' }}>
                <details open>
                  <summary style={{ cursor: 'pointer', fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <FileText size={14} /> Developer console: View Raw Frame Payload
                  </summary>
                  <pre style={{ marginTop: '1rem', padding: '1rem', background: '#030712', borderRadius: '6px', overflowX: 'auto', overflowY: 'auto', maxHeight: '400px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#a7f3d0' }}>
                    {JSON.stringify({ telemetry, analysis, alert }, null, 2)}
                  </pre>
                </details>
              </div>
            </>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
              <Shield size={36} style={{ marginBottom: '0.5rem' }} />
              <p>Select an extracted frame from the sidebar list to review security details.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
