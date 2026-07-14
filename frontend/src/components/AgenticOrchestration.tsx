import { useState, useEffect } from 'react';
import { Bot, Layers, RefreshCw, Brain } from 'lucide-react';

interface AgenticOrchestrationProps {
  apiBase: string;
  activeSessionId: string | null;
}

export const AgenticOrchestration: React.FC<AgenticOrchestrationProps> = ({ apiBase, activeSessionId }) => {
  const [contextSummaries, setContextSummaries] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!activeSessionId) return;
    
    const fetchContext = async () => {
      setLoading(true);
      try {
        // Try session-specific context summaries
        const res = await fetch(`${apiBase}/sessions/${activeSessionId}/summary`);
        if (res.ok) {
          const data = await res.json();
          if (data.key_events && data.key_events.length > 0) {
            setContextSummaries(data.key_events);
          }
        }
      } catch (err) {
        console.error("Error loading context:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchContext();
  }, [activeSessionId, apiBase]);

  if (!activeSessionId) {
    return (
      <div className="view-body">
        <div className="premium-card" style={{ textAlign: 'center', padding: '3rem' }}>
          <Brain size={48} style={{ color: 'var(--text-muted)', marginBottom: '1rem' }} />
          <h3>No Active Session</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>Upload a video to see the agentic orchestration in action.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="view-body">
      {/* Architecture Diagram */}
      <div className="premium-card">
        <h3 className="card-title" style={{ fontSize: '1.1rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Layers size={18} /> Multi-Agent Orchestration Architecture
        </h3>
        
        <div style={{ padding: '1.5rem', background: 'rgba(0,0,0,0.3)', borderRadius: '12px' }}>
          {/* Top: Input Layer */}
          <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.75rem', padding: '0.6rem 1.5rem', background: 'linear-gradient(135deg, #1e40af, #3b82f6)', borderRadius: '8px', fontWeight: 700, fontSize: '0.85rem' }}>
              Video Upload → Frame Extraction (Hybrid Motion + Scene Detection)
            </div>
          </div>

          {/* Arrow */}
          <div style={{ textAlign: 'center', fontSize: '1.2rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>▼</div>

          {/* Processing Pipeline */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem', marginBottom: '1.5rem' }}>
            <div style={{ padding: '0.75rem', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#f59e0b' }}>Telemetry</div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>GPS, Altitude, Speed</div>
            </div>
            <div style={{ padding: '0.75rem', background: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.3)', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#818cf8' }}>VLM Analysis</div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Groq llama-4-scout</div>
            </div>
            <div style={{ padding: '0.75rem', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#10b981' }}>Vector Index</div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Pinecone (llama-embed)</div>
            </div>
          </div>

          {/* Arrow */}
          <div style={{ textAlign: 'center', fontSize: '1.2rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>▼</div>

          {/* Orchestrator */}
          <div style={{ padding: '1rem', background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15))', border: '1px solid rgba(139, 92, 246, 0.4)', borderRadius: '10px', marginBottom: '1.5rem' }}>
            <div style={{ textAlign: 'center', fontWeight: 800, fontSize: '0.9rem', color: '#a78bfa', marginBottom: '0.75rem' }}>
              AI ORCHESTRATOR — NVIDIA Nemotron-3-Ultra-550B
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem' }}>
              <div style={{ padding: '0.6rem', background: 'rgba(0,0,0,0.3)', borderRadius: '6px', textAlign: 'center' }}>
                <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#818cf8' }}>Analysis Agent</div>
                <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>Pattern + Risk + Temporal</div>
              </div>
              <div style={{ padding: '0.6rem', background: 'rgba(0,0,0,0.3)', borderRadius: '6px', textAlign: 'center' }}>
                <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#34d399' }}>QA Agent</div>
                <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>Validation + Confidence</div>
              </div>
              <div style={{ padding: '0.6rem', background: 'rgba(0,0,0,0.3)', borderRadius: '6px', textAlign: 'center' }}>
                <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#f59e0b' }}>Question Agent</div>
                <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>NL Search + Q&A</div>
              </div>
            </div>
          </div>

          {/* Arrow */}
          <div style={{ textAlign: 'center', fontSize: '1.2rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>▼</div>

          {/* Alert Engine */}
          <div style={{ padding: '0.75rem 1rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', textAlign: 'center', marginBottom: '1.5rem' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#ef4444' }}>Alert Engine (Rule-Based + LLM-Validated)</div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>Behavioral Analysis → Shoplifting Detection → Severity Classification</div>
          </div>

          {/* Arrow */}
          <div style={{ textAlign: 'center', fontSize: '1.2rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>▼</div>

          {/* Output Layer */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '0.5rem' }}>
            <div style={{ padding: '0.5rem', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '6px', textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', fontWeight: 600 }}>Dashboard</div>
            </div>
            <div style={{ padding: '0.5rem', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '6px', textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', fontWeight: 600 }}>Alerts</div>
            </div>
            <div style={{ padding: '0.5rem', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '6px', textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', fontWeight: 600 }}>Agent Chat</div>
            </div>
            <div style={{ padding: '0.5rem', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '6px', textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', fontWeight: 600 }}>MongoDB</div>
            </div>
          </div>
        </div>
      </div>

      {/* Agent Capabilities */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
        <div className="premium-card" style={{ overflow: 'visible' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <Bot size={16} style={{ color: '#818cf8' }} />
            <h4 style={{ fontSize: '0.85rem', fontWeight: 700 }}>Analysis Agent</h4>
          </div>
          <ul style={{ fontSize: '0.78rem', color: 'var(--text-muted)', paddingLeft: '1rem', lineHeight: '1.8' }}>
            <li>Deep frame analysis</li>
            <li>Pattern recognition</li>
            <li>Temporal progression</li>
            <li>Risk assessment</li>
          </ul>
        </div>
        <div className="premium-card" style={{ overflow: 'visible' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <Bot size={16} style={{ color: '#34d399' }} />
            <h4 style={{ fontSize: '0.85rem', fontWeight: 700 }}>QA Agent</h4>
          </div>
          <ul style={{ fontSize: '0.78rem', color: 'var(--text-muted)', paddingLeft: '1rem', lineHeight: '1.8' }}>
            <li>Confidence validation</li>
            <li>False positive detection</li>
            <li>Cross-frame consistency</li>
            <li>Quality scoring</li>
          </ul>
        </div>
        <div className="premium-card" style={{ overflow: 'visible' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <Bot size={16} style={{ color: '#f59e0b' }} />
            <h4 style={{ fontSize: '0.85rem', fontWeight: 700 }}>Question Agent</h4>
          </div>
          <ul style={{ fontSize: '0.78rem', color: 'var(--text-muted)', paddingLeft: '1rem', lineHeight: '1.8' }}>
            <li>Natural language Q&A</li>
            <li>Evidence retrieval</li>
            <li>Source verification</li>
            <li>Follow-up suggestions</li>
          </ul>
        </div>
      </div>

      {/* Context Summaries */}
      <div className="premium-card" style={{ marginTop: '1rem', overflow: 'visible' }}>
        <h3 className="card-title" style={{ fontSize: '1rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Brain size={16} /> Context Summaries & Agent Reasoning
          {loading && <RefreshCw size={14} className="animate-spin" style={{ marginLeft: 'auto' }} />}
        </h3>
        
        {contextSummaries.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {contextSummaries.map((event, idx) => (
              <div key={idx} style={{ 
                padding: '0.75rem 1rem', 
                background: 'rgba(255,255,255,0.02)', 
                borderRadius: '6px', 
                borderLeft: `3px solid ${event.severity === 'HIGH' ? '#ef4444' : event.severity === 'MEDIUM' ? '#f59e0b' : '#10b981'}`,
                fontSize: '0.82rem'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                  <span style={{ fontWeight: 700, color: 'var(--text-main)' }}>{event.frame || `Event ${idx + 1}`}</span>
                  <span style={{ 
                    fontSize: '0.7rem', fontWeight: 700, 
                    color: event.severity === 'HIGH' ? '#ef4444' : event.severity === 'MEDIUM' ? '#f59e0b' : '#10b981' 
                  }}>
                    {event.severity}
                  </span>
                </div>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.78rem', lineHeight: '1.5' }}>
                  {event.description || 'Processing context...'}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', padding: '2rem' }}>
            {loading ? 'Loading context summaries...' : 'Context summaries will appear after video processing completes.'}
          </p>
        )}
      </div>
    </div>
  );
};
