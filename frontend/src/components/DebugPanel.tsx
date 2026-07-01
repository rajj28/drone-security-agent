import { useState, useEffect } from 'react';
import { RefreshCw, CheckCircle, AlertTriangle, XCircle, Trash2, Bug } from 'lucide-react';

interface DebugPanelProps {
  apiBase: string;
}

export const DebugPanel: React.FC<DebugPanelProps> = ({ apiBase }) => {
  const [debugData, setDebugData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [clearMsg, setClearMsg] = useState('');

  const fetchDebug = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/debug`);
      if (res.ok) {
        setDebugData(await res.json());
      }
    } catch (err) {
      console.error("Debug fetch failed:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchDebug(); }, []);

  const handleClearSessions = async () => {
    if (!confirm('This will delete ALL sessions from MongoDB and disk. Are you sure?')) return;
    setClearing(true);
    try {
      const res = await fetch(`${apiBase}/debug/clear-sessions`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setClearMsg(data.message);
      }
    } catch (err) {
      setClearMsg('Failed to clear sessions');
    } finally {
      setClearing(false);
    }
  };

  const statusIcon = (status: string) => {
    if (status === 'ok' || status === 'configured') return <CheckCircle size={16} style={{ color: '#10b981' }} />;
    if (status === 'quota_exhausted' || status === 'degraded') return <AlertTriangle size={16} style={{ color: '#f59e0b' }} />;
    if (status === 'error' || status === 'invalid_key' || status === 'missing') return <XCircle size={16} style={{ color: '#ef4444' }} />;
    return <XCircle size={16} style={{ color: 'var(--text-muted)' }} />;
  };

  const statusColor = (status: string) => {
    if (status === 'ok' || status === 'configured') return '#10b981';
    if (status === 'quota_exhausted' || status === 'degraded') return '#f59e0b';
    if (status === 'error' || status === 'invalid_key' || status === 'missing') return '#ef4444';
    return 'var(--text-muted)';
  };

  return (
    <div className="view-body">
      <div className="premium-card" style={{ overflow: 'visible' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h3 className="card-title" style={{ fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Bug size={18} /> System Debug & Health Check
          </h3>
          <button className="btn btn-secondary" onClick={fetchDebug} disabled={loading} style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem' }}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>

        {debugData && (
          <>
            {/* Overall Status */}
            <div style={{ 
              padding: '1rem', marginBottom: '1.5rem', borderRadius: '8px',
              background: debugData.overall === 'healthy' ? 'rgba(16,185,129,0.08)' : debugData.overall === 'degraded' ? 'rgba(245,158,11,0.08)' : 'rgba(239,68,68,0.08)',
              border: `1px solid ${debugData.overall === 'healthy' ? 'rgba(16,185,129,0.3)' : debugData.overall === 'degraded' ? 'rgba(245,158,11,0.3)' : 'rgba(239,68,68,0.3)'}`
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, fontSize: '0.95rem', color: debugData.overall === 'healthy' ? '#10b981' : debugData.overall === 'degraded' ? '#f59e0b' : '#ef4444' }}>
                {debugData.overall === 'healthy' ? '✅' : debugData.overall === 'degraded' ? '⚠️' : '❌'} 
                System: {debugData.overall.toUpperCase()}
              </div>
              {debugData.overall !== 'healthy' && (
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                  {debugData.overall === 'degraded' ? 'Some services have issues but fallbacks are active.' : 'Critical issues detected — pipeline may not work.'}
                </p>
              )}
            </div>

            {/* Service Checks */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1.5rem' }}>
              {debugData.checks.map((check: any, idx: number) => (
                <div key={idx} style={{ 
                  display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.6rem 0.75rem',
                  background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)'
                }}>
                  {statusIcon(check.status)}
                  <div style={{ flex: 1 }}>
                    <span style={{ fontWeight: 600, fontSize: '0.82rem' }}>{check.name}</span>
                    {check.detail && <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginLeft: '0.5rem' }}>— {check.detail}</span>}
                  </div>
                  <span style={{ fontSize: '0.7rem', fontWeight: 700, color: statusColor(check.status), textTransform: 'uppercase' }}>
                    {check.status.replace('_', ' ')}
                  </span>
                </div>
              ))}
            </div>

            {/* Config */}
            <div style={{ padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', marginBottom: '1.5rem' }}>
              <h4 style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.5rem' }}>Current Config</h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.3rem', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                <span style={{ color: 'var(--text-muted)' }}>Vision Provider:</span>
                <span style={{ color: '#818cf8' }}>{debugData.config.vision_provider}</span>
                <span style={{ color: 'var(--text-muted)' }}>Agent LLM:</span>
                <span style={{ color: '#818cf8' }}>{debugData.config.agent_llm_provider}</span>
                <span style={{ color: 'var(--text-muted)' }}>Max Frames:</span>
                <span>{debugData.config.max_frames}</span>
                <span style={{ color: 'var(--text-muted)' }}>Vision Workers:</span>
                <span>{debugData.config.max_vision_workers}</span>
              </div>
            </div>

            {/* Tips */}
            <div style={{ padding: '1rem', background: 'rgba(99,102,241,0.05)', borderRadius: '8px', border: '1px solid rgba(99,102,241,0.1)', marginBottom: '1.5rem' }}>
              <h4 style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.5rem' }}>💡 Troubleshooting Tips</h4>
              <ul style={{ fontSize: '0.75rem', color: 'var(--text-muted)', paddingLeft: '1rem', lineHeight: '2' }}>
                <li><strong>Quota exhausted:</strong> {debugData.tips.quota_exhausted}</li>
                <li><strong>Model degraded:</strong> {debugData.tips.degraded}</li>
                <li><strong>Processing slow:</strong> {debugData.tips.vision_slow}</li>
              </ul>
            </div>
          </>
        )}

        {/* Actions */}
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '1rem' }}>
          <h4 style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.75rem' }}>🔧 Actions</h4>
          <button 
            className="btn btn-secondary" 
            onClick={handleClearSessions} 
            disabled={clearing}
            style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
          >
            <Trash2 size={14} /> {clearing ? 'Clearing...' : 'Clear All Sessions'}
          </button>
          {clearMsg && <p style={{ fontSize: '0.75rem', color: '#10b981', marginTop: '0.5rem' }}>{clearMsg}</p>}
        </div>

        {/* Troubleshooting Guide */}
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '1.5rem', marginTop: '1.5rem' }}>
          <h4 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '1rem' }}>📖 When to Debug — Quick Reference</h4>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.78rem' }}>
            <div style={{ padding: '0.6rem 0.75rem', background: 'rgba(239,68,68,0.05)', border: '1px solid rgba(239,68,68,0.15)', borderRadius: '6px' }}>
              <strong style={{ color: '#ef4444' }}>Pipeline stuck at 40% (Vision stage)</strong>
              <p style={{ color: 'var(--text-muted)', marginTop: '0.2rem' }}>Groq daily token limit (500K) exhausted. Fix: Wait 24h, OR use a new Groq account (different email), OR change <code>VISION_PROVIDER=gemini</code> in .env and restart server.</p>
            </div>

            <div style={{ padding: '0.6rem 0.75rem', background: 'rgba(245,158,11,0.05)', border: '1px solid rgba(245,158,11,0.15)', borderRadius: '6px' }}>
              <strong style={{ color: '#f59e0b' }}>503 Service Unavailable (NVIDIA)</strong>
              <p style={{ color: 'var(--text-muted)', marginTop: '0.2rem' }}>NVIDIA NIM model temporarily down. Auto-retries built in. If persistent, wait 1-2 hours. System falls back to Groq automatically.</p>
            </div>

            <div style={{ padding: '0.6rem 0.75rem', background: 'rgba(245,158,11,0.05)', border: '1px solid rgba(245,158,11,0.15)', borderRadius: '6px' }}>
              <strong style={{ color: '#f59e0b' }}>Frame analysis empty / same reasoning</strong>
              <p style={{ color: 'var(--text-muted)', marginTop: '0.2rem' }}>Vision API failed silently. Check Groq status above. If "quota_exhausted" — that's the cause.</p>
            </div>

            <div style={{ padding: '0.6rem 0.75rem', background: 'rgba(99,102,241,0.05)', border: '1px solid rgba(99,102,241,0.15)', borderRadius: '6px' }}>
              <strong style={{ color: '#818cf8' }}>Alert Center shows 0 alerts</strong>
              <p style={{ color: 'var(--text-muted)', marginTop: '0.2rem' }}>Alerts depend on vision data. If vision failed → no analysis → no alerts. Fix the vision provider first.</p>
            </div>

            <div style={{ padding: '0.6rem 0.75rem', background: 'rgba(99,102,241,0.05)', border: '1px solid rgba(99,102,241,0.15)', borderRadius: '6px' }}>
              <strong style={{ color: '#818cf8' }}>Semantic Search "No Matches"</strong>
              <p style={{ color: 'var(--text-muted)', marginTop: '0.2rem' }}>Frames aren't indexed in Pinecone yet. Pipeline must complete fully (100%). If it failed before indexing, re-upload.</p>
            </div>

            <div style={{ padding: '0.6rem 0.75rem', background: 'rgba(99,102,241,0.05)', border: '1px solid rgba(99,102,241,0.15)', borderRadius: '6px' }}>
              <strong style={{ color: '#818cf8' }}>Security Agent error / 503</strong>
              <p style={{ color: 'var(--text-muted)', marginTop: '0.2rem' }}>LLM provider failing. Check NVIDIA status above. Fallback to Groq is automatic.</p>
            </div>

            <div style={{ padding: '0.6rem 0.75rem', background: 'rgba(16,185,129,0.05)', border: '1px solid rgba(16,185,129,0.15)', borderRadius: '6px' }}>
              <strong style={{ color: '#10b981' }}>Old session keeps appearing</strong>
              <p style={{ color: 'var(--text-muted)', marginTop: '0.2rem' }}>Click "Clear All Sessions" above, then hard-refresh browser (Ctrl+Shift+R).</p>
            </div>

            <div style={{ padding: '0.6rem 0.75rem', background: 'rgba(16,185,129,0.05)', border: '1px solid rgba(16,185,129,0.15)', borderRadius: '6px' }}>
              <strong style={{ color: '#10b981' }}>Frontend says "Connection Refused"</strong>
              <p style={{ color: 'var(--text-muted)', marginTop: '0.2rem' }}>Backend not running. Start it: <code>venv\Scripts\python.exe -m uvicorn src.api:app --host 0.0.0.0 --port 8000</code></p>
            </div>
          </div>

          {/* Quick .env Guide */}
          <div style={{ marginTop: '1.25rem', padding: '1rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px' }}>
            <h5 style={{ fontSize: '0.8rem', fontWeight: 700, marginBottom: '0.5rem' }}>⚙️ Key .env Changes (restart server after)</h5>
            <pre style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: '#a7f3d0', lineHeight: '1.8', margin: 0 }}>
{`# Switch vision if Groq exhausted:
VISION_PROVIDER=gemini

# Reduce frames if slow:
MAX_FRAMES=10

# Switch orchestration if NVIDIA down:
AGENT_LLM_PROVIDER=groq`}
            </pre>
          </div>

          {/* Nuclear Option */}
          <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', background: 'rgba(239,68,68,0.05)', border: '1px solid rgba(239,68,68,0.15)', borderRadius: '6px' }}>
            <strong style={{ fontSize: '0.8rem', color: '#ef4444' }}>☢️ Nuclear Reset (if everything fails)</strong>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.3rem', lineHeight: '1.6' }}>
              1. Stop server (Ctrl+C)<br/>
              2. Click "Clear All Sessions" above<br/>
              3. Restart server<br/>
              4. Hard-refresh browser (Ctrl+Shift+R)<br/>
              5. Upload video again
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
