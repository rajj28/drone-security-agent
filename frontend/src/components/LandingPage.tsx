import { CheckCircle, Shield, Layers } from 'lucide-react';

interface LandingPageProps {
  onEnterDashboard: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onEnterDashboard }) => {

  const requirements = [
    {
      category: "Core Requirements",
      items: [
        { text: "Feature spec (value + key requirements)", done: true, extra: "FEATURE_SPEC.md" },
        { text: "Architecture for telemetry & video pipeline", done: true, extra: "Multi-stage: Extract → Telemetry → VLM → Alerts → Summary" },
        { text: "Prototype implementation", done: true, extra: "Full-stack Python + React" },
        { text: "Video frame analysis with AI", done: true, extra: "Groq VLM (llama-4-scout)" },
        { text: "AI-generated component", done: true, extra: "Multi-agent orchestration (NVIDIA NIM)" },
        { text: "Cross-domain frame indexing", done: true, extra: "Pinecone vector DB + semantic search" },
        { text: "QA test cases", done: true, extra: "42 pytest tests — all passing" },
      ]
    },
    {
      category: "Expected Output",
      items: [
        { text: "Logs: Objects with location & time", done: true, extra: "Per-frame telemetry + VLM descriptions" },
        { text: "Alerts: Security events triggered", done: true, extra: "Rule-based + LLM-validated (MEDIUM/HIGH/CRITICAL)" },
        { text: "Indexed Frames: Queryable by time/object", done: true, extra: "Natural language vector search" },
      ]
    },
    {
      category: "Bonus (Both Implemented)",
      items: [
        { text: "Video summarization", done: true, extra: "AI session summary with key events" },
        { text: "Agent Q&A (follow-up questions)", done: true, extra: "Security Agent chat (NVIDIA Nemotron-550B)" },
      ]
    },
    {
      category: "Beyond Requirements",
      items: [
        { text: "Production React dashboard with real-time status", done: true, extra: "" },
        { text: "Multi-model AI (Groq + NVIDIA NIM + Gemini)", done: true, extra: "" },
        { text: "CLIP + BLIP + HuggingFace API support", done: true, extra: "Cloud enhanced analyzer pipeline" },
        { text: "Multi-agent orchestration with reasoning chains", done: true, extra: "" },
        { text: "AI Learning Agent (LangGraph state machine)", done: true, extra: "" },
        { text: "MongoDB session persistence", done: true, extra: "" },
        { text: "Intelligent frame extraction (motion detection)", done: true, extra: "" },
        { text: "Person tracking across frames", done: true, extra: "" },
        { text: "Behavioral threat analysis (shoplifting detection)", done: true, extra: "" },
        { text: "LLM-validated alert decision making", done: true, extra: "Rule-based + LLM reasoning layer" },
        { text: "Cloud deployment ready (Docker + GCP)", done: true, extra: "" },
      ]
    }
  ];

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-main)', color: 'var(--text-main)' }}>
      
      {/* Top Header */}
      <div style={{ padding: '1rem 2rem', borderBottom: '1px solid rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Shield size={28} style={{ color: 'var(--primary)' }} />
          <div>
            <h1 style={{ fontSize: '1.3rem', fontWeight: 800, margin: 0 }}>Drone Security Analyst Agent</h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>FlytBase AI Engineer Assignment Submission</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className="tech-badge">Groq</span>
          <span className="tech-badge">NVIDIA NIM</span>
          <span className="tech-badge">Pinecone</span>
          <span className="tech-badge">LangChain</span>
          <span className="tech-badge">React</span>
          <button className="btn btn-primary" onClick={onEnterDashboard} style={{ marginLeft: '1rem' }}>
            <Layers size={16} /> Enter Dashboard →
          </button>
        </div>
      </div>

      {/* Side by Side */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', flexGrow: 1, overflow: 'hidden' }}>
        
        {/* Left: PDF */}
        <div style={{ borderRight: '1px solid rgba(255,255,255,0.08)', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '0.75rem 1.5rem', background: 'rgba(255,255,255,0.02)', borderBottom: '1px solid rgba(255,255,255,0.05)', fontWeight: 700, fontSize: '0.9rem' }}>
            📄 Assignment Document (Original PDF)
          </div>
          <iframe 
            src="/assignment.pdf#toolbar=0&view=FitH" 
            style={{ flexGrow: 1, width: '100%', border: 'none', background: '#1a1a2e' }}
            title="Assignment PDF"
          />
        </div>

        {/* Right: Implementation Checklist */}
        <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ padding: '0.75rem 1.5rem', background: 'rgba(255,255,255,0.02)', borderBottom: '1px solid rgba(255,255,255,0.05)', fontWeight: 700, fontSize: '0.9rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>✅ What We Implemented</span>
          </div>
          <div style={{ flexGrow: 1, overflowY: 'auto', padding: '1.5rem' }}>
            {requirements.map((section, sIdx) => (
              <div key={sIdx} style={{ marginBottom: '1.5rem' }}>
                <h3 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--primary)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  {section.category}
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                  {section.items.map((item, iIdx) => (
                    <div key={iIdx} style={{ 
                      display: 'flex', alignItems: 'flex-start', gap: '0.5rem', 
                      padding: '0.5rem 0.75rem', 
                      background: 'rgba(16, 185, 129, 0.03)', 
                      borderRadius: '6px',
                      border: '1px solid rgba(16, 185, 129, 0.1)'
                    }}>
                      <CheckCircle size={15} style={{ color: '#10b981', marginTop: '2px', flexShrink: 0 }} />
                      <div>
                        <span style={{ fontWeight: 600, fontSize: '0.82rem' }}>{item.text}</span>
                        {item.extra && <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginLeft: '0.5rem' }}>— {item.extra}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {/* Architecture */}
            <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px' }}>
              <h3 style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.5rem' }}>Architecture</h3>
              <pre style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: '#a7f3d0', lineHeight: '1.5', margin: 0 }}>
{`Video → Frame Extractor (Hybrid Motion) → Groq VLM Analysis
  → Pinecone Index → Alert Engine (Rules + NVIDIA NIM)
  → Session Summary → React Dashboard`}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
