import { useState, useEffect, useRef } from 'react';
import { Play, Pause, SkipForward, X, Volume2 } from 'lucide-react';

interface GuidedTourProps {
  onNavigate: (tab: string) => void;
  onClose: () => void;
  apiBase: string;
  onSessionCreated: (sessionId: string) => void;
}

const tourSteps = [
  {
    tab: 'upload',
    title: 'Video Upload',
    scrollTo: '.premium-card',
    action: 'upload', // Special action: auto-upload sample video
    speech: "Welcome to the Drone Security Analyst Agent. Let me show you the full system in action. I'm going to upload our sample theft video right now — real CCTV footage of phone thieves caught in a retail store. Watch the progress bar — our AI pipeline extracts key frames using hybrid motion detection, then analyzes each one. This takes about 2 minutes, and the tour continues automatically when it finishes.",
    highlight: "Auto-uploading sample video. Pipeline: Extract → Telemetry → VLM → Index → Alerts → Summary",
  },
  {
    tab: 'upload',
    title: 'Processing...',
    scrollTo: '.processing-card',
    action: 'wait', // Special action: wait for pipeline to finish
    speech: "The pipeline is running. Frame extraction uses motion detection to find the most important moments — no boring static frames. Then Gemini's vision model analyzes each frame, describing what it sees. After that, frames get indexed in Pinecone for semantic search, alerts are generated, and a session summary is created. Sit tight — I'll continue the tour once everything's ready.",
    highlight: "Pipeline running: Extraction → Telemetry → Vision Analysis → Indexing → Alerts → Summary",
  },
  {
    tab: 'frames',
    title: 'Frame Analysis',
    scrollTo: '.premium-card',
    domActions: [
      { type: 'click', selector: '[class*="nav-item"]', delay: 1500 },
      { type: 'scroll', selector: 'details', delay: 6000 },
    ],
    waitAfter: 4000,
    speech: "Here we go — Frame Analysis. Each frame was analyzed by Gemini's vision model — with optional CLIP and BLIP cross-checks when Cloud Enhancers are enabled. Click any frame on the left sidebar. You see the actual image, drone telemetry data — GPS, altitude, speed — and the AI Scene Vision panel with people count, scene description, and detected objects in green. The reasoning section explains the threat logic. And scroll down — there's the full raw JSON payload with every data point the AI extracted.",
    highlight: "Per-frame: VLM description, threat assessment, reasoning, detected objects, telemetry data.",
  },
  {
    tab: 'alerts',
    title: 'Alert Center',
    scrollTo: '.premium-card',
    speech: "Alert Center — two-layer security system. First layer checks rules: after-hours activity, restricted zones, shoplifting patterns, suspicious hand movements. Second layer — a Gemini-powered LLM validates each alert and can escalate or dismiss. Only MEDIUM, HIGH, and CRITICAL alerts show here. Each alert has the frame, timestamp, and AI reasoning.",
    highlight: "Two-layer alert system: Rule-based + LLM-validated. Filtered to MEDIUM/HIGH/CRITICAL.",
  },
  {
    tab: 'search',
    title: 'Semantic Search',
    scrollTo: '.search-bar-row',
    domActions: [
      { type: 'type', selector: '.search-input', text: 'person stealing phone', delay: 2000 },
      { type: 'click', selector: '.btn-primary', delay: 5000 },
    ],
    waitAfter: 10000, // Wait 10s for search results to load
    speech: "Semantic Search — the cross-domain indexing feature. Watch — I'll type 'person stealing phone' and search. Every frame is embedded into a Pinecone vector database. The system returns the most semantically similar frames with similarity scores. No keyword matching — pure AI understanding. This is what the assignment specifically asked for.",
    highlight: "Vector search over Pinecone DB. Natural language queries return semantically similar frames.",
  },
  {
    tab: 'summary',
    title: 'Session Summary',
    scrollTo: '.premium-card',
    speech: "Session Summary — the executive overview. Total frames analyzed, alerts triggered, and a chronological timeline of key security events with severity levels. One page that answers: what happened, when, and how serious.",
    highlight: "Executive overview: frame count, alert count, key events timeline with severity levels.",
  },
  {
    tab: 'chat',
    title: 'Security Agent',
    scrollTo: '.premium-card',
    domActions: [
      { type: 'type', selector: 'input[type="text"], textarea', text: 'Was there any theft detected?', delay: 2000 },
      { type: 'click', selector: 'button[type="submit"], .btn-primary', delay: 5500 },
    ],
    waitAfter: 20000, // Wait 20s for AI response
    speech: "Security Agent — conversational AI for follow-up questions. Watch — I'll ask: 'Was there any theft detected?'. It retrieves relevant frame data from Pinecone, combines it with the full analysis context, and generates a concise answer. Powered by Gemini with retrieval-augmented context. This is the bonus Q&A feature the assignment mentioned.",
    highlight: "Natural language Q&A over video data. Evidence-based answers with source retrieval.",
  },
  {
    tab: 'orchestration',
    title: 'AI Orchestration',
    scrollTo: '.premium-card',
    speech: "Multi-Agent Orchestration — three specialized agents coordinated by a central orchestrator. Analysis Agent for pattern recognition, QA Agent for validation, Question Agent for interaction. All running on Gemini. The architecture diagram shows the full data flow. Production-grade design.",
    highlight: "Multi-agent system: Analysis + QA + Question agents coordinated by a Gemini orchestrator.",
  },
  {
    tab: 'debug',
    title: 'Debug Panel',
    scrollTo: '.premium-card',
    speech: "Finally, the Debug panel. Live health checks for every API provider, current config, and a full troubleshooting guide. If anything breaks, this page tells you exactly what and how to fix it. That concludes the tour — the system is fully operational. Thanks for watching!",
    highlight: "Live API health checks, system config, troubleshooting guide, and session management.",
  },
];

export const GuidedTour: React.FC<GuidedTourProps> = ({ onNavigate, onClose, apiBase, onSessionCreated }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [speaking, setSpeaking] = useState(false);
  const [paused, setPaused] = useState(false);
  const [autoMode, setAutoMode] = useState(false);
  const [tourFinished, setTourFinished] = useState(false);
  const [waitingForPipeline, setWaitingForPipeline] = useState(false);
  const [pipelineProgress, setPipelineProgress] = useState(0);
  const [pipelineSessionId, setPipelineSessionId] = useState<string | null>(null);
  const synthRef = useRef<SpeechSynthesisUtterance | null>(null);
  const pollRef = useRef<any>(null);

  const step = tourSteps[currentStep];

  useEffect(() => {
    onNavigate(step.tab);
    
    // Wait for tab to render before doing anything
    setTimeout(() => {
      // Auto-scroll to highlighted element
      if (step.scrollTo) {
        const el = document.querySelector(step.scrollTo);
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
      // Auto-open collapsible details elements on the page
      document.querySelectorAll('details').forEach(d => d.setAttribute('open', ''));
      
      // Start narration AFTER tab is visible
      const currentStepData = tourSteps[currentStep];
      if (autoMode && currentStepData.action !== 'wait') {
        speak(tourSteps[currentStep].speech);
      }
    }, 800);

    // Execute DOM actions with proper delays (after tab renders)
    const currentStepData = tourSteps[currentStep];
    if (currentStepData.domActions && autoMode) {
      currentStepData.domActions.forEach((action: any) => {
        setTimeout(() => {
          if (action.type === 'click') {
            const el = document.querySelector(action.selector) as HTMLElement;
            if (el) el.click();
          } else if (action.type === 'type') {
            const el = document.querySelector(action.selector) as HTMLInputElement;
            if (el) {
              el.focus();
              el.value = '';
              let i = 0;
              const typeInterval = setInterval(() => {
                if (i < action.text.length) {
                  const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
                  if (nativeInputValueSetter) {
                    nativeInputValueSetter.call(el, action.text.substring(0, i + 1));
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                  }
                  i++;
                } else {
                  clearInterval(typeInterval);
                }
              }, 80);
            }
          } else if (action.type === 'scroll') {
            const el = document.querySelector(action.selector);
            if (el) {
              el.setAttribute('open', '');
              el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
          }
        }, action.delay + 800); // Add 800ms base for tab render
      });
    }

    // Handle wait action
    if (currentStepData.action === 'wait' && autoMode && pipelineSessionId) {
      // Already polling
    }
  }, [currentStep]);

  const autoUploadSampleVideo = async () => {
    if (pipelineSessionId) return; // Already started
    try {
      const uploadRes = await fetch(`${apiBase}/upload-sample-video?extraction_strategy=hybrid&max_frames=15`, {
        method: 'POST',
      });
      
      if (uploadRes.ok) {
        const data = await uploadRes.json();
        setPipelineSessionId(data.session_id);
        onSessionCreated(data.session_id);
        startPolling(data.session_id);
      } else {
        console.error("Sample upload failed:", await uploadRes.text());
      }
    } catch (err) {
      console.error("Auto-upload failed:", err);
    }
  };

  const startPolling = (sessionId: string) => {
    setWaitingForPipeline(true);
    setPipelineProgress(0);
    speak(tourSteps[1].speech);
    
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${apiBase}/processing-status/${sessionId}`);
        if (res.ok) {
          const data = await res.json();
          setPipelineProgress(data.progress || 0);
          if (data.status === 'completed') {
            clearInterval(pollRef.current);
            setWaitingForPipeline(false);
            setPipelineProgress(100);
            setTimeout(() => setCurrentStep(2), 1000);
          } else if (data.status === 'failed') {
            clearInterval(pollRef.current);
            setWaitingForPipeline(false);
            speak("Pipeline had an issue, but let me show you what we have so far.");
            setTimeout(() => setCurrentStep(2), 2000);
          }
        }
      } catch (err) { /* keep polling */ }
    }, 5000);
  };

  useEffect(() => {
    // Load voices (needed for some browsers)
    window.speechSynthesis.getVoices();
    return () => {
      window.speechSynthesis.cancel();
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const speak = (text: string) => {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0; // Normal smooth pace
    utterance.pitch = 1.2; // Young male — not deep, not female
    utterance.volume = 1.0;
    
    // Male voices only, but skip the deep ones
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(v => v.name.includes('Microsoft Mark'))
      || voices.find(v => v.name.includes('Google UK English Male'))
      || voices.find(v => v.name.includes('Microsoft David'))
      || voices.find(v => v.name.includes('Daniel'))
      || voices.find(v => v.name.includes('Male') && v.lang.startsWith('en'))
      || voices.find(v => v.lang === 'en-US');
    if (preferred) utterance.voice = preferred;
    
    utterance.onstart = () => { setSpeaking(true); setPaused(false); };
    utterance.onend = () => { 
      setSpeaking(false); setPaused(false);
      // Auto-advance after narration finishes (respect waitAfter for steps that need results)
      if (currentStep < tourSteps.length - 1) {
        const waitTime = tourSteps[currentStep].waitAfter || 1500;
        setTimeout(() => setCurrentStep(prev => prev + 1), waitTime);
      }
    };
    utterance.onerror = () => { setSpeaking(false); };
    
    synthRef.current = utterance;
    window.speechSynthesis.speak(utterance);
  };

  const handlePlay = () => {
    if (paused) {
      window.speechSynthesis.resume();
      setPaused(false);
    } else {
      speak(tourSteps[currentStep].speech);
    }
  };

  const handlePause = () => {
    window.speechSynthesis.pause();
    setPaused(true);
  };

  const handleNext = () => {
    window.speechSynthesis.cancel();
    setSpeaking(false);
    if (currentStep < tourSteps.length - 1) {
      setCurrentStep(prev => prev + 1);
    } else {
      setAutoMode(false);
      setTourFinished(true);
    }
  };

  const handleReplay = () => {
    setTourFinished(false);
    setCurrentStep(2); // Skip upload + wait steps — reuse existing session
    setAutoMode(true);
    setTimeout(() => speak(tourSteps[2].speech), 600);
  };

  const handleClose = () => {
    window.speechSynthesis.cancel();
    setAutoMode(false);
    if (pollRef.current) clearInterval(pollRef.current);
    onClose();
  };

  return (
    <div style={{
      zIndex: 9999, width: '100%',
      background: 'linear-gradient(135deg, #1e1b4b, #312e81)', 
      border: '1px solid rgba(139, 92, 246, 0.4)',
      borderRadius: '8px', padding: '0.6rem 0.7rem',
      fontSize: '0.7rem',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <Volume2 size={12} style={{ color: '#a78bfa' }} />
          <span style={{ fontSize: '0.6rem', color: '#a78bfa', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.03em' }}>
            Tour {currentStep + 1}/{tourSteps.length}
          </span>
        </div>
        <button onClick={handleClose} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '2px' }}>
          <X size={14} />
        </button>
      </div>

      {/* Current Step */}
      <h4 style={{ fontSize: '0.78rem', fontWeight: 700, color: 'white', marginBottom: '0.2rem' }}>
        {waitingForPipeline ? 'Processing...' : step.title}
      </h4>
      <p style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.7)', lineHeight: '1.4', marginBottom: waitingForPipeline ? '0.4rem' : '0.5rem' }}>
        {waitingForPipeline ? 'The pipeline is running. The tour continues automatically when it finishes.' : step.highlight}
      </p>
      
      {/* Pipeline Progress Bar */}
      {waitingForPipeline && (
        <div style={{ marginBottom: '0.5rem' }}>
          <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${pipelineProgress}%`, background: 'linear-gradient(90deg, #7c3aed, #a78bfa)', borderRadius: '2px', transition: 'width 0.5s ease' }} />
          </div>
          <span style={{ fontSize: '0.6rem', color: 'rgba(255,255,255,0.5)' }}>{pipelineProgress}%</span>
        </div>
      )}

      {/* Controls */}
      {tourFinished ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <button onClick={handleReplay} style={{ display: 'flex', alignItems: 'center', gap: '0.2rem', padding: '0.25rem 0.5rem', background: '#7c3aed', border: 'none', borderRadius: '4px', color: 'white', fontSize: '0.6rem', fontWeight: 600, cursor: 'pointer' }}>
            <Play size={10} /> Replay Tour
          </button>
          <button onClick={handleClose} style={{ display: 'flex', alignItems: 'center', gap: '0.2rem', padding: '0.25rem 0.5rem', background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px', color: 'white', fontSize: '0.6rem', fontWeight: 600, cursor: 'pointer' }}>
            <X size={10} /> Close
          </button>
          <span style={{ fontSize: '0.6rem', color: '#10b981', marginLeft: 'auto' }}>Tour Complete</span>
        </div>
      ) : (
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', flexWrap: 'wrap' }}>
        {speaking && !paused ? (
          <button onClick={handlePause} style={{ display: 'flex', alignItems: 'center', gap: '0.2rem', padding: '0.25rem 0.5rem', background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px', color: 'white', fontSize: '0.6rem', fontWeight: 600, cursor: 'pointer' }}>
            <Pause size={10} /> Pause
          </button>
        ) : (
          <button onClick={handlePlay} style={{ display: 'flex', alignItems: 'center', gap: '0.2rem', padding: '0.25rem 0.5rem', background: '#7c3aed', border: 'none', borderRadius: '4px', color: 'white', fontSize: '0.6rem', fontWeight: 600, cursor: 'pointer' }}>
            <Play size={10} /> {paused ? 'Resume' : 'Narrate'}
          </button>
        )}
        
        <button onClick={() => { setAutoMode(true); speak(tourSteps[currentStep].speech); autoUploadSampleVideo(); }} style={{ display: 'flex', alignItems: 'center', gap: '0.2rem', padding: '0.25rem 0.5rem', background: 'rgba(16,185,129,0.2)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: '4px', color: '#10b981', fontSize: '0.6rem', fontWeight: 600, cursor: 'pointer' }}>
          <Volume2 size={10} /> {autoMode ? 'Auto On' : 'Auto'}
        </button>

        <button onClick={handleNext} style={{ display: 'flex', alignItems: 'center', gap: '0.2rem', padding: '0.25rem 0.5rem', background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px', color: 'white', fontSize: '0.6rem', fontWeight: 600, cursor: 'pointer' }}>
          <SkipForward size={10} /> {currentStep === tourSteps.length - 1 ? 'End' : 'Skip'}
        </button>

        {/* Progress dots */}
        <div style={{ display: 'flex', gap: '3px', marginLeft: 'auto' }}>
          {tourSteps.map((_, idx) => (
            <div key={idx} style={{ 
              width: '4px', height: '4px', borderRadius: '50%',
              background: idx === currentStep ? '#a78bfa' : idx < currentStep ? '#10b981' : 'rgba(255,255,255,0.2)'
            }} />
          ))}
        </div>
      </div>
      )}
    </div>
  );
};
