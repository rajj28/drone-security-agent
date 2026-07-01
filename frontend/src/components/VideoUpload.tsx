import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, CheckCircle, AlertTriangle, RefreshCw } from 'lucide-react';

interface VideoUploadProps {
  apiBase: string;
  onUploadSuccess: (sessionId: string) => void;
  activeSessionId: string | null;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({ apiBase, onUploadSuccess, activeSessionId }) => {
  const [file, setFile] = useState<File | null>(null);
  const [strategy, setStrategy] = useState<string>('hybrid');
  const [maxFrames, setMaxFrames] = useState<number>(20);
  const [uploading, setUploading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Processing status states
  const [statusData, setStatusData] = useState<any>(null);
  const [polling, setPolling] = useState<boolean>(false);
  const [funnyLineIndex, setFunnyLineIndex] = useState<number>(0);
  const [showMusicPrompt, setShowMusicPrompt] = useState<boolean>(false);
  const [musicPlaying, setMusicPlaying] = useState<boolean>(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  
  const funnyLines = [
    "☕ Grabbing coffee while the AI does the heavy lifting...",
    "🤖 Not a bad engineer, just waiting on API calls...",
    "🧠 Teaching AI to spot thieves... it's a slow learner",
    "🐌 Free tier speed. Premium results though!",
    "🎬 Hollywood editors take longer, trust me",
    "🔍 Analyzing pixels with the intensity of a cat watching a laser",
    "💡 Fun fact: This AI has analyzed more frames than you've scrolled today",
    "⚡ If this was paid tier, we'd be done by now... just saying",
    "🎯 Precision takes time. Especially at 0 cost.",
    "🤔 The AI is thinking really hard about your video...",
    "🏃 Running faster than my last deployment...",
    "🎪 Meanwhile, the agents are having a debate about threat levels",
    "🔬 CSI-level analysis happening right now. Without the dramatic zoom.",
    "🎲 Rolling dice... just kidding, it's actual machine learning",
    "📡 Pinging satellites... okay not really, just Groq's servers",
    "🧑‍💻 The code is working. I'm just as surprised as you.",
    "🍕 This would be faster if I ordered pizza and waited for delivery",
    "🦾 3 AI agents are arguing about your video right now",
    "🎭 Plot twist: the AI found the security threat was the friends we made along the way",
    "🏗️ Rome wasn't built in a day. This pipeline takes ~2 minutes.",
    "🧊 Cooling down the GPU... figuratively. It's all cloud.",
    "🎸 *elevator music plays*",
    "🦊 The AI is being sneakier than the thieves in your video",
    "📝 Dear diary, today I processed 13 frames and only hit 4 rate limits",
    "🌙 If you started this at night, go touch grass. Oh wait, it's dark.",
    "🤹 Juggling 3 different AI providers. No pressure.",
    "🧪 Science takes time. So does calling Groq 13 times on free tier.",
    "🎯 Accuracy > Speed. That's what I tell my manager anyway.",
    "🐢 Slow and steady wins the security analysis race",
    "💪 Almost there! The AI just needs one more coffee break",
    "🔮 Predicting threats... and also predicting you're getting impatient",
    "🎪 Behind the scenes: 550 billion parameters are working for free",
    "🧲 Attracting insights from your video like a magnet",
    "🌈 Every frame analyzed brings us closer to catching bad guys",
    "🎬 Director's cut: This scene takes longer in post-production",
  ];
  
  // Rotate funny lines every 5 seconds while processing
  useEffect(() => {
    let interval: any;
    if (statusData && statusData.status === 'processing') {
      interval = setInterval(() => {
        setFunnyLineIndex(prev => (prev + 1) % funnyLines.length);
      }, 5000);
      // Show music prompt when processing starts
      if (!showMusicPrompt && !musicPlaying) {
        setShowMusicPrompt(true);
      }
    }
    // Stop music when completed or failed
    if (statusData && (statusData.status === 'completed' || statusData.status === 'failed')) {
      if (audioRef.current) {
        audioRef.current.pause();
        setMusicPlaying(false);
      }
    }
    return () => { if (interval) clearInterval(interval); };
  }, [statusData?.status]);

  const handlePlayMusic = () => {
    if (!audioRef.current) {
      audioRef.current = new Audio('/sunflower.mp3');
      audioRef.current.loop = true;
      audioRef.current.volume = 0.3;
    }
    audioRef.current.play();
    setMusicPlaying(true);
    setShowMusicPrompt(false);
  };

  const handleDeclineMusic = () => {
    setShowMusicPrompt(false);
  };

  const handleToggleMusic = () => {
    if (audioRef.current) {
      if (musicPlaying) {
        audioRef.current.pause();
        setMusicPlaying(false);
      } else {
        audioRef.current.play();
        setMusicPlaying(true);
      }
    }
  };
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Poll status when a session is active and processing
  useEffect(() => {
    let intervalId: any;
    
    const checkStatus = async (sessionId: string) => {
      try {
        const res = await fetch(`${apiBase}/processing-status/${sessionId}`);
        if (res.status === 200) {
          const data = await res.json();
          setStatusData(data);
          
          if (data.status === 'completed' || data.status === 'failed') {
            setPolling(false);
          }
        }
      } catch (err) {
        console.error("Error checking status:", err);
      }
    };

    if (activeSessionId) {
      // Fetch status initially
      checkStatus(activeSessionId);
      
      // Setup interval to poll
      setPolling(true);
      intervalId = setInterval(() => {
        checkStatus(activeSessionId);
      }, 5000);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [activeSessionId, apiBase]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      validateAndSetFile(droppedFile);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (selectedFile: File) => {
    const ext = selectedFile.name.substring(selectedFile.name.lastIndexOf('.')).toLowerCase();
    const allowed = ['.mp4', '.avi', '.mov', '.dav', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.mpg', '.3gp', '.ts', '.m4v', '.m2ts'];
    
    if (!allowed.includes(ext)) {
      setError(`Unsupported file type '${ext}'. Please select a valid security video.`);
      setFile(null);
      return;
    }
    
    setError(null);
    setFile(selectedFile);
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError(null);
    setStatusData(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const uploadUrl = new URL(`${apiBase}/upload-video`);
      uploadUrl.searchParams.append('extraction_strategy', strategy);
      uploadUrl.searchParams.append('max_frames', maxFrames.toString());

      const res = await fetch(uploadUrl.toString(), {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || 'Upload failed');
      }

      const data = await res.json();
      onUploadSuccess(data.session_id);
      setFile(null); // Clear selected file after success
    } catch (err: any) {
      setError(err.message || 'An error occurred during video upload.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="view-body">
      <div className="premium-card">
        <h3 className="card-title" style={{ fontSize: '1.1rem', marginBottom: '1.5rem' }}>
          📹 Video Upload & Pipeline Controls
        </h3>

        {/* Info Banner */}
        <div style={{ padding: '0.75rem 1rem', background: 'rgba(99, 102, 241, 0.06)', border: '1px solid rgba(99, 102, 241, 0.15)', borderRadius: '8px', marginBottom: '1.25rem', fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
          <strong style={{ color: 'var(--primary)' }}>⚡ How it works:</strong> Upload any security video (up to ~30MB works best). 
          AI extracts key frames, runs VLM analysis, generates alerts & indexes for semantic search. 
          Takes 1-3 min on free tier — grab a coffee or enjoy tunes while we cook! 🎵
        </div>
        
        <input 
          type="file" 
          ref={fileInputRef} 
          onChange={handleFileChange} 
          style={{ display: 'none' }}
          accept=".mp4,.avi,.mov,.dav,.mkv,.wmv,.flv,.webm,.ts"
        />

        {/* Split: Sample Video + Upload Zone */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
          {/* Left: Sample Video */}
          <div style={{ padding: '1.25rem', background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '10px', textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🎬</div>
            <h4 style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.5rem' }}>Don't have a video?</h4>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.75rem', lineHeight: '1.5' }}>
              Download our sample theft video (29MB), then upload it on the right →
            </p>
            <a 
              href="/sample-video.mp4" 
              download="Sneaky-Thieves-Sample.mp4"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', padding: '0.5rem 1rem', background: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: '6px', color: '#10b981', fontSize: '0.78rem', fontWeight: 700, textDecoration: 'none' }}
            >
              ⬇️ Download Sample (29MB)
            </a>
            <p style={{ fontSize: '0.63rem', color: 'var(--text-muted)', marginTop: '0.5rem', fontStyle: 'italic' }}>
              "Sneaky Thieves Caught Stealing Phones On Camera"
            </p>
          </div>

          {/* Right: Upload Zone */}
          <div 
            className="upload-container" 
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={triggerFileSelect}
            style={{ margin: 0 }}
          >
            <UploadCloud className="upload-icon" size={40} />
            <div className="upload-text">
              {file ? (
                <p style={{ fontWeight: 600, color: 'var(--primary)', fontSize: '0.85rem' }}>{file.name}</p>
              ) : (
                <p style={{ fontWeight: 600, fontSize: '0.85rem' }}>Drag & drop here, or click to browse</p>
              )}
              <p className="upload-hint" style={{ fontSize: '0.7rem' }}>
                MP4, AVI, MOV, MKV, DAV (≤30MB for smooth run)
              </p>
            </div>
          </div>
        </div>

        {error && (
          <div style={{ color: 'var(--color-critical)', fontSize: '0.9rem', marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertTriangle size={16} /> {error}
          </div>
        )}

        <div className="upload-param-grid">
          <div className="input-group">
            <label className="input-label">Frame Extraction Strategy</label>
            <select 
              className="custom-select" 
              value={strategy} 
              onChange={(e) => setStrategy(e.target.value)}
              style={{ width: '100%' }}
            >
              <option value="hybrid">🧠 Hybrid (Motion + Scene changes - Recommended)</option>
              <option value="motion_based">🎬 Motion-Based (Significant activity)</option>
              <option value="scene_change">🎭 Scene Change (Distinct screen cuts)</option>
              <option value="uniform">⏱️ Uniform (Equal time intervals)</option>
            </select>
          </div>

          <div className="input-group">
            <label className="input-label">Maximum Frames to Extract</label>
            <div className="slider-container">
              <input 
                type="range" 
                className="custom-range" 
                min={5} 
                max={20} 
                step={5}
                value={maxFrames}
                onChange={(e) => setMaxFrames(parseInt(e.target.value))}
              />
              <span className="slider-val">{maxFrames}</span>
            </div>
          </div>
        </div>

        <button 
          className="btn btn-primary" 
          disabled={!file || uploading} 
          onClick={handleUpload}
          style={{ width: '100%' }}
        >
          {uploading ? (
            <>
              <RefreshCw className="animate-spin" size={16} /> Uploading Video...
            </>
          ) : (
            '🚀 Initialize AI Security Pipeline'
          )}
        </button>
      </div>

      {statusData && (
        <div className="premium-card processing-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h4 style={{ fontWeight: 700, fontSize: '0.95rem' }}>
              Pipeline Status: <span style={{ textTransform: 'capitalize', color: 'var(--primary)' }}>{statusData.status}</span>
            </h4>
            {polling && (
              <span className="status-value" style={{ color: 'var(--text-muted)' }}>
                <RefreshCw size={14} className="animate-spin" /> Live Polling...
              </span>
            )}
          </div>

          <p className="status-label" style={{ marginBottom: '0.5rem' }}>
            Current Step: <strong style={{ color: 'var(--text-main)' }}>{statusData.current_step.replace(/_/g, ' ')}</strong>
          </p>

          <div className="progress-track">
            <div className="progress-bar" style={{ width: `${statusData.progress || 0}%` }}></div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            <span>Start</span>
            <span>{statusData.progress || 0}%</span>
            <span>Done</span>
          </div>
          {statusData.status === 'processing' && (
            <p style={{ fontSize: '0.8rem', color: '#818cf8', fontStyle: 'italic', textAlign: 'center', marginBottom: '1rem', minHeight: '1.2rem', transition: 'opacity 0.3s' }}>
              {funnyLines[funnyLineIndex]}
            </p>
          )}

          {/* Music Prompt */}
          {showMusicPrompt && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', padding: '0.75rem', background: 'rgba(99, 102, 241, 0.08)', borderRadius: '8px', marginBottom: '1rem', border: '1px solid rgba(99, 102, 241, 0.2)' }}>
              <span style={{ fontSize: '0.85rem' }}>🎵 Want some vibes while you wait?</span>
              <button onClick={handlePlayMusic} style={{ padding: '0.3rem 0.75rem', background: 'var(--primary)', color: 'white', border: 'none', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 700, cursor: 'pointer' }}>Play Sunflower 🌻</button>
              <button onClick={handleDeclineMusic} style={{ padding: '0.3rem 0.75rem', background: 'transparent', color: 'var(--text-muted)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px', fontSize: '0.75rem', cursor: 'pointer' }}>Nah</button>
            </div>
          )}

          {/* Music Toggle */}
          {audioRef.current && !showMusicPrompt && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <span style={{ fontSize: '0.75rem', color: '#818cf8' }}>🎶 Sunflower — Post Malone & Swae Lee</span>
              <button onClick={handleToggleMusic} style={{ padding: '0.2rem 0.5rem', background: 'transparent', color: '#818cf8', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '4px', fontSize: '0.7rem', cursor: 'pointer' }}>
                {musicPlaying ? '⏸ Pause' : '▶ Play'}
              </button>
            </div>
          )}

          <h5 style={{ fontSize: '0.85rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
            Completed Tasks
          </h5>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {statusData.processing_steps && statusData.processing_steps.map((step: string) => (
              <div key={step} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                <CheckCircle size={14} style={{ color: 'var(--color-clear)' }} />
                <span>{step.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
              </div>
            ))}
            
            {statusData.status === 'failed' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-critical)', fontSize: '0.85rem' }}>
                <AlertTriangle size={14} />
                <span>Error: {statusData.error}</span>
              </div>
            )}

            {statusData.status === 'completed' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-clear)', fontSize: '0.85rem', fontWeight: 600 }}>
                <CheckCircle size={14} />
                <span>Video processing completed successfully! All assets ready.</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
