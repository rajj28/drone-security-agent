import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, CheckCircle, AlertTriangle, RefreshCw, Music, Square, Sparkles } from 'lucide-react';

const FUNNY_LINES = [
  "Teaching the AI the difference between a burglar and a very committed raccoon...",
  "Enhancing... enhancing... okay fine, it never works like in the movies.",
  "Asking each pixel individually if it saw anything suspicious...",
  "Running facial recognition on a guy who is 90% hoodie...",
  "Consulting three neural networks and one magic 8-ball...",
  "The suspect walked left. Then right. Riveting footage, honestly.",
  "Cross-referencing your video with every heist movie ever made...",
  "Zooming in without losing quality... we lost some quality.",
  "Politely asking the motion detector to calm down about that plastic bag...",
  "Our vision model just gasped. Could be nothing. Could be everything.",
  "Detecting sneakiness levels... currently at 'cartoon villain'.",
  "Reviewing frames like a detective with a corkboard and too much string...",
  "Interrogating pixels. So far, they're not talking.",
  "That's not a ghost, that's lens flare. Probably. Double-checking anyway.",
  "Your video is being watched more closely than a season finale...",
  "Somewhere in the cloud, a GPU is sweating.",
  "The AI paused to admire your camera work. Back to crime-spotting now.",
  "Suspicious activity detected: this progress bar's relationship with the truth.",
  "Frame 12 looks shifty. Frame 13 is an accomplice. Investigating...",
  "Aligning satellites... just kidding, we don't have satellites. Analyzing frames.",
];

interface VideoUploadProps {
  apiBase: string;
  onUploadSuccess: (sessionId: string) => void;
  activeSessionId: string | null;
  onSessionDeleted?: () => void;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({ apiBase, onUploadSuccess, activeSessionId, onSessionDeleted }) => {
  const [file, setFile] = useState<File | null>(null);
  const [strategy, setStrategy] = useState<string>('hybrid');
  const [maxFrames, setMaxFrames] = useState<number>(30);
  const [cloudEnhancers, setCloudEnhancers] = useState<boolean>(false);
  const [uploading, setUploading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Processing status states
  const [statusData, setStatusData] = useState<any>(null);
  const [polling, setPolling] = useState<boolean>(false);
  const [cancelling, setCancelling] = useState<boolean>(false);

  // Waiting-room entertainment
  const [lineIndex, setLineIndex] = useState<number>(0);
  const [musicPlaying, setMusicPlaying] = useState<boolean>(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Rotate the funny loading lines while processing
  useEffect(() => {
    if (statusData?.status !== 'processing') return;
    const id = setInterval(() => {
      setLineIndex((i) => (i + 1) % FUNNY_LINES.length);
    }, 4000);
    return () => clearInterval(id);
  }, [statusData?.status]);

  const toggleMusic = () => {
    if (!audioRef.current) {
      audioRef.current = new Audio('/sunflower.mp3');
      audioRef.current.loop = true;
      audioRef.current.volume = 0.35;
    }
    if (musicPlaying) {
      audioRef.current.pause();
      setMusicPlaying(false);
    } else {
      audioRef.current.play().catch(() => {});
      setMusicPlaying(true);
    }
  };

  // Stop the music if the component unmounts
  useEffect(() => {
    return () => {
      audioRef.current?.pause();
    };
  }, []);

  // Poll status when a session is active and processing
  useEffect(() => {
    let intervalId: any;

    const checkStatus = async (sessionId: string) => {
      try {
        const res = await fetch(`${apiBase}/processing-status/${sessionId}`);
        if (res.status === 200) {
          const data = await res.json();
          setStatusData(data);

          // Stop polling once processing is finished — no point hitting the API forever
          if (data.status === 'completed' || data.status === 'failed') {
            setPolling(false);
            if (intervalId) {
              clearInterval(intervalId);
              intervalId = null;
            }
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

  const handleUploadSample = async () => {
    setUploading(true);
    setError(null);
    setStatusData(null);

    try {
      const uploadUrl = new URL(`${apiBase}/upload-sample-video`);
      uploadUrl.searchParams.append('extraction_strategy', strategy);
      uploadUrl.searchParams.append('max_frames', maxFrames.toString());
      uploadUrl.searchParams.append('use_cloud_enhancers', cloudEnhancers.toString());

      const res = await fetch(uploadUrl.toString(), {
        method: 'POST',
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || 'Sample video analysis failed');
      }

      const data = await res.json();
      onUploadSuccess(data.session_id);
    } catch (err: any) {
      setError(err.message || 'An error occurred during sample video initialization.');
    } finally {
      setUploading(false);
    }
  };

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
      uploadUrl.searchParams.append('use_cloud_enhancers', cloudEnhancers.toString());

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

  const handleCancelSession = async () => {
    if (!activeSessionId) return;
    const confirmMsg = statusData?.status === 'processing'
      ? "Are you sure you want to cancel the active pipeline and delete all data for this session?"
      : "Are you sure you want to delete this session and all its results?";
    
    if (!window.confirm(confirmMsg)) return;
    
    setCancelling(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/sessions/${activeSessionId}`, {
        method: 'DELETE'
      });
      if (!res.ok) {
        throw new Error("Failed to delete session");
      }
      setStatusData(null);
      if (onSessionDeleted) {
        onSessionDeleted();
      }
    } catch (err: any) {
      setError(err.message || "Failed to cancel and delete session.");
    } finally {
      setCancelling(false);
    }
  };

  return (
    <div className="view-body">
      <div className="premium-card">
        <h3 className="card-title" style={{ fontSize: '1.1rem', marginBottom: '1.5rem' }}>
          Video Upload & Pipeline Controls
        </h3>

        {/* Info Banner */}
        <div style={{ padding: '0.75rem 1rem', background: 'rgba(99, 102, 241, 0.06)', border: '1px solid rgba(99, 102, 241, 0.15)', borderRadius: '8px', marginBottom: '1.25rem', fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
          <strong style={{ color: 'var(--primary)' }}>How it works:</strong> Upload a surveillance video (up to ~30MB recommended).
          The system extracts key frames, analyzes each one with vision AI, generates security alerts, and indexes everything for semantic search.
          Processing typically takes 1–3 minutes.
        </div>
        
        <input 
          type="file" 
          ref={fileInputRef} 
          onChange={handleFileChange} 
          style={{ display: 'none' }}
          accept=".mp4,.avi,.mov,.dav,.mkv,.wmv,.flv,.webm,.ts"
        />

        {/* Split: Sample Video + Upload Zone */}
        <div className="resp-grid-2" style={{ gap: '1rem', marginBottom: '1rem' }}>
          {/* Left: Sample Video */}
          <div style={{ padding: '1.25rem', background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '10px', textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
            <h4 style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.5rem' }}>Don't have a video?</h4>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.75rem', lineHeight: '1.5' }}>
              Run security analysis on our pre-loaded sample video instantly.
            </p>
            <button 
              disabled={uploading}
              onClick={handleUploadSample}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', padding: '0.5rem 1rem', background: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: '6px', color: '#10b981', fontSize: '0.78rem', fontWeight: 700, cursor: 'pointer' }}
            >
              Analyze Sample Video (Instant)
            </button>
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
              <option value="hybrid">Hybrid (Motion + Scene changes — Recommended)</option>
              <option value="motion_based">Motion-Based (Significant activity)</option>
              <option value="scene_change">Scene Change (Distinct screen cuts)</option>
              <option value="uniform">Uniform (Equal time intervals)</option>
            </select>
          </div>

          <div className="input-group">
            <label className="input-label">Maximum Frames to Extract</label>
            <div className="slider-container">
              <input 
                type="range" 
                className="custom-range" 
                min={5} 
                max={30} 
                step={5}
                value={maxFrames}
                onChange={(e) => setMaxFrames(parseInt(e.target.value))}
              />
              <span className="slider-val">{maxFrames}</span>
            </div>
          </div>
        </div>

        {/* Cloud Enhancers toggle — HF CLIP+BLIP cross-check for Gemini */}
        <div
          onClick={() => setCloudEnhancers(v => !v)}
          style={{
            display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer',
            padding: '0.85rem 1rem', borderRadius: '10px', marginBottom: '1.25rem',
            background: cloudEnhancers ? 'rgba(168, 85, 247, 0.08)' : 'rgba(255,255,255,0.02)',
            border: cloudEnhancers ? '1px solid rgba(168, 85, 247, 0.45)' : '1px solid var(--border-color)',
            transition: 'all 0.2s ease',
          }}
        >
          <Sparkles size={18} style={{ color: cloudEnhancers ? '#a855f7' : 'var(--text-muted)', flexShrink: 0 }} />
          <div style={{ flexGrow: 1 }}>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, color: cloudEnhancers ? '#c084fc' : 'var(--text-main)' }}>
              Enable Cloud Enhancers {cloudEnhancers ? '— ON' : ''}
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
              Cross-checks Gemini with CLIP threat scoring + BLIP captioning (HuggingFace cloud GPUs).
              Reduces hallucination and wrong analysis — recommended for production runs, slower per frame.
            </div>
          </div>
          {/* Switch */}
          <div style={{
            width: '38px', height: '21px', borderRadius: '11px', flexShrink: 0, position: 'relative',
            background: cloudEnhancers ? '#a855f7' : 'rgba(255,255,255,0.12)',
            transition: 'background 0.2s ease',
          }}>
            <div style={{
              position: 'absolute', top: '2.5px', left: cloudEnhancers ? '19px' : '3px',
              width: '16px', height: '16px', borderRadius: '50%', background: 'white',
              transition: 'left 0.2s ease',
            }} />
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
            'Start Security Analysis'
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
            <>
              <p style={{ fontSize: '0.8rem', color: '#818cf8', textAlign: 'center', marginBottom: '0.5rem', minHeight: '2.4rem', fontStyle: 'italic' }}>
                {FUNNY_LINES[lineIndex]}
              </p>
              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  Usually takes 1–3 minutes. Bored?
                </span>
                <button
                  onClick={toggleMusic}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', padding: '0.35rem 0.85rem', background: musicPlaying ? 'rgba(99, 102, 241, 0.15)' : 'rgba(99, 102, 241, 0.06)', border: '1px solid rgba(99, 102, 241, 0.3)', borderRadius: '6px', color: 'var(--primary)', fontSize: '0.72rem', fontWeight: 700, cursor: 'pointer' }}
                >
                  {musicPlaying ? (
                    <>
                      <Square size={12} /> Stop Music
                    </>
                  ) : (
                    <>
                      <Music size={12} /> Play a Song
                    </>
                  )}
                </button>
              </div>
            </>
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

          <div style={{ marginTop: '1.5rem', borderTop: '1px solid rgba(255, 255, 255, 0.08)', paddingTop: '1rem', display: 'flex', justifyContent: 'flex-end' }}>
            <button
              disabled={cancelling}
              onClick={handleCancelSession}
              style={{
                padding: '0.45rem 1rem',
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.25)',
                borderRadius: '6px',
                color: '#ef4444',
                fontSize: '0.78rem',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem'
              }}
            >
              {cancelling ? (
                <>
                  <RefreshCw className="animate-spin" size={12} /> Processing...
                </>
              ) : statusData.status === 'processing' ? (
                'Cancel & Delete Session'
              ) : (
                'Delete Session & Results'
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
