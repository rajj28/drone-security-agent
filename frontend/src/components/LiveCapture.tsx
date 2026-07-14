import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Radio, Square, RefreshCw, AlertTriangle, Wifi, Smartphone } from 'lucide-react';

interface LiveCaptureProps {
  apiBase: string;
  onAnalysisStarted: (sessionId: string) => void;
}

type CaptureMode = 'camera' | 'stream';

export const LiveCapture: React.FC<LiveCaptureProps> = ({ apiBase, onAnalysisStarted }) => {
  const [mode, setMode] = useState<CaptureMode>('camera');
  const [interval_s, setIntervalS] = useState<number>(2);
  const [maxFrames, setMaxFrames] = useState<number>(30);
  const [streamUrl, setStreamUrl] = useState<string>('');

  const [capturing, setCapturing] = useState<boolean>(false);
  const [stopping, setStopping] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [frameCount, setFrameCount] = useState<number>(0);
  const [elapsed, setElapsed] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const captureTimerRef = useRef<any>(null);
  const pollTimerRef = useRef<any>(null);
  const elapsedTimerRef = useRef<any>(null);
  const sessionIdRef = useRef<string | null>(null);
  const frameCountRef = useRef<number>(0);

  const cleanupTimers = () => {
    if (captureTimerRef.current) { clearInterval(captureTimerRef.current); captureTimerRef.current = null; }
    if (pollTimerRef.current) { clearInterval(pollTimerRef.current); pollTimerRef.current = null; }
    if (elapsedTimerRef.current) { clearInterval(elapsedTimerRef.current); elapsedTimerRef.current = null; }
  };

  const stopCameraTracks = () => {
    mediaStreamRef.current?.getTracks().forEach(t => t.stop());
    mediaStreamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  };

  // Full cleanup on unmount
  useEffect(() => {
    return () => {
      cleanupTimers();
      stopCameraTracks();
    };
  }, []);

  const captureAndSendFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const sid = sessionIdRef.current;
    if (!video || !canvas || !sid || video.readyState < 2) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob: Blob | null = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
    if (!blob) return;

    const form = new FormData();
    form.append('file', blob, `frame_${Date.now()}.jpg`);
    try {
      const res = await fetch(`${apiBase}/live/${sid}/frame`, { method: 'POST', body: form });
      if (res.ok) {
        const data = await res.json();
        if (typeof data.frame_count === 'number') {
          frameCountRef.current = data.frame_count;
          setFrameCount(data.frame_count);
        }
        if (data.full) {
          // Reached max frames — finalize automatically
          handleStop();
        }
      }
    } catch (err) {
      console.error('Frame push failed:', err);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase]);

  const handleStartCamera = async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access is not available. Note: mobile browsers require HTTPS (or localhost) for camera access.');
      return;
    }
    try {
      // Prefer the rear camera on phones — that's the "drone eye" view
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err: any) {
      setError(`Camera access denied or unavailable: ${err.message || err}`);
      return;
    }

    try {
      const res = await fetch(`${apiBase}/live/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: 'browser', max_frames: maxFrames, capture_interval: interval_s }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || 'Failed to start live session');
      const data = await res.json();
      sessionIdRef.current = data.session_id;
      setSessionId(data.session_id);
      setFrameCount(0);
      frameCountRef.current = 0;
      setElapsed(0);
      setCapturing(true);

      captureTimerRef.current = setInterval(captureAndSendFrame, interval_s * 1000);
      elapsedTimerRef.current = setInterval(() => setElapsed(e => e + 1), 1000);
    } catch (err: any) {
      stopCameraTracks();
      setError(err.message || 'Failed to start live session');
    }
  };

  const handleStartStream = async () => {
    setError(null);
    if (!streamUrl.trim()) {
      setError('Enter your drone / IP camera stream URL (e.g. rtsp://192.168.1.10:554/live)');
      return;
    }
    try {
      const res = await fetch(`${apiBase}/live/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source: 'stream',
          stream_url: streamUrl.trim(),
          max_frames: maxFrames,
          capture_interval: interval_s,
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || 'Failed to start stream capture');
      const data = await res.json();
      sessionIdRef.current = data.session_id;
      setSessionId(data.session_id);
      setFrameCount(0);
      frameCountRef.current = 0;
      setElapsed(0);
      setCapturing(true);

      elapsedTimerRef.current = setInterval(() => setElapsed(e => e + 1), 1000);
      pollTimerRef.current = setInterval(async () => {
        const sid = sessionIdRef.current;
        if (!sid) return;
        try {
          const r = await fetch(`${apiBase}/live/${sid}/status`);
          if (r.ok) {
            const s = await r.json();
            if (typeof s.frame_count === 'number') {
              frameCountRef.current = s.frame_count;
              setFrameCount(s.frame_count);
            }
            if (s.error) setError(`Stream issue: ${s.error}`);
            if (s.full) handleStop();
          }
        } catch { /* transient poll errors are fine */ }
      }, 2000);
    } catch (err: any) {
      setError(err.message || 'Failed to start stream capture');
    }
  };

  const handleStop = async () => {
    const sid = sessionIdRef.current;
    if (!sid || stopping) return;
    setStopping(true);
    cleanupTimers();
    stopCameraTracks();

    try {
      const res = await fetch(`${apiBase}/live/${sid}/stop`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to stop capture');
      if (data.frame_count === 0) {
        setError(data.error || 'No frames were captured — nothing to analyze.');
      } else {
        // Hand off to the standard pipeline status view
        onAnalysisStarted(sid);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to stop capture');
    } finally {
      setCapturing(false);
      setStopping(false);
      setSessionId(null);
      sessionIdRef.current = null;
    }
  };

  const formatElapsed = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;

  return (
    <div className="view-body">
      <div className="premium-card">
        <h3 className="card-title">
          <Radio size={18} style={{ color: 'var(--color-critical)' }} /> Live Capture — Drone & Mobile Camera
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
          Record live from this device's camera (open this page on your phone) or pull frames directly
          from a drone / IP camera stream. Captured frames run through the same AI security pipeline
          as uploaded videos.
        </p>

        {/* Mode selector */}
        {!capturing && (
          <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.25rem' }}>
            <button
              onClick={() => setMode('camera')}
              style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
                padding: '0.75rem', borderRadius: '8px', cursor: 'pointer', fontWeight: 700, fontSize: '0.85rem',
                background: mode === 'camera' ? 'rgba(99, 102, 241, 0.15)' : 'rgba(255,255,255,0.03)',
                border: mode === 'camera' ? '1px solid var(--primary)' : '1px solid var(--border-color)',
                color: mode === 'camera' ? 'var(--primary)' : 'var(--text-muted)',
              }}
            >
              <Smartphone size={16} /> Phone / Webcam Camera
            </button>
            <button
              onClick={() => setMode('stream')}
              style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
                padding: '0.75rem', borderRadius: '8px', cursor: 'pointer', fontWeight: 700, fontSize: '0.85rem',
                background: mode === 'stream' ? 'rgba(99, 102, 241, 0.15)' : 'rgba(255,255,255,0.03)',
                border: mode === 'stream' ? '1px solid var(--primary)' : '1px solid var(--border-color)',
                color: mode === 'stream' ? 'var(--primary)' : 'var(--text-muted)',
              }}
            >
              <Wifi size={16} /> Drone / Stream URL
            </button>
          </div>
        )}

        {/* Stream URL input */}
        {mode === 'stream' && !capturing && (
          <div className="input-group" style={{ marginBottom: '1rem' }}>
            <label className="input-label">Drone / IP Camera Stream URL</label>
            <input
              type="text"
              value={streamUrl}
              onChange={(e) => setStreamUrl(e.target.value)}
              placeholder="rtsp://192.168.1.10:554/live  or  http://camera/mjpeg"
              style={{
                width: '100%', padding: '0.6rem 0.75rem', borderRadius: '6px',
                background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border-color)',
                color: 'var(--text-main)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem',
              }}
            />
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
              RTSP / RTMP / HTTP-MJPEG supported. The stream must be reachable from the server —
              for a drone on your local Wi-Fi, run the backend locally.
            </p>
          </div>
        )}

        {/* Capture settings */}
        {!capturing && (
          <div className="upload-param-grid">
            <div className="input-group">
              <label className="input-label">Capture Interval (seconds between frames)</label>
              <select className="custom-select" value={interval_s} onChange={(e) => setIntervalS(Number(e.target.value))}>
                <option value={1}>Every 1 second (fast action)</option>
                <option value={2}>Every 2 seconds (recommended)</option>
                <option value={3}>Every 3 seconds</option>
                <option value={5}>Every 5 seconds (long patrol)</option>
              </select>
            </div>
            <div className="input-group">
              <label className="input-label">Max Frames (auto-stops when reached)</label>
              <div className="slider-container">
                <input
                  type="range" min={10} max={100} step={5}
                  className="custom-range"
                  value={maxFrames}
                  onChange={(e) => setMaxFrames(Number(e.target.value))}
                />
                <span className="slider-val">{maxFrames}</span>
              </div>
            </div>
          </div>
        )}

        {/* Camera preview (kept mounted so the ref exists when capture starts) */}
        <div style={{ display: mode === 'camera' ? 'block' : 'none', marginBottom: '1rem' }}>
          <video
            ref={videoRef}
            muted
            playsInline
            style={{
              width: '100%', maxHeight: '360px', borderRadius: '8px', background: '#000',
              border: capturing ? '2px solid var(--color-critical)' : '1px solid var(--border-color)',
              objectFit: 'contain',
            }}
          />
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>

        {/* Live status while capturing */}
        {capturing && (
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '0.75rem 1rem', borderRadius: '8px', marginBottom: '1rem',
            background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.3)',
          }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, fontSize: '0.85rem', color: 'var(--color-critical)' }}>
              <span className="pulse-dot"></span> RECORDING LIVE
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
              {formatElapsed(elapsed)} &nbsp;|&nbsp; {frameCount} / {maxFrames} frames
            </span>
          </div>
        )}

        {error && (
          <div style={{
            display: 'flex', alignItems: 'flex-start', gap: '0.5rem', padding: '0.75rem 1rem',
            borderRadius: '8px', marginBottom: '1rem', fontSize: '0.8rem',
            background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.3)', color: 'var(--color-critical)',
          }}>
            <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: '0.1rem' }} />
            <span>{error}</span>
          </div>
        )}

        {/* Start / Stop */}
        {!capturing ? (
          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            onClick={mode === 'camera' ? handleStartCamera : handleStartStream}
          >
            <Camera size={16} /> {mode === 'camera' ? 'Start Live Camera Capture' : 'Connect & Capture Stream'}
          </button>
        ) : (
          <button
            className="btn btn-primary"
            style={{ width: '100%', background: 'var(--color-critical)', borderColor: 'var(--color-critical)' }}
            onClick={handleStop}
            disabled={stopping}
          >
            {stopping ? (
              <>
                <RefreshCw className="animate-spin" size={16} /> Finalizing & Starting Analysis...
              </>
            ) : (
              <>
                <Square size={16} /> Stop & Run Security Analysis
              </>
            )}
          </button>
        )}

        {sessionId && capturing && (
          <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.75rem', fontFamily: 'var(--font-mono)' }}>
            Live Session: {sessionId}
          </p>
        )}
      </div>
    </div>
  );
};
