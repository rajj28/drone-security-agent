import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, RefreshCw, Compass } from 'lucide-react';

interface SecurityAgentProps {
  apiBase: string;
  activeSessionId: string | null;
  onSelectFrame: (frameId: string) => void;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  confidence?: number;
}

export const SecurityAgent: React.FC<SecurityAgentProps> = ({ apiBase, activeSessionId, onSelectFrame }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: "Hello! I am your AI Security Analyst Agent. I have scanned the extracted video frames and compiled the telemetry database. You can ask me natural language questions about suspicious activities, clothing colors, timeline details, or zone entries."
    }
  ]);
  const [input, setInput] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const suggestedQuestions = [
    "Was anything detected after hours?",
    "Give me a complete timeline of incidents today",
    "How many people were detected during monitoring?",
    "What was the most suspicious event today?"
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async (textToSend: string) => {
    if (!textToSend.trim() || loading) return;

    const userMessage: Message = { role: 'user', content: textToSend };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(`${apiBase}/qa`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: textToSend,
          session_id: activeSessionId,
        }),
      });

      if (!res.ok) {
        throw new Error('Agent failed to respond.');
      }

      const data = await res.json();
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources || [],
        confidence: data.confidence || 0,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err: any) {
      const errorMessage: Message = {
        role: 'assistant',
        content: "Sorry, I encountered an error connecting to my neural network. Please verify that the API server is online and running."
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSend(input);
  };

  if (!activeSessionId) {
    return (
      <div className="view-body">
        <div className="premium-card" style={{ textAlign: 'center', padding: '3rem' }}>
          <Bot size={48} style={{ color: 'var(--text-muted)', marginBottom: '1rem' }} />
          <h3>No Active Video Session</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Please select or upload a video session to activate the AI Security Agent.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="view-body" style={{ height: 'calc(100vh - 180px)', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div className="chat-window">
        {/* Chat Bubbles */}
        <div className="chat-history">
          {messages.map((msg, index) => (
            <div key={index} className={`chat-bubble ${msg.role === 'user' ? 'bubble-user' : 'bubble-agent'}`}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem', fontSize: '0.75rem', fontWeight: 700, opacity: 0.8 }}>
                {msg.role === 'user' ? <User size={12} /> : <Bot size={12} />}
                <span>{msg.role === 'user' ? 'Security Operator' : 'AI Security Guard'}</span>
              </div>
              
              <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>

              {msg.sources && msg.sources.length > 0 && (
                <div style={{ marginTop: '0.75rem', paddingTop: '0.5rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                  <span style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                    Reference Frames:
                  </span>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                    {msg.sources.map(src => {
                      const frameName = src.endsWith('.jpg') ? src : `${src}.jpg`;
                      return (
                        <button
                          key={src}
                          onClick={() => onSelectFrame(frameName)}
                          className="btn"
                          style={{
                            padding: '0.15rem 0.45rem',
                            fontSize: '0.7rem',
                            borderRadius: '4px',
                            fontFamily: 'var(--font-mono)',
                            border: '1px solid rgba(59, 130, 246, 0.3)',
                            background: 'rgba(59, 130, 246, 0.1)',
                            color: 'var(--primary)',
                            cursor: 'pointer'
                          }}
                        >
                          {src.replace('.jpg', '')}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="chat-bubble bubble-agent">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem', fontSize: '0.75rem', fontWeight: 700, opacity: 0.8 }}>
                <Bot size={12} />
                <span>AI Security Guard</span>
              </div>
              <div className="typing-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Suggested Questions */}
        <div style={{ padding: '0.75rem 1.5rem', background: 'rgba(255, 255, 255, 0.01)', borderTop: '1px solid var(--border-color)', display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <Compass size={12} /> Quick Queries:
          </span>
          {suggestedQuestions.map((q, idx) => (
            <button
              key={idx}
              className="btn btn-secondary"
              onClick={() => handleSend(q)}
              disabled={loading}
              style={{ padding: '0.25rem 0.65rem', fontSize: '0.75rem', borderRadius: '15px' }}
            >
              {q}
            </button>
          ))}
        </div>

        {/* Chat Input Bar */}
        <form onSubmit={handleSubmit} className="chat-input-bar">
          <input
            type="text"
            className="chat-text-input"
            placeholder="Ask AI about video events, clothing features, suspicious zones..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
          />
          <button type="submit" className="btn btn-primary" disabled={loading || !input.trim()}>
            {loading ? <RefreshCw className="animate-spin" size={16} /> : <Send size={16} />}
            Send
          </button>
        </form>
      </div>
    </div>
  );
};
