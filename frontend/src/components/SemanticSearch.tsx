import React, { useState } from 'react';
import { Search, Image, Play, RefreshCw, AlertTriangle } from 'lucide-react';

interface SemanticSearchProps {
  apiBase: string;
  activeSessionId: string | null;
  onSelectFrame: (frameId: string) => void;
}

export const SemanticSearch: React.FC<SemanticSearchProps> = ({ apiBase, activeSessionId, onSelectFrame }) => {
  const [query, setQuery] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<any[]>([]);
  const [searched, setSearched] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setSearched(true);

    try {
      const res = await fetch(`${apiBase}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          top_k: 6,
          session_id: activeSessionId,
        }),
      });

      if (!res.ok) {
        throw new Error('Search request failed. Make sure Pinecone index is active.');
      }

      const data = await res.json();
      setResults(data.results || []);
    } catch (err: any) {
      setError(err.message || 'Error occurred during semantic search.');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  if (!activeSessionId) {
    return (
      <div className="view-body">
        <div className="premium-card" style={{ textAlign: 'center', padding: '3rem' }}>
          <Search size={48} style={{ color: 'var(--text-muted)', marginBottom: '1rem' }} />
          <h3>No Active Video Session</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Please select or upload a video session to query visual intelligence.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="view-body">
      <div className="premium-card">
        <h3 className="card-title" style={{ fontSize: '1.1rem', marginBottom: '1.5rem' }}>
          🔍 Natural Language Visual Vector Search
        </h3>
        
        <form onSubmit={handleSearch} className="search-bar-row">
          <input
            type="text"
            className="search-input"
            placeholder="Search session frames (e.g., 'someone wearing dark clothes', 'vehicle loitering', 'person reaching for pocket')..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit" className="btn btn-primary" disabled={loading || !query.trim()}>
            {loading ? <RefreshCw className="animate-spin" size={16} /> : <Search size={16} />}
            Search
          </button>
        </form>
      </div>

      {error && (
        <div className="premium-card" style={{ borderLeft: '4px solid var(--color-critical)', color: 'var(--color-critical)', padding: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertTriangle size={16} /> {error}
          </div>
        </div>
      )}

      {loading ? (
        <div className="premium-card" style={{ textAlign: 'center', padding: '3rem' }}>
          <RefreshCw className="animate-spin" style={{ color: 'var(--primary)', marginBottom: '1rem' }} size={32} />
          <p style={{ color: 'var(--text-muted)' }}>Retrieving visual indices from vector database...</p>
        </div>
      ) : searched && results.length === 0 ? (
        <div className="premium-card" style={{ textAlign: 'center', padding: '3rem' }}>
          <Image size={36} style={{ color: 'var(--text-muted)', marginBottom: '1rem' }} />
          <h3>No Matches Found</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Try adjusting your query description (e.g. check keywords for colors, objects, or behaviors).
          </p>
        </div>
      ) : results.length > 0 ? (
        <div className="search-container">
          <div className="search-grid">
            {results.map((hit, idx) => {
              const frameId = hit.frame_id || '';
              // Ensure correct format for image retrieval
              const frameName = frameId.endsWith('.jpg') ? frameId : `${frameId}.jpg`;
              const scorePercent = (hit.similarity_score * 100).toFixed(1);
              
              return (
                <div key={idx} className="search-result-card" onClick={() => onSelectFrame(frameName)}>
                  <div className="result-img-box">
                    <img 
                      src={`${apiBase}/sessions/${activeSessionId}/frame-image/${frameName}`} 
                      alt={`Match ${frameId}`} 
                    />
                    <span className="result-score-badge">Match: {scorePercent}%</span>
                  </div>
                  
                  <div className="result-info-box">
                    <div className="result-frame-id">ID: {frameId}</div>
                    
                    <p className="result-description">
                      {hit.description || 'Description not indexed'}
                    </p>
                    
                    {hit.threat_assessment && hit.threat_assessment !== 'clear' && (
                      <span 
                        style={{ 
                          fontSize: '0.75rem', 
                          color: 'var(--color-critical)', 
                          fontWeight: 700,
                          textTransform: 'uppercase',
                          marginTop: '0.25rem' 
                        }}
                      >
                        ⚠️ Threat: {hit.threat_assessment}
                      </span>
                    )}

                    <button 
                      className="btn btn-secondary" 
                      style={{ marginTop: '0.5rem', width: '100%', padding: '0.35rem', fontSize: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.25rem' }}
                    >
                      <Play size={12} /> Analyze Details
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="premium-card" style={{ padding: '2rem' }}>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            Enter a description of what you're looking for in the video (e.g. clothing colors, actions, suspects, objects). The vector database will compare visual description embeddings and return matches sorted by relevance.
          </p>
        </div>
      )}
    </div>
  );
};
