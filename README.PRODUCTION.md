# Production-Grade Security Video Analysis System

## Overview

This is a comprehensive production-grade solution for efficient video indexing, alert management, and data storage for security video analysis. The system is designed to handle large-scale video processing with real-time threat detection and alerting.

## Architecture

### Core Components

1. **Video Processing Service** - Enhanced GPT-4o + CLIP video analysis
2. **Alert Management System** - Real-time alert processing and notifications
3. **API Gateway** - RESTful API with authentication and monitoring
4. **Database Layer** - PostgreSQL for structured data, Redis for caching
5. **Storage Layer** - MinIO for video files, Pinecone for vector search
6. **Monitoring Stack** - Prometheus + Grafana for observability

### Technology Stack

- **Backend**: Python 3.11, FastAPI, AsyncIO
- **Database**: PostgreSQL 15, Redis 7
- **Storage**: MinIO (S3-compatible), Pinecone (vector search)
- **Monitoring**: Prometheus, Grafana
- **Containerization**: Docker, Docker Compose
- **Load Balancing**: Nginx
- **Security**: JWT authentication, rate limiting

## Quick Start

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM
- 100GB+ storage space
- OpenAI API key
- Pinecone API key

### Installation

1. **Clone and Setup**
```bash
git clone <repository-url>
cd drone-security-agent-1
```

2. **Configure Environment**
```bash
cp .env.production.example .env.production
# Edit .env.production with your configuration
```

3. **Start Services**
```bash
docker-compose -f docker-compose.production.yml up -d
```

4. **Initialize Database**
```bash
docker-compose -f docker-compose.production.yml exec postgres psql -U security_user -d security_analysis -f /docker-entrypoint-initdb.d/init.sql
```

5. **Verify Deployment**
```bash
curl http://localhost:8080/health
```

## Configuration

### Environment Variables

Key configuration options in `.env.production`:

- `DB_PASSWORD` - PostgreSQL database password
- `OPENAI_API_KEY` - OpenAI API key for GPT-4o vision analysis
- `PINECONE_API_KEY` - Pinecone API key for vector search
- `JWT_SECRET` - Secret for JWT authentication
- `MAX_CONCURRENT_VIDEOS` - Maximum concurrent video processing
- `STORAGE_RETENTION_DAYS` - Data retention period

### Performance Tuning

- **Database Pool**: Configure connection pool size based on load
- **Redis Cache**: Adjust memory limits for caching
- **Video Processing**: Set concurrent processing limits
- **Storage**: Configure retention policies and cleanup

## API Documentation

### Authentication

All API endpoints require JWT authentication:

```bash
curl -H "Authorization: Bearer <jwt_token>" http://localhost:8080/api/v1/videos
```

### Key Endpoints

#### Video Management
- `POST /api/v1/videos/upload` - Upload video for processing
- `GET /api/v1/videos` - List videos
- `GET /api/v1/videos/{video_id}` - Get video details
- `DELETE /api/v1/videos/{video_id}` - Delete video

#### Frame Analysis
- `GET /api/v1/videos/{video_id}/frames` - Get video frames
- `GET /api/v1/frames/{frame_id}` - Get frame analysis
- `POST /api/v1/frames/search` - Search frames

#### Alerts
- `GET /api/v1/alerts` - Get alerts
- `GET /api/v1/alerts/{alert_id}` - Get alert details
- `PUT /api/v1/alerts/{alert_id}/status` - Update alert status

#### Analytics
- `GET /api/v1/analytics/dashboard` - Dashboard data
- `GET /api/v1/analytics/storage` - Storage statistics

### WebSocket

Real-time updates via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## Data Storage

### Database Schema

**Video Index Table**
- Video metadata and processing status
- File information and timestamps
- Location and camera details

**Frame Analysis Table**
- Frame-level analysis results
- Threat levels and confidence scores
- People and object detection results

**Alerts Table**
- Alert details and status
- Severity levels and assignments
- Resolution tracking

### Vector Search

Frame embeddings stored in Pinecone for:
- Similar frame search
- Content-based retrieval
- Pattern matching

### Caching Strategy

Redis caching for:
- Recent frame analysis results
- Alert statistics
- User sessions
- Rate limiting

## Alert System

### Alert Types

- **Threat Detection** - Security threats identified by AI
- **System Health** - Processing errors and system issues
- **Storage** - Storage capacity warnings

### Notification Channels

- **Email** - SMTP-based email notifications
- **Webhook** - HTTP webhook callbacks
- **Slack** - Slack integration
- **SMS** - Twilio integration (optional)

### Alert Escalation

Automatic escalation based on:
- Alert severity
- Time without acknowledgment
- Business hours

## Monitoring

### Metrics

Key metrics tracked:
- Video processing throughput
- Frame analysis accuracy
- Alert generation rates
- System resource usage
- API response times

### Dashboards

Grafana dashboards for:
- System overview
- Performance metrics
- Alert statistics
- Storage usage

### Health Checks

- `/health` - System health status
- `/metrics` - Prometheus metrics
- Component health checks

## Performance Optimization

### Video Processing

- **Batch Processing** - Process frames in batches
- **Concurrent Processing** - Multiple videos simultaneously
- **Smart Sampling** - Intelligent frame extraction
- **GPU Acceleration** - CUDA support for CLIP/BLIP

### Database Optimization

- **Connection Pooling** - Efficient database connections
- **Indexing Strategy** - Optimized query performance
- **Partitioning** - Time-based data partitioning
- **Cleanup Jobs** - Automated data cleanup

### Caching

- **Multi-level Caching** - Application + database caching
- **Cache Invalidation** - Smart cache updates
- **Compression** - Compressed cached data
- **TTL Management** - Time-based expiration

## Security

### Authentication

- **JWT Tokens** - Secure API authentication
- **Token Rotation** - Automatic token refresh
- **Session Management** - Secure session handling

### Authorization

- **Role-based Access** - User permissions
- **API Rate Limiting** - Prevent abuse
- **IP Whitelisting** - Access control

### Data Protection

- **Encryption at Rest** - Database encryption
- **Encryption in Transit** - HTTPS/TLS
- **Access Logging** - Audit trails
- **Data Retention** - Automated cleanup

## Scaling

### Horizontal Scaling

- **Load Balancer** - Nginx distribution
- **Service Replicas** - Multiple instances
- **Database Sharding** - Data distribution
- **Cache Clustering** - Redis cluster

### Vertical Scaling

- **Resource Allocation** - CPU/Memory optimization
- **Storage Scaling** - Distributed storage
- **Network Optimization** - Bandwidth management

## Deployment

### Production Deployment

1. **Infrastructure Setup**
   - Configure servers and networking
   - Set up load balancer
   - Configure SSL certificates

2. **Database Setup**
   - Deploy PostgreSQL cluster
   - Configure Redis cluster
   - Set up backup strategies

3. **Application Deployment**
   - Build and deploy containers
   - Configure environment variables
   - Set up monitoring

4. **Monitoring Setup**
   - Deploy Prometheus
   - Configure Grafana dashboards
   - Set up alerting rules

### CI/CD Pipeline

- **Automated Testing** - Unit and integration tests
- **Container Builds** - Automated Docker builds
- **Rolling Updates** - Zero-downtime deployments
- **Rollback Strategy** - Quick recovery

## Troubleshooting

### Common Issues

**Video Processing Fails**
- Check OpenAI API quota
- Verify video format compatibility
- Check storage space

**High Memory Usage**
- Reduce concurrent processing
- Optimize batch sizes
- Check for memory leaks

**Slow API Response**
- Check database connections
- Verify cache hit rates
- Monitor system resources

### Debugging

- **Application Logs** - Detailed logging
- **Performance Metrics** - System monitoring
- **Error Tracking** - Exception handling
- **Health Checks** - Component status

## Maintenance

### Regular Tasks

- **Database Maintenance** - Vacuum and analyze
- **Log Rotation** - Manage log files
- **Backup Verification** - Test restore procedures
- **Security Updates** - Apply patches

### Monitoring

- **Performance Metrics** - Track system health
- **Storage Usage** - Monitor disk space
- **API Usage** - Track request patterns
- **Error Rates** - Monitor failures

## Support

### Documentation

- **API Documentation** - OpenAPI/Swagger specs
- **Architecture Docs** - System design
- **Troubleshooting Guide** - Common issues
- **Best Practices** - Optimization tips

### Contact

- **Technical Support** - System issues
- **Feature Requests** - Enhancement ideas
- **Bug Reports** - Issue tracking
- **Security Issues** - Vulnerability reports
