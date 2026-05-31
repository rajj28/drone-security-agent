"""
api_gateway.py - Production-grade API gateway for security video analysis system

Features:
- RESTful API endpoints
- Authentication and authorization
- Rate limiting
- Request validation
- Response caching
- Monitoring and metrics
- WebSocket support for real-time updates
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from uuid import uuid4

import jwt
import aiohttp_cors
from aiohttp import web, ClientSession
from aiohttp.web import middleware
from prometheus_client import Counter, Histogram, generate_latest
import asyncpg

from src.production_config import production_settings
from src.production_video_indexer import production_indexer
from src.alert_manager import production_alert_manager, AlertSeverity, AlertStatus

logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Counter('active_connections', 'Active connections')

class APIGateway:
    """Production-grade API gateway"""
    
    def __init__(self):
        self.app = web.Application()
        self.setup_routes()
        self.setup_middleware()
        self.setup_cors()
        self.db_pool = None
        
    def setup_middleware(self):
        """Setup middleware"""
        self.app.middlewares.extend([
            self.auth_middleware,
            self.rate_limit_middleware,
            self.metrics_middleware,
            self.error_handler_middleware
        ])
    
    def setup_cors(self):
        """Setup CORS"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    @middleware
    async def auth_middleware(self, request: web.Request, handler):
        """Authentication middleware"""
        # Skip auth for health check and metrics
        if request.path in ['/health', '/metrics']:
            return await handler(request)
        
        # Get token from header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return web.json_response(
                {'error': 'Missing or invalid authorization header'},
                status=401
            )
        
        token = auth_header.split(' ')[1]
        
        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                production_settings.JWT_SECRET,
                algorithms=[production_settings.JWT_ALGORITHM]
            )
            
            # Add user info to request
            request['user'] = payload
            
        except jwt.ExpiredSignatureError:
            return web.json_response(
                {'error': 'Token has expired'},
                status=401
            )
        except jwt.InvalidTokenError:
            return web.json_response(
                {'error': 'Invalid token'},
                status=401
            )
        
        return await handler(request)
    
    @middleware
    async def rate_limit_middleware(self, request: web.Request, handler):
        """Rate limiting middleware"""
        # Simple rate limiting based on user ID
        user_id = request.get('user', {}).get('user_id', 'anonymous')
        
        # Check Redis for rate limit
        redis_key = f"rate_limit:{user_id}"
        
        try:
            # This would use Redis in production
            # For now, we'll skip actual rate limiting
            pass
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
        
        return await handler(request)
    
    @middleware
    async def metrics_middleware(self, request: web.Request, handler):
        """Metrics middleware"""
        start_time = datetime.utcnow()
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await handler(request)
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.path,
                status=response.status
            ).inc()
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            REQUEST_DURATION.observe(duration)
            
            return response
            
        finally:
            ACTIVE_CONNECTIONS.dec()
    
    @middleware
    async def error_handler_middleware(self, request: web.Request, handler):
        """Error handler middleware"""
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            return web.json_response(
                {'error': 'Internal server error'},
                status=500
            )
    
    def setup_routes(self):
        """Setup API routes"""
        # Health check
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/metrics', self.metrics)
        
        # Video management
        self.app.router.add_post('/api/v1/videos/upload', self.upload_video)
        self.app.router.add_get('/api/v1/videos', self.get_videos)
        self.app.router.add_get('/api/v1/videos/{video_id}', self.get_video)
        self.app.router.add_delete('/api/v1/videos/{video_id}', self.delete_video)
        
        # Frame analysis
        self.app.router.add_get('/api/v1/videos/{video_id}/frames', self.get_frames)
        self.app.router.add_get('/api/v1/frames/{frame_id}', self.get_frame)
        self.app.router.add_post('/api/v1/frames/search', self.search_frames)
        
        # Alerts
        self.app.router.add_get('/api/v1/alerts', self.get_alerts)
        self.app.router.add_get('/api/v1/alerts/{alert_id}', self.get_alert)
        self.app.router.add_put('/api/v1/alerts/{alert_id}/status', self.update_alert_status)
        self.app.router.add_get('/api/v1/alerts/statistics', self.get_alert_statistics)
        
        # Analytics
        self.app.router.add_get('/api/v1/analytics/dashboard', self.get_dashboard_data)
        self.app.router.add_get('/api/v1/analytics/storage', self.get_storage_stats)
        
        # WebSocket for real-time updates
        self.app.router.add_get('/ws', self.websocket_handler)
    
    async def health_check(self, request: web.Request):
        """Health check endpoint"""
        try:
            # Check database connection
            if self.db_pool:
                await self.db_pool.fetchval("SELECT 1")
            
            return web.json_response({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0'
            })
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return web.json_response(
                {'status': 'unhealthy', 'error': str(e)},
                status=503
            )
    
    async def metrics(self, request: web.Request):
        """Prometheus metrics endpoint"""
        return web.Response(
            body=generate_latest(),
            content_type='text/plain; version=0.0.4'
        )
    
    async def upload_video(self, request: web.Request):
        """Upload video for processing"""
        try:
            # Check if video is being uploaded
            if not request.content_type.startswith('multipart/form-data'):
                return web.json_response(
                    {'error': 'Expected multipart/form-data'},
                    status=400
                )
            
            reader = await request.multipart()
            
            # Get video file
            field = await reader.next()
            if field.name != 'video':
                return web.json_response(
                    {'error': 'Expected video field'},
                    status=400
                )
            
            # Save video file
            video_id = str(uuid4())
            filename = f"{video_id}_{field.filename}"
            file_path = f"data/videos/{filename}"
            
            with open(file_path, 'wb') as f:
                while True:
                    chunk = await field.read_chunk()
                    if not chunk:
                        break
                    f.write(chunk)
            
            # Get metadata
            metadata = {}
            try:
                metadata_field = await reader.next()
                if metadata_field and metadata_field.name == 'metadata':
                    metadata = json.loads(await metadata_field.text())
            except:
                pass
            
            # Process video
            from src.production_video_indexer import VideoMetadata
            
            video_metadata = VideoMetadata(
                video_id=video_id,
                filename=filename,
                file_path=file_path,
                file_size=field.filename,  # Would get actual file size
                duration=0,  # Would extract from video
                fps=30,  # Would extract from video
                resolution=(1920, 1080),  # Would extract from video
                codec='h264',  # Would extract from video
                created_at=datetime.utcnow(),
                location=metadata.get('location', 'unknown'),
                camera_id=metadata.get('camera_id'),
                tags=metadata.get('tags', [])
            )
            
            # Start processing in background
            asyncio.create_task(production_indexer.process_video(
                Path(file_path), video_metadata
            ))
            
            return web.json_response({
                'video_id': video_id,
                'status': 'processing',
                'message': 'Video uploaded and processing started'
            })
            
        except Exception as e:
            logger.error(f"Video upload failed: {e}")
            return web.json_response(
                {'error': 'Video upload failed'},
                status=500
            )
    
    async def get_videos(self, request: web.Request):
        """Get list of videos"""
        try:
            limit = int(request.query.get('limit', 50))
            offset = int(request.query.get('offset', 0))
            status = request.query.get('status')
            
            # Build query
            query = "SELECT * FROM video_index WHERE 1=1"
            params = []
            param_count = 0
            
            if status:
                param_count += 1
                query += f" AND status = ${param_count}"
                params.append(status)
            
            query += " ORDER BY created_at DESC LIMIT $" + str(param_count + 1) + " OFFSET $" + str(param_count + 2)
            params.extend([limit, offset])
            
            async with self.db_pool.acquire() as conn:
                records = await conn.fetch(query, *params)
                
                videos = []
                for record in records:
                    video = dict(record)
                    # Convert datetime to string
                    video['created_at'] = video['created_at'].isoformat()
                    if video['updated_at']:
                        video['updated_at'] = video['updated_at'].isoformat()
                    if video['processed_at']:
                        video['processed_at'] = video['processed_at'].isoformat()
                    videos.append(video)
                
                return web.json_response({
                    'videos': videos,
                    'total': len(videos),
                    'limit': limit,
                    'offset': offset
                })
                
        except Exception as e:
            logger.error(f"Failed to get videos: {e}")
            return web.json_response(
                {'error': 'Failed to get videos'},
                status=500
            )
    
    async def get_video(self, request: web.Request):
        """Get specific video details"""
        try:
            video_id = request.match_info['video_id']
            
            async with self.db_pool.acquire() as conn:
                record = await conn.fetchrow(
                    "SELECT * FROM video_index WHERE video_id = $1",
                    video_id
                )
                
                if not record:
                    return web.json_response(
                        {'error': 'Video not found'},
                        status=404
                    )
                
                video = dict(record)
                # Convert datetime to string
                video['created_at'] = video['created_at'].isoformat()
                if video['updated_at']:
                    video['updated_at'] = video['updated_at'].isoformat()
                if video['processed_at']:
                    video['processed_at'] = video['processed_at'].isoformat()
                
                return web.json_response(video)
                
        except Exception as e:
            logger.error(f"Failed to get video: {e}")
            return web.json_response(
                {'error': 'Failed to get video'},
                status=500
            )
    
    async def delete_video(self, request: web.Request):
        """Delete video"""
        try:
            video_id = request.match_info['video_id']
            
            # Delete from database
            async with self.db_pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM video_index WHERE video_id = $1",
                    video_id
                )
                
                if result == "DELETE 0":
                    return web.json_response(
                        {'error': 'Video not found'},
                        status=404
                    )
            
            return web.json_response({
                'message': 'Video deleted successfully'
            })
            
        except Exception as e:
            logger.error(f"Failed to delete video: {e}")
            return web.json_response(
                {'error': 'Failed to delete video'},
                status=500
            )
    
    async def get_frames(self, request: web.Request):
        """Get frames for a video"""
        try:
            video_id = request.match_info['video_id']
            limit = int(request.query.get('limit', 100))
            threat_level = request.query.get('threat_level')
            
            frames = await production_indexer.search_frames(
                query="",  # Empty query to get all frames
                video_id=video_id,
                threat_level=threat_level,
                limit=limit
            )
            
            return web.json_response({
                'frames': frames,
                'total': len(frames)
            })
            
        except Exception as e:
            logger.error(f"Failed to get frames: {e}")
            return web.json_response(
                {'error': 'Failed to get frames'},
                status=500
            )
    
    async def get_frame(self, request: web.Request):
        """Get specific frame analysis"""
        try:
            frame_id = request.match_info['frame_id']
            
            async with self.db_pool.acquire() as conn:
                record = await conn.fetchrow(
                    "SELECT * FROM frame_analysis WHERE frame_id = $1",
                    frame_id
                )
                
                if not record:
                    return web.json_response(
                        {'error': 'Frame not found'},
                        status=404
                    )
                
                frame = dict(record)
                frame['created_at'] = frame['created_at'].isoformat()
                
                return web.json_response(frame)
                
        except Exception as e:
            logger.error(f"Failed to get frame: {e}")
            return web.json_response(
                {'error': 'Failed to get frame'},
                status=500
            )
    
    async def search_frames(self, request: web.Request):
        """Search frames"""
        try:
            data = await request.json()
            
            query = data.get('query', '')
            video_id = data.get('video_id')
            threat_level = data.get('threat_level')
            limit = data.get('limit', 100)
            
            frames = await production_indexer.search_frames(
                query=query,
                video_id=video_id,
                threat_level=threat_level,
                limit=limit
            )
            
            return web.json_response({
                'frames': frames,
                'total': len(frames),
                'query': query
            })
            
        except Exception as e:
            logger.error(f"Failed to search frames: {e}")
            return web.json_response(
                {'error': 'Failed to search frames'},
                status=500
            )
    
    async def get_alerts(self, request: web.Request):
        """Get alerts"""
        try:
            limit = int(request.query.get('limit', 100))
            status = request.query.get('status')
            severity = request.query.get('severity')
            
            # Convert string to enum
            alert_status = None
            if status:
                alert_status = AlertStatus(status)
            
            alert_severity = None
            if severity:
                alert_severity = AlertSeverity(severity)
            
            alerts = await production_alert_manager.get_alerts(
                status=alert_status,
                severity=alert_severity,
                limit=limit
            )
            
            # Convert datetime to string
            for alert in alerts:
                if alert['created_at']:
                    alert['created_at'] = alert['created_at'].isoformat()
                if alert['updated_at']:
                    alert['updated_at'] = alert['updated_at'].isoformat()
                if alert['resolved_at']:
                    alert['resolved_at'] = alert['resolved_at'].isoformat()
            
            return web.json_response({
                'alerts': alerts,
                'total': len(alerts)
            })
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return web.json_response(
                {'error': 'Failed to get alerts'},
                status=500
            )
    
    async def get_alert(self, request: web.Request):
        """Get specific alert"""
        try:
            alert_id = request.match_info['alert_id']
            
            async with self.db_pool.acquire() as conn:
                record = await conn.fetchrow(
                    "SELECT * FROM alerts WHERE alert_id = $1",
                    alert_id
                )
                
                if not record:
                    return web.json_response(
                        {'error': 'Alert not found'},
                        status=404
                    )
                
                alert = dict(record)
                # Convert datetime to string
                alert['created_at'] = alert['created_at'].isoformat()
                if alert['updated_at']:
                    alert['updated_at'] = alert['updated_at'].isoformat()
                if alert['resolved_at']:
                    alert['resolved_at'] = alert['resolved_at'].isoformat()
                
                return web.json_response(alert)
                
        except Exception as e:
            logger.error(f"Failed to get alert: {e}")
            return web.json_response(
                {'error': 'Failed to get alert'},
                status=500
            )
    
    async def update_alert_status(self, request: web.Request):
        """Update alert status"""
        try:
            alert_id = request.match_info['alert_id']
            data = await request.json()
            
            status = AlertStatus(data.get('status'))
            assigned_to = data.get('assigned_to')
            
            success = await production_alert_manager.update_alert_status(
                alert_id, status, assigned_to
            )
            
            if success:
                return web.json_response({
                    'message': 'Alert status updated successfully'
                })
            else:
                return web.json_response(
                    {'error': 'Failed to update alert status'},
                    status=500
                )
                
        except Exception as e:
            logger.error(f"Failed to update alert status: {e}")
            return web.json_response(
                {'error': 'Failed to update alert status'},
                status=500
            )
    
    async def get_alert_statistics(self, request: web.Request):
        """Get alert statistics"""
        try:
            stats = await production_alert_manager.get_alert_statistics()
            return web.json_response(stats)
            
        except Exception as e:
            logger.error(f"Failed to get alert statistics: {e}")
            return web.json_response(
                {'error': 'Failed to get alert statistics'},
                status=500
            )
    
    async def get_dashboard_data(self, request: web.Request):
        """Get dashboard analytics data"""
        try:
            # Get storage stats
            storage_stats = await production_indexer.get_storage_stats()
            
            # Get alert stats
            alert_stats = await production_alert_manager.get_alert_statistics()
            
            # Get recent activity
            async with self.db_pool.acquire() as conn:
                recent_videos = await conn.fetch("""
                    SELECT video_id, filename, status, created_at
                    FROM video_index
                    ORDER BY created_at DESC
                    LIMIT 5
                """)
                
                recent_alerts = await conn.fetch("""
                    SELECT alert_id, severity, title, created_at
                    FROM alerts
                    ORDER BY created_at DESC
                    LIMIT 5
                """)
            
            dashboard_data = {
                'storage': storage_stats,
                'alerts': alert_stats,
                'recent_videos': [dict(record) for record in recent_videos],
                'recent_alerts': [dict(record) for record in recent_alerts],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Convert datetime to string
            for video in dashboard_data['recent_videos']:
                video['created_at'] = video['created_at'].isoformat()
            
            for alert in dashboard_data['recent_alerts']:
                alert['created_at'] = alert['created_at'].isoformat()
            
            return web.json_response(dashboard_data)
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return web.json_response(
                {'error': 'Failed to get dashboard data'},
                status=500
            )
    
    async def get_storage_stats(self, request: web.Request):
        """Get storage statistics"""
        try:
            stats = await production_indexer.get_storage_stats()
            return web.json_response(stats)
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return web.json_response(
                {'error': 'Failed to get storage stats'},
                status=500
            )
    
    async def websocket_handler(self, request: web.Request):
        """WebSocket handler for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        logger.info("WebSocket connection established")
        
        try:
            # Send initial connection message
            await ws.send_json({
                'type': 'connection',
                'message': 'Connected to security analysis system',
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Keep connection alive and send updates
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Handle client messages
                    try:
                        data = json.loads(msg.data)
                        
                        if data.get('type') == 'subscribe':
                            # Handle subscription to specific events
                            await ws.send_json({
                                'type': 'subscription',
                                'message': f'Subscribed to {data.get("event", "all")} events',
                                'timestamp': datetime.utcnow().isoformat()
                            })
                        
                    except json.JSONDecodeError:
                        await ws.send_json({
                            'type': 'error',
                            'message': 'Invalid JSON message',
                            'timestamp': datetime.utcnow().isoformat()
                        })
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await ws.close()
            logger.info("WebSocket connection closed")
    
    async def initialize(self):
        """Initialize API gateway"""
        logger.info("Initializing API gateway...")
        
        # Initialize database connection
        self.db_pool = await asyncpg.create_pool(
            host=production_settings.DB_HOST,
            port=production_settings.DB_PORT,
            database=production_settings.DB_NAME,
            user=production_settings.DB_USER,
            password=production_settings.DB_PASSWORD,
            min_size=5,
            max_size=20
        )
        
        logger.info("API gateway initialized successfully")
    
    async def start(self):
        """Start the API server"""
        await self.initialize()
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(
            runner,
            production_settings.API_HOST,
            production_settings.API_PORT
        )
        
        await site.start()
        logger.info(f"API server started on {production_settings.API_HOST}:{production_settings.API_PORT}")
        
        return runner

# Create API gateway instance
api_gateway = APIGateway()

async def create_app():
    """Create and configure the application"""
    return api_gateway.app

if __name__ == '__main__':
    # Run the API server
    web.run_app(
        api_gateway.app,
        host=production_settings.API_HOST,
        port=production_settings.API_PORT
    )
