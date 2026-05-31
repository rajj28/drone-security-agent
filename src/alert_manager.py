"""
alert_manager.py - Production-grade alert management system

Features:
- Real-time alert processing
- Multi-channel notifications (email, SMS, webhook)
- Alert escalation and deduplication
- Alert analytics and reporting
- Integration with external monitoring systems
"""

import asyncio
import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

import aiohttp
import asyncpg
from jinja2 import Template

from src.production_config import production_settings

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"

@dataclass
class Alert:
    alert_id: str
    video_id: str
    frame_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    description: str
    confidence: float
    metadata: Dict[str, Any]
    status: AlertStatus = AlertStatus.OPEN
    created_at: datetime = None
    assigned_to: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class NotificationChannel:
    """Base class for notification channels"""
    
    async def send_notification(self, alert: Alert, message: str) -> bool:
        raise NotImplementedError

class EmailNotification(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self):
        self.smtp_host = production_settings.SMTP_HOST
        self.smtp_port = production_settings.SMTP_PORT
        self.smtp_user = production_settings.SMTP_USER
        self.smtp_password = production_settings.SMTP_PASSWORD
    
    async def send_notification(self, alert: Alert, message: str) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = alert.metadata.get('recipient_email', 'security@company.com')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create HTML email template
            html_template = """
            <html>
            <body>
                <h2>Security Alert: {{ alert.title }}</h2>
                <table border="1" style="border-collapse: collapse;">
                    <tr><td><strong>Alert ID:</strong></td><td>{{ alert.alert_id }}</td></tr>
                    <tr><td><strong>Severity:</strong></td><td>{{ alert.severity.value.upper() }}</td></tr>
                    <tr><td><strong>Video ID:</strong></td><td>{{ alert.video_id }}</td></tr>
                    <tr><td><strong>Frame ID:</strong></td><td>{{ alert.frame_id }}</td></tr>
                    <tr><td><strong>Confidence:</strong></td><td>{{ "%.2f"|format(alert.confidence) }}</td></tr>
                    <tr><td><strong>Created:</strong></td><td>{{ alert.created_at.strftime('%Y-%m-%d %H:%M:%S') }}</td></tr>
                </table>
                <h3>Description:</h3>
                <p>{{ alert.description }}</p>
                <h3>Message:</h3>
                <p>{{ message }}</p>
                <hr>
                <p><small>This is an automated security alert. Please respond accordingly.</small></p>
            </body>
            </html>
            """
            
            template = Template(html_template)
            html_body = template.render(alert=alert, message=message)
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

class WebhookNotification(NotificationChannel):
    """Webhook notification channel"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send_notification(self, alert: Alert, message: str) -> bool:
        try:
            payload = {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "confidence": alert.confidence,
                "video_id": alert.video_id,
                "frame_id": alert.frame_id,
                "created_at": alert.created_at.isoformat(),
                "message": message,
                "metadata": alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent for alert {alert.alert_id}")
                        return True
                    else:
                        logger.error(f"Webhook notification failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False

class SlackNotification(NotificationChannel):
    """Slack notification channel"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send_notification(self, alert: Alert, message: str) -> bool:
        try:
            color_map = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "#ff0000"
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "good"),
                        "title": f"Security Alert: {alert.title}",
                        "fields": [
                            {"title": "Alert ID", "value": alert.alert_id, "short": True},
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Video ID", "value": alert.video_id, "short": True},
                            {"title": "Confidence", "value": f"{alert.confidence:.2f}", "short": True},
                            {"title": "Description", "value": alert.description, "short": False},
                            {"title": "Message", "value": message, "short": False}
                        ],
                        "footer": "Security Analysis System",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert {alert.alert_id}")
                        return True
                    else:
                        logger.error(f"Slack notification failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

class AlertDeduplicator:
    """Alert deduplication system"""
    
    def __init__(self):
        self.recent_alerts = {}
        self.deduplication_window = timedelta(minutes=5)
    
    def is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate"""
        key = f"{alert.video_id}_{alert.alert_type}_{alert.severity.value}"
        
        if key in self.recent_alerts:
            last_alert_time = self.recent_alerts[key]
            if datetime.utcnow() - last_alert_time < self.deduplication_window:
                return True
        
        self.recent_alerts[key] = datetime.utcnow()
        return False
    
    def cleanup_old_alerts(self):
        """Clean up old alert records"""
        cutoff_time = datetime.utcnow() - self.deduplication_window
        self.recent_alerts = {
            k: v for k, v in self.recent_alerts.items() 
            if v > cutoff_time
        }

class AlertEscalation:
    """Alert escalation system"""
    
    def __init__(self):
        self.escalation_rules = {
            AlertSeverity.LOW: timedelta(hours=1),
            AlertSeverity.MEDIUM: timedelta(minutes=30),
            AlertSeverity.HIGH: timedelta(minutes=15),
            AlertSeverity.CRITICAL: timedelta(minutes=5)
        }
    
    def should_escalate(self, alert: Alert) -> bool:
        """Check if alert should be escalated"""
        if alert.status != AlertStatus.OPEN:
            return False
        
        escalation_time = self.escalation_rules.get(alert.severity, timedelta(hours=1))
        return datetime.utcnow() - alert.created_at > escalation_time
    
    def get_escalation_message(self, alert: Alert) -> str:
        """Get escalation message"""
        return f"ALERT ESCALATION: {alert.title} has not been acknowledged within the escalation timeframe. Immediate attention required."

class ProductionAlertManager:
    """Production-grade alert management system"""
    
    def __init__(self):
        self.notification_channels = []
        self.deduplicator = AlertDeduplicator()
        self.escalation = AlertEscalation()
        self.db_pool = None
        self.alert_queue = asyncio.Queue()
        self.processing = False
        
    async def initialize(self):
        """Initialize alert manager"""
        logger.info("Initializing production alert manager...")
        
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
        
        # Initialize notification channels
        await self._setup_notification_channels()
        
        # Start alert processing
        asyncio.create_task(self._process_alerts())
        
        # Start escalation checker
        asyncio.create_task(self._check_escalations())
        
        logger.info("Production alert manager initialized successfully")
    
    async def _setup_notification_channels(self):
        """Setup notification channels"""
        # Email notification
        self.notification_channels.append(EmailNotification())
        
        # Webhook notifications (if configured)
        webhook_url = os.getenv('WEBHOOK_URL')
        if webhook_url:
            self.notification_channels.append(WebhookNotification(webhook_url))
        
        # Slack notifications (if configured)
        slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if slack_webhook_url:
            self.notification_channels.append(SlackNotification(slack_webhook_url))
    
    async def create_alert(self, alert_data: Dict[str, Any]) -> str:
        """Create new alert"""
        try:
            alert = Alert(
                alert_id=alert_data['alert_id'],
                video_id=alert_data['video_id'],
                frame_id=alert_data['frame_id'],
                alert_type=alert_data['alert_type'],
                severity=AlertSeverity(alert_data['severity']),
                title=alert_data['title'],
                description=alert_data['description'],
                confidence=alert_data['confidence'],
                metadata=alert_data.get('metadata', {})
            )
            
            # Check for duplicates
            if self.deduplicator.is_duplicate(alert):
                logger.info(f"Duplicate alert detected and skipped: {alert.alert_id}")
                return alert.alert_id
            
            # Store in database
            await self._store_alert(alert)
            
            # Add to processing queue
            await self.alert_queue.put(alert)
            
            logger.info(f"Alert created: {alert.alert_id}")
            return alert.alert_id
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            raise
    
    async def _store_alert(self, alert: Alert):
        """Store alert in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO alerts 
                (alert_id, video_id, frame_id, alert_type, severity, title, 
                 description, confidence, metadata, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (alert_id) DO NOTHING
            """, 
                alert.alert_id,
                alert.video_id,
                alert.frame_id,
                alert.alert_type,
                alert.severity.value,
                alert.title,
                alert.description,
                alert.confidence,
                json.dumps(alert.metadata),
                alert.status.value,
                alert.created_at
            )
    
    async def _process_alerts(self):
        """Process alerts from queue"""
        self.processing = True
        
        while self.processing:
            try:
                # Wait for alert
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                
                # Send notifications
                await self._send_notifications(alert)
                
                # Log alert
                logger.info(f"Processed alert: {alert.alert_id} (Severity: {alert.severity.value})")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications through all channels"""
        message = f"Security alert detected in video {alert.video_id}. Please review and take appropriate action."
        
        # Send notifications concurrently
        tasks = [
            channel.send_notification(alert, message)
            for channel in self.notification_channels
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Notification channel {i} failed: {result}")
            else:
                logger.info(f"Notification channel {i} succeeded: {result}")
    
    async def _check_escalations(self):
        """Check for alert escalations"""
        while self.processing:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Get open alerts that need escalation
                async with self.db_pool.acquire() as conn:
                    records = await conn.fetch("""
                        SELECT * FROM alerts 
                        WHERE status = 'open' 
                        AND created_at < NOW() - INTERVAL '1 hour'
                        ORDER BY created_at ASC
                    """)
                
                for record in records:
                    alert = Alert(
                        alert_id=record['alert_id'],
                        video_id=record['video_id'],
                        frame_id=record['frame_id'],
                        alert_type=record['alert_type'],
                        severity=AlertSeverity(record['severity']),
                        title=record['title'],
                        description=record['description'],
                        confidence=record['confidence'],
                        metadata=json.loads(record['metadata']),
                        status=AlertStatus(record['status']),
                        created_at=record['created_at']
                    )
                    
                    if self.escalation.should_escalate(alert):
                        await self._escalate_alert(alert)
                
            except Exception as e:
                logger.error(f"Error checking escalations: {e}")
    
    async def _escalate_alert(self, alert: Alert):
        """Escalate alert"""
        escalation_message = self.escalation.get_escalation_message(alert)
        
        # Send escalation notifications
        await self._send_notifications(alert)
        
        # Update alert status
        await self.update_alert_status(alert.alert_id, AlertStatus.ACKNOWLEDGED)
        
        logger.warning(f"Alert escalated: {alert.alert_id}")
    
    async def update_alert_status(self, alert_id: str, status: AlertStatus, 
                                 assigned_to: Optional[str] = None) -> bool:
        """Update alert status"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE alerts 
                    SET status = $1, assigned_to = $2, updated_at = $3,
                        resolved_at = CASE WHEN $1 IN ('resolved', 'false_positive') 
                                        THEN $3 ELSE resolved_at END
                    WHERE alert_id = $4
                """, status.value, assigned_to, datetime.utcnow(), alert_id)
            
            logger.info(f"Alert status updated: {alert_id} -> {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update alert status: {e}")
            return False
    
    async def get_alerts(self, status: Optional[AlertStatus] = None,
                        severity: Optional[AlertSeverity] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Get alerts with filters"""
        try:
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []
            param_count = 0
            
            if status:
                param_count += 1
                query += f" AND status = ${param_count}"
                params.append(status.value)
            
            if severity:
                param_count += 1
                query += f" AND severity = ${param_count}"
                params.append(severity.value)
            
            query += " ORDER BY created_at DESC LIMIT $" + str(param_count + 1)
            params.append(limit)
            
            async with self.db_pool.acquire() as conn:
                records = await conn.fetch(query, *params)
                
                return [dict(record) for record in records]
                
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Total alerts
                total_alerts = await conn.fetchval("SELECT COUNT(*) FROM alerts")
                
                # Alerts by severity
                severity_stats = await conn.fetch("""
                    SELECT severity, COUNT(*) as count 
                    FROM alerts 
                    GROUP BY severity
                """)
                
                # Alerts by status
                status_stats = await conn.fetch("""
                    SELECT status, COUNT(*) as count 
                    FROM alerts 
                    GROUP BY status
                """)
                
                # Recent alerts (last 24 hours)
                recent_alerts = await conn.fetchval("""
                    SELECT COUNT(*) FROM alerts 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
                
                return {
                    "total_alerts": total_alerts,
                    "recent_alerts_24h": recent_alerts,
                    "by_severity": {row['severity']: row['count'] for row in severity_stats},
                    "by_status": {row['status']: row['count'] for row in status_stats}
                }
                
        except Exception as e:
            logger.error(f"Failed to get alert statistics: {e}")
            return {}
    
    async def cleanup_old_alerts(self, days: int = 30):
        """Clean up old resolved alerts"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            async with self.db_pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM alerts 
                    WHERE status IN ('resolved', 'false_positive') 
                    AND created_at < $1
                """, cutoff_date)
            
            logger.info(f"Cleaned up old alerts: {result}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old alerts: {e}")
    
    async def shutdown(self):
        """Shutdown alert manager"""
        logger.info("Shutting down production alert manager...")
        
        self.processing = False
        
        if self.db_pool:
            await self.db_pool.close()
        
        logger.info("Production alert manager shutdown complete")

# Global instance
production_alert_manager = ProductionAlertManager()
