"""MongoDB storage for Railway deployment persistence."""
import os
import io
from datetime import datetime
from typing import Dict, List, Optional, Any
from pymongo import MongoClient, ASCENDING
from gridfs import GridFS
import logging

logger = logging.getLogger(__name__)

class MongoDBStorage:
    """MongoDB storage for sessions, frames, and images."""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.fs = None
        self.sessions_collection = None
        self._connect()
    
    def _connect(self):
        """Connect to MongoDB with proper SSL handling for Cloud Run."""
        try:
            # Get MongoDB URI from environment
            mongo_uri = os.environ.get('MONGODB_URI')
            
            if not mongo_uri:
                logger.warning("MONGODB_URI not set, using local MongoDB")
                mongo_uri = "mongodb://localhost:27017/"
            
            # Connection options for Cloud Run / container environments
            # Note: retryWrites and w='majority' removed due to pymongo transaction compatibility issues
            connection_options = {
                'serverSelectionTimeoutMS': 10000,
                'connectTimeoutMS': 10000,
                'socketTimeoutMS': 30000
            }
            
            # Handle SSL/TLS for different URI types
            if mongo_uri.startswith('mongodb+srv://'):
                # SRV connection string - SSL usually enabled by default
                logger.info("Using MongoDB SRV connection string")
                # For Atlas SRV connections, sometimes we need to allow invalid certs
                # due to container CA certificate issues
                connection_options['tlsAllowInvalidCertificates'] = True
                self.client = MongoClient(mongo_uri, **connection_options)
            elif 'ssl=true' in mongo_uri.lower() or 'tls=true' in mongo_uri.lower():
                # Non-SRV with SSL enabled
                logger.info("Using MongoDB non-SRV connection with SSL")
                connection_options['tlsAllowInvalidCertificates'] = True
                self.client = MongoClient(mongo_uri, **connection_options)
            else:
                # Local or non-SSL connection
                logger.info("Using MongoDB connection without SSL")
                self.client = MongoClient(mongo_uri, **connection_options)
            
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client.drone_security
            self.fs = GridFS(self.db)
            self.sessions_collection = self.db.sessions
            
            # Create indexes
            self.sessions_collection.create_index("session_id", unique=True)
            self.sessions_collection.create_index("created_at", ASCENDING)
            
            logger.info("✅ MongoDB connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.client = None
            return False
    
    def is_connected(self) -> bool:
        """Check if MongoDB is connected."""
        if not self.client:
            return False
        try:
            self.client.admin.command('ping')
            return True
        except:
            return False
    
    def save_session(self, session_id: str, data: Dict[str, Any]):
        """Save session data to MongoDB."""
        if not self.is_connected():
            logger.warning("MongoDB not connected, skipping session save")
            return False
        
        try:
            data['updated_at'] = datetime.utcnow()
            data['session_id'] = session_id
            
            self.sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": data},
                upsert=True
            )
            logger.info(f"✅ Session {session_id} saved to MongoDB")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save session: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data from MongoDB."""
        if not self.is_connected():
            return None
        
        try:
            return self.sessions_collection.find_one(
                {"session_id": session_id},
                {"_id": 0}  # Exclude MongoDB _id
            )
        except Exception as e:
            logger.error(f"❌ Failed to get session: {e}")
            return None
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions from MongoDB."""
        if not self.is_connected():
            return []
        
        try:
            sessions = list(self.sessions_collection.find({}, {"_id": 0}))
            return sessions
        except Exception as e:
            logger.error(f"❌ Failed to get sessions: {e}")
            return []
    
    def save_frame_image(self, session_id: str, frame_name: str, image_data: bytes) -> str:
        """Save frame image to GridFS."""
        if not self.is_connected():
            logger.warning("MongoDB not connected, cannot save image")
            return ""
        
        try:
            # Create unique filename
            gridfs_filename = f"{session_id}/{frame_name}"
            
            # Check if file already exists
            existing = self.fs.find_one({"filename": gridfs_filename})
            if existing:
                return str(existing._id)
            
            # Save to GridFS
            file_id = self.fs.put(
                image_data,
                filename=gridfs_filename,
                session_id=session_id,
                frame_name=frame_name,
                content_type="image/jpeg"
            )
            
            logger.info(f"✅ Frame {frame_name} saved to GridFS (ID: {file_id})")
            return str(file_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to save frame image: {e}")
            return ""
    
    def get_frame_image(self, session_id: str, frame_name: str) -> Optional[bytes]:
        """Get frame image from GridFS."""
        if not self.is_connected():
            return None
        
        try:
            gridfs_filename = f"{session_id}/{frame_name}"
            file_obj = self.fs.find_one({"filename": gridfs_filename})
            
            if file_obj:
                return file_obj.read()
            
            # Try without session prefix
            file_obj = self.fs.find_one({"filename": frame_name})
            if file_obj:
                return file_obj.read()
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get frame image: {e}")
            return None
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Singleton instance
_mongo_storage = None

def get_mongodb_storage() -> MongoDBStorage:
    """Get or create MongoDB storage instance."""
    global _mongo_storage
    if _mongo_storage is None:
        _mongo_storage = MongoDBStorage()
    return _mongo_storage
