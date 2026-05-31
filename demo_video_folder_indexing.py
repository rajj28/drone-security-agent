#!/usr/bin/env python3

"""
demo_video_folder_indexing.py - Demonstration of video-specific folder indexing

This script shows how new video indexing works:
1. Each video gets its own folder
2. All analysis results are stored within that folder
3. Data is consumed directly from the video folder
4. Independent indexing per video
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.video_folder_indexer import VideoFolderIndexer, VideoMetadata

async def demo_video_indexing():
    """Demonstrate the video folder indexing system"""
    
    print("🎥 VIDEO FOLDER INDEXING DEMONSTRATION")
    print("=" * 50)
    
    # Initialize the indexer
    indexer = VideoFolderIndexer()
    
    # Create sample video metadata
    video_metadata = VideoMetadata(
        video_id="demo_video_001",
        filename="theft_scenario.mp4",
        file_path="sample_videos/theft_scenario.mp4",
        file_size=1024000,  # 1MB
        duration=30.0,      # 30 seconds
        fps=30.0,
        resolution=(1920, 1080),
        codec="h264",
        created_at=datetime.utcnow(),
        location="Mobile Phone Shop",
        camera_id="CAM_001",
        tags=["theft", "shoplifting", "suspicious"]
    )
    
    print(f"📁 Creating video folder for: {video_metadata.video_id}")
    
    # Create video folder structure
    video_folder = indexer.create_video_folder(video_metadata)
    print(f"✅ Video folder created: {video_folder}")
    
    # Show folder structure
    print("\n📂 Folder Structure Created:")
    print(f"{video_metadata.video_id}/")
    print(f"├── {indexer.metadata_file}")
    print(f"├── {indexer.index_file}")
    print(f"├── {indexer.alerts_file}")
    print(f"├── {indexer.frames_dir_name}/")
    print(f"└── {indexer.analysis_dir_name}/")
    
    # Show metadata
    print(f"\n📋 Video Metadata:")
    metadata = indexer.load_video_metadata(video_metadata.video_id)
    print(f"  Video ID: {metadata.video_id}")
    print(f"  Filename: {metadata.filename}")
    print(f"  Location: {metadata.location}")
    print(f"  Duration: {metadata.duration}s")
    print(f"  Resolution: {metadata.resolution}")
    print(f"  Status: {metadata.indexing_status}")
    
    # Simulate frame processing with existing frames
    print(f"\n🎬 Simulating Frame Processing...")
    
    # Use existing frames from theft analysis
    frames_dir = Path("data/theft_analysis_frames")
    if frames_dir.exists():
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))[:5]  # Use first 5 frames
        
        frames_list = []
        for i, frame_file in enumerate(frame_files):
            frame_id = f"{video_metadata.video_id}_frame_{i+1:03d}"
            timestamp = i * 1.0  # 1 second intervals
            frame_number = i + 1
            
            # Copy frame to video folder
            dest_frame = video_folder / indexer.frames_dir_name / f"frame_{i+1:06d}.jpg"
            import shutil
            shutil.copy2(frame_file, dest_frame)
            
            frames_list.append((frame_id, timestamp, frame_number))
        
        print(f"✅ Copied {len(frames_list)} frames to video folder")
        
        # Process frames
        telemetry_base = {
            'location': metadata.location,
            'camera_id': metadata.camera_id,
            'is_after_hours': False,
            'is_restricted_zone': False
        }
        
        await indexer.process_video_frames(video_metadata.video_id, frames_list, telemetry_base)
        
        print(f"✅ Processed {len(frames_list)} frames")
        
        # Show results
        print(f"\n📊 Processing Results:")
        updated_metadata = indexer.load_video_metadata(video_metadata.video_id)
        print(f"  Total Frames: {updated_metadata.total_frames}")
        print(f"  Processed Frames: {updated_metadata.processed_frames}")
        print(f"  Threat Frames: {updated_metadata.threat_frames}")
        print(f"  Status: {updated_metadata.indexing_status}")
        
        # Show frame analyses
        print(f"\n🔍 Frame Analyses:")
        analysis_folder = video_folder / indexer.analysis_dir_name
        for analysis_file in sorted(analysis_folder.glob("*.json")):
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            
            print(f"  Frame {analysis['frame_number']}: {analysis['threat_level']} (confidence: {analysis['confidence']:.2f})")
        
        # Show alerts
        print(f"\n🚨 Generated Alerts:")
        alerts = indexer.get_video_alerts(video_metadata.video_id)
        for alert in alerts[:3]:  # Show first 3 alerts
            print(f"  {alert['title']}: {alert['description']}")
        
        # Show summary
        print(f"\n📈 Video Summary:")
        summary = indexer.get_video_analysis(video_metadata.video_id)
        if 'statistics' in summary:
            stats = summary['statistics']
            print(f"  Total Alerts: {stats['total_alerts']}")
            print(f"  High Alerts: {stats['high_alerts']}")
            print(f"  Medium Alerts: {stats['medium_alerts']}")
            print(f"  Low Alerts: {stats['low_alerts']}")
        
        # Demonstrate consumption from video folder
        print(f"\n🔄 Consuming Data from Video Folder:")
        
        # Search for threat frames
        threat_frames = indexer.search_video_frames(video_metadata.video_id, threat_level="high")
        print(f"  Found {len(threat_frames)} high-threat frames")
        
        # Get specific frame analysis
        if frames_list:
            first_frame_id = frames_list[0][0]
            frame_analysis = indexer.get_video_analysis(video_metadata.video_id, first_frame_id)
            print(f"  Frame {first_frame_id} analysis loaded: {'success' if 'frame_id' in frame_analysis else 'failed'}")
        
        # Show folder structure with actual files
        print(f"\n📁 Actual Folder Structure:")
        for root, dirs, files in os.walk(video_folder):
            level = root.replace(str(video_folder), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    
    else:
        print(f"❌ Sample frames not found at {frames_dir}")
    
    print(f"\n🎯 Key Benefits of Video Folder Indexing:")
    print(f"  ✅ Each video is completely self-contained")
    print(f"  ✅ All analysis data stored with the video")
    print(f"  ✅ Easy to move/copy entire video analysis")
    print(f"  ✅ Independent processing per video")
    print(f"  ✅ Fast access to video-specific data")
    print(f"  ✅ Scalable to thousands of videos")
    
    print(f"\n📂 Video Location: {video_folder}")
    print(f"📊 Summary File: {video_folder / indexer.summary_file}")
    print(f"🚨 Alerts File: {video_folder / indexer.alerts_file}")
    print(f"📋 Index File: {video_folder / indexer.index_file}")

async def demo_multiple_videos():
    """Demonstrate multiple videos being indexed independently"""
    
    print("\n🎥 MULTIPLE VIDEOS DEMONSTRATION")
    print("=" * 50)
    
    indexer = VideoFolderIndexer()
    
    # Create multiple video metadata
    videos = [
        VideoMetadata(
            video_id="shop_camera_001",
            filename="shop_001.mp4",
            file_path="videos/shop_001.mp4",
            file_size=2048000,
            duration=60.0,
            fps=30.0,
            resolution=(1920, 1080),
            codec="h264",
            created_at=datetime.utcnow(),
            location="Main Shop Floor",
            camera_id="CAM_001",
            tags=["shop", "customer", "normal"]
        ),
        VideoMetadata(
            video_id="entrance_camera_002",
            filename="entrance_002.mp4",
            file_path="videos/entrance_002.mp4",
            file_size=1536000,
            duration=45.0,
            fps=25.0,
            resolution=(1280, 720),
            codec="h264",
            created_at=datetime.utcnow(),
            location="Main Entrance",
            camera_id="CAM_002",
            tags=["entrance", "security", "access"]
        ),
        VideoMetadata(
            video_id="storage_camera_003",
            filename="storage_003.mp4",
            file_path="videos/storage_003.mp4",
            file_size=3072000,
            duration=90.0,
            fps=30.0,
            resolution=(1920, 1080),
            codec="h264",
            created_at=datetime.utcnow(),
            location="Storage Area",
            camera_id="CAM_003",
            tags=["storage", "restricted", "security"]
        )
    ]
    
    # Create folders for all videos
    for video in videos:
        folder = indexer.create_video_folder(video)
        print(f"✅ Created folder: {folder}")
    
    # List all videos
    all_videos = indexer.list_videos()
    print(f"\n📋 All Indexed Videos:")
    for video in all_videos:
        print(f"  {video['video_id']}: {video['location']} ({video['indexing_status']})")
    
    print(f"\n📊 Total Videos: {len(all_videos)}")

if __name__ == "__main__":
    import os
    
    # Run single video demo
    asyncio.run(demo_video_indexing())
    
    # Run multiple videos demo
    asyncio.run(demo_multiple_videos())
