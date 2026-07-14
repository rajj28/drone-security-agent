#!/usr/bin/env python3

"""
demo_simple_learning_agent.py - Demonstration of the intelligent learning agent

This script shows how the learning agent:
1. Learns patterns from video analysis
2. Remembers incidents and behaviors
3. Adapts threat detection over time
4. Builds cumulative knowledge
5. Provides adaptive analysis
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.video_folder_indexer import VideoFolderIndexer, VideoMetadata
from src.simple_learning_agent import simple_learning_agent

async def demo_learning_capabilities():
    """Demonstrate learning agent capabilities"""
    
    print("INTELLIGENT LEARNING AGENT DEMONSTRATION")
    print("=" * 60)
    
    # Show current knowledge base
    print("\nCurrent Knowledge Base:")
    insights = simple_learning_agent.get_learning_insights()
    
    print(f"  Total Patterns Learned: {insights['total_patterns']}")
    print(f"  Total Incidents Recorded: {insights['total_incidents']}")
    print(f"  Total Videos Analyzed: {insights['total_videos']}")
    print(f"  Knowledge Graph Size: {insights['knowledge_graph_size']}")
    
    if insights['pattern_types']:
        print(f"  Pattern Types: {list(insights['pattern_types'].keys())}")
    
    if insights['top_patterns']:
        print(f"  Top Pattern: {insights['top_patterns'][0]['description']} (frequency: {insights['top_patterns'][0]['frequency']})")
    
    # Process a new video with learning
    print(f"\nProcessing New Video with Learning...")
    
    indexer = VideoFolderIndexer()
    
    # Create video metadata
    video_metadata = VideoMetadata(
        video_id="learning_demo_001",
        filename="shoplifting_demo.mp4",
        file_path="demo_videos/shoplifting_demo.mp4",
        file_size=2048000,
        duration=45.0,
        fps=30.0,
        resolution=(1920, 1080),
        codec="h264",
        created_at=datetime.utcnow(),
        location="Electronics Store",
        camera_id="CAM_004",
        tags=["shoplifting", "electronics", "theft"]
    )
    
    # Create video folder
    video_folder = indexer.create_video_folder(video_metadata)
    print(f"Created video folder: {video_folder}")
    
    # Simulate frame processing with learning
    frames_dir = Path("data/theft_analysis_frames")
    if frames_dir.exists():
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))[:3]  # Use 3 frames for demo
        
        frames_list = []
        for i, frame_file in enumerate(frame_files):
            frame_id = f"{video_metadata.video_id}_frame_{i+1:03d}"
            timestamp = i * 2.0  # 2 second intervals
            frame_number = i + 1
            
            # Copy frame to video folder
            dest_frame = video_folder / indexer.frames_dir_name / f"frame_{i+1:06d}.jpg"
            import shutil
            shutil.copy2(frame_file, dest_frame)
            
            frames_list.append((frame_id, timestamp, frame_number))
        
        print(f"Copied {len(frames_list)} frames for learning analysis")
        
        # Process frames with learning
        telemetry_base = {
            'location': video_metadata.location,
            'camera_id': video_metadata.camera_id,
            'is_after_hours': False,
            'is_restricted_zone': False
        }
        
        await indexer.process_video_frames(video_metadata.video_id, frames_list, telemetry_base)
        
        print(f"Completed learning analysis")
        
        # Show learning results
        print(f"\nLearning Results for {video_metadata.video_id}:")
        
        # Get video learning profile
        profile = simple_learning_agent.video_profiles.get(video_metadata.video_id)
        if profile:
            print(f"  Learning Score: {profile.learning_score:.2f}")
            print(f"  Adaptation Level: {profile.adaptation_level}%")
            print(f"  Patterns Detected: {profile.patterns_detected}")
            print(f"  Unique Behaviors: {list(profile.unique_behaviors)}")
            print(f"  Threat Frames: {profile.threat_frames}/{profile.total_frames}")
        
        # Show adaptive analysis for a frame
        if frames_list:
            first_frame_id = frames_list[0][0]
            frame_analysis = indexer.get_video_analysis(video_metadata.video_id, first_frame_id)
            
            if 'adaptive_confidence' in frame_analysis.get('analysis_result', {}):
                adaptive_result = frame_analysis['analysis_result']
                print(f"\nAdaptive Analysis for {first_frame_id}:")
                print(f"  Original Confidence: {adaptive_result.get('original_analysis', {}).get('gpt4o_enhanced', {}).get('confidence', 0):.2f}")
                print(f"  Adaptive Confidence: {adaptive_result.get('adaptive_confidence', 0):.2f}")
                print(f"  Recognized Patterns: {adaptive_result.get('recognized_patterns', [])}")
                print(f"  Similar Videos: {adaptive_result.get('similar_videos', [])}")
        
        # Show updated knowledge base
        print(f"\nUpdated Knowledge Base:")
        updated_insights = simple_learning_agent.get_learning_insights()
        
        print(f"  Total Patterns: {updated_insights['total_patterns']} (+{updated_insights['total_patterns'] - insights['total_patterns']})")
        print(f"  Total Incidents: {updated_insights['total_incidents']} (+{updated_insights['total_incidents'] - insights['total_incidents']})")
        print(f"  Total Videos: {updated_insights['total_videos']} (+{updated_insights['total_videos'] - insights['total_videos']})")
        
        if updated_insights['top_patterns']:
            latest_pattern = updated_insights['top_patterns'][0]
            print(f"  Latest Pattern: {latest_pattern['description']} (confidence: {latest_pattern['confidence']:.2f})")
        
        # Show learning trends
        if 'learning_trends' in updated_insights:
            trends = updated_insights['learning_trends']
            print(f"  Average Learning Score: {trends['avg_learning_score']:.2f}")
            print(f"  Average Adaptation Level: {trends['avg_adaptation_level']:.1f}%")
            print(f"  Total Behaviors Learned: {trends['total_behaviors_learned']}")
    
    # Demonstrate pattern recognition
    print(f"\nPattern Recognition Demo:")
    test_texts = [
        "Person reaching for pocket and concealing phone",
        "Individual looking around nervously before grabbing item",
        "Group of people gathering near exit suspiciously",
        "Normal customer browsing products peacefully"
    ]
    
    for text in test_texts:
        patterns = simple_learning_agent.recognize_patterns(text)
        print(f"  Text: '{text}'")
        print(f"    Recognized: {patterns}")
    
    # Show knowledge graph connections
    print(f"\nKnowledge Graph Connections:")
    for pattern, videos in list(simple_learning_agent.knowledge_graph.items())[:5]:
        if pattern.startswith('pattern_') or pattern.startswith('similar_videos_'):
            continue
        print(f"  {pattern}: {len(videos)} videos")
    
    print(f"\nLearning Agent Benefits:")
    print(f"  Learns from every video analyzed")
    print(f"  Remembers patterns and incidents")
    print(f"  Adapts threat detection over time")
    print(f"  Provides contextual analysis")
    print(f"  Builds cumulative knowledge")
    print(f"  Recognizes similar behaviors")
    print(f"  Improves confidence scoring")
    print(f"  Creates video-specific learning profiles")

async def demo_cross_video_learning():
    """Demonstrate cross-video pattern learning"""
    
    print(f"\nCROSS-VIDEO LEARNING DEMONSTRATION")
    print("=" * 60)
    
    # Process multiple videos to show cross-video learning
    indexer = VideoFolderIndexer()
    
    videos = [
        {
            'video_id': 'shop_cam_001',
            'location': 'Main Shop Floor',
            'camera_id': 'CAM_001',
            'tags': ['shop', 'customer', 'normal']
        },
        {
            'video_id': 'entrance_cam_002',
            'location': 'Main Entrance',
            'camera_id': 'CAM_002',
            'tags': ['entrance', 'security', 'access']
        },
        {
            'video_id': 'storage_cam_003',
            'location': 'Storage Area',
            'camera_id': 'CAM_003',
            'tags': ['storage', 'restricted', 'security']
        }
    ]
    
    # Process each video
    for video_info in videos:
        print(f"\nProcessing {video_info['video_id']}...")
        
        video_metadata = VideoMetadata(
            video_id=video_info['video_id'],
            filename=f"{video_info['video_id']}.mp4",
            file_path=f"videos/{video_info['video_id']}.mp4",
            file_size=1024000,
            duration=30.0,
            fps=30.0,
            resolution=(1920, 1080),
            codec="h264",
            created_at=datetime.utcnow(),
            location=video_info['location'],
            camera_id=video_info['camera_id'],
            tags=video_info['tags']
        )
        
        # Create folder and process
        video_folder = indexer.create_video_folder(video_metadata)
        
        # Use existing frames for demo
        frames_dir = Path("data/theft_analysis_frames")
        if frames_dir.exists():
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))[:2]
            
            frames_list = []
            for i, frame_file in enumerate(frame_files):
                frame_id = f"{video_metadata.video_id}_frame_{i+1:03d}"
                timestamp = i * 1.0
                frame_number = i + 1
                
                dest_frame = video_folder / indexer.frames_dir_name / f"frame_{i+1:06d}.jpg"
                import shutil
                shutil.copy2(frame_file, dest_frame)
                
                frames_list.append((frame_id, timestamp, frame_number))
            
            telemetry_base = {
                'location': video_info['location'],
                'camera_id': video_info['camera_id'],
                'is_after_hours': False,
                'is_restricted_zone': 'restricted' in video_info['tags']
            }
            
            await indexer.process_video_frames(video_metadata.video_id, frames_list, telemetry_base)
            
            # Show learning profile
            profile = simple_learning_agent.video_profiles.get(video_metadata.video_id)
            if profile:
                print(f"  Learning Score: {profile.learning_score:.2f}")
                print(f"  Adaptation Level: {profile.adaptation_level}%")
    
    # Show cross-video insights
    print(f"\nCross-Video Learning Insights:")
    
    # Show video similarities
    videos = list(simple_learning_agent.video_profiles.keys())
    for i, video1 in enumerate(videos):
        for video2 in videos[i+1:]:
            similarity = simple_learning_agent.calculate_video_similarity(video1, video2)
            if similarity > 0.3:
                print(f"  {video1} ↔ {video2}: {similarity:.2f} similarity")
    
    # Show shared patterns
    pattern_videos = defaultdict(list)
    for pattern, video_list in simple_learning_agent.knowledge_graph.items():
        if pattern.startswith('pattern_') or 'pattern' in pattern:
            pattern_videos[pattern].extend(video_list)
    
    print(f"\nShared Patterns Across Videos:")
    for pattern, video_list in pattern_videos.items():
        if len(video_list) > 1:
            print(f"  {pattern}: {len(video_list)} videos")
    
    # Final knowledge summary
    print(f"\nFinal Knowledge Summary:")
    final_insights = simple_learning_agent.get_learning_insights()
    
    print(f"  Total Patterns: {final_insights['total_patterns']}")
    print(f"  Total Incidents: {final_insights['total_incidents']}")
    print(f"  Total Videos: {final_insights['total_videos']}")
    print(f"  Knowledge Graph Size: {final_insights['knowledge_graph_size']}")
    
    if 'adaptation_summary' in final_insights:
        adaptation = final_insights['adaptation_summary']
        print(f"  High Adaptation Videos: {adaptation['high_adaptation_videos']}")
        print(f"  Medium Adaptation Videos: {adaptation['medium_adaptation_videos']}")
        print(f"  Low Adaptation Videos: {adaptation['low_adaptation_videos']}")

if __name__ == "__main__":
    from collections import defaultdict
    
    # Run learning demonstration
    asyncio.run(demo_learning_capabilities())
    
    # Run cross-video learning
    asyncio.run(demo_cross_video_learning())
    
    print(f"\nLEARNING AGENT DEMONSTRATION COMPLETE!")
    print(f"Knowledge Base: {simple_learning_agent.knowledge_base_path}")
    print(f"The agent is now smarter and will continue learning from each video!")
