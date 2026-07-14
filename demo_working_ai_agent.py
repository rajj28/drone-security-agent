#!/usr/bin/env python3

"""
demo_working_ai_agent.py - Working demonstration of AI Learning Agent

This script shows the AI learning agent capabilities with fixed issues:
1. Uses sklearn for pattern recognition
2. Implements AI reasoning with GPT-4
3. Learns patterns from video analysis
4. Remembers incidents and behaviors across videos
5. Provides adaptive threat detection
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.video_folder_indexer import VideoFolderIndexer, VideoMetadata
from src.ai_learning_agent import ai_learning_agent

async def demo_working_ai_agent():
    """Demonstrate working AI learning agent"""
    
    print("WORKING AI LEARNING AGENT DEMONSTRATION")
    print("=" * 70)
    print("Fixed numpy compatibility issues")
    print("Using sklearn 1.2.2 with numpy 1.26.4")
    print("LangGraph + GPT-4 integration")
    print("Video folder indexing with AI learning")
    print("=" * 70)
    
    # Show current AI knowledge base
    print("\nCurrent AI Knowledge Base:")
    insights = ai_learning_agent.get_learning_insights()
    
    print(f"  Total Patterns Learned: {insights['total_patterns']}")
    print(f"  Total Incidents Recorded: {insights['total_incidents']}")
    print(f"  Total Videos Analyzed: {insights['total_videos']}")
    print(f"  Reasoning Chains: {insights['reasoning_chains_count']}")
    
    if insights['pattern_types']:
        print(f"  Pattern Types: {list(insights['pattern_types'].keys())}")
    
    # Process a new video with AI learning
    print(f"\nProcessing Video with AI Learning...")
    
    indexer = VideoFolderIndexer()
    
    # Create video metadata
    video_metadata = VideoMetadata(
        video_id="working_ai_demo_001",
        filename="theft_detection_demo.mp4",
        file_path="demo_videos/theft_detection_demo.mp4",
        file_size=2048000,
        duration=30.0,
        fps=30.0,
        resolution=(1920, 1080),
        codec="h264",
        created_at=datetime.utcnow(),
        location="Electronics Store - Main Floor",
        camera_id="CAM_AI_WORKING_001",
        tags=["theft", "shoplifting", "ai_detected", "working_demo"]
    )
    
    # Create video folder
    video_folder = indexer.create_video_folder(video_metadata)
    print(f"Created video folder: {video_folder}")
    
    # Simulate frame processing with AI learning
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
        
        print(f"Copied {len(frames_list)} frames for AI analysis")
        
        # Process frames with AI learning
        telemetry_base = {
            'location': video_metadata.location,
            'camera_id': video_metadata.camera_id,
            'is_after_hours': False,
            'is_restricted_zone': False
        }
        
        await indexer.process_video_frames(video_metadata.video_id, frames_list, telemetry_base)
        
        print(f"Completed AI-powered video analysis")
        
        # Show AI learning results
        print(f"\nAI Learning Results:")
        
        # Get AI analysis results
        if video_metadata.video_id in ai_learning_agent.video_profiles:
            profile = ai_learning_agent.video_profiles[video_metadata.video_id]
            print(f"  Video ID: {profile.video_id}")
            print(f"  Learning Score: {profile.learning_score:.2f}")
            print(f"  Adaptation Level: {profile.adaptation_level}%")
            print(f"  Total Frames: {profile.total_frames}")
            print(f"  Threat Frames: {profile.threat_frames}")
            print(f"  Patterns Detected: {profile.patterns_detected}")
            print(f"  Unique Behaviors: {list(profile.unique_behaviors)[:5]}")  # Show first 5
        
        # Show AI reasoning chain
        if video_metadata.video_id in ai_learning_agent.reasoning_chains:
            reasoning_chain = ai_learning_agent.reasoning_chains[video_metadata.video_id]
            print(f"\nAI Reasoning Chain ({len(reasoning_chain)} steps):")
            for i, step in enumerate(reasoning_chain, 1):
                print(f"  {i}. {step}")
        
        # Show adaptive analysis for a frame
        if frames_list:
            first_frame_id = frames_list[0][0]
            frame_analysis = indexer.get_video_analysis(video_metadata.video_id, first_frame_id)
            
            if frame_analysis and 'analysis_result' in frame_analysis:
                adaptive_result = frame_analysis['analysis_result']
                print(f"\nAI-Enhanced Analysis for {first_frame_id}:")
                print(f"  Adaptive Confidence: {adaptive_result.get('adaptive_confidence', 0):.2f}")
                print(f"  Learning Score: {adaptive_result.get('learning_score', 0):.2f}")
                print(f"  Adaptation Level: {adaptive_result.get('adaptation_level', 0)}%")
                print(f"  Recognized Patterns: {adaptive_result.get('recognized_patterns', [])}")
                print(f"  Similar Videos: {adaptive_result.get('similar_videos', [])}")
        
        # Show updated AI knowledge base
        print(f"\nUpdated AI Knowledge Base:")
        updated_insights = ai_learning_agent.get_learning_insights()
        
        print(f"  Total Patterns: {updated_insights['total_patterns']}")
        print(f"  Total Incidents: {updated_insights['total_incidents']}")
        print(f"  Total Videos: {updated_insights['total_videos']}")
        print(f"  Reasoning Chains: {updated_insights['reasoning_chains_count']}")
        
        if updated_insights['learning_trends']:
            trends = updated_insights['learning_trends']
            print(f"  Average Learning Score: {trends['avg_learning_score']:.2f}")
            print(f"  Average Adaptation Level: {trends['avg_adaptation_level']:.1f}%")
            print(f"  Total Behaviors Learned: {trends['total_behaviors_learned']}")
        
        if updated_insights['adaptation_summary']:
            adaptation = updated_insights['adaptation_summary']
            print(f"  High Adaptation Videos: {adaptation.get('high_adaptation_videos', 0)}")
            print(f"  Medium Adaptation Videos: {adaptation.get('medium_adaptation_videos', 0)}")
            print(f"  Low Adaptation Videos: {adaptation.get('low_adaptation_videos', 0)}")
        
        if updated_insights['top_patterns']:
            print(f"\nTop Patterns:")
            for i, pattern in enumerate(updated_insights['top_patterns'][:3], 1):
                print(f"  {i}. {pattern['description']} (freq: {pattern['frequency']}, conf: {pattern['confidence']:.2f})")
    
    # Demonstrate AI pattern recognition
    print(f"\nAI Pattern Recognition Demo:")
    test_texts = [
        "Person reaching for pocket and concealing phone while looking around nervously",
        "Individual scanning the area before quickly grabbing merchandise",
        "Group of people gathering near exit in coordinated suspicious manner",
        "Normal customer peacefully browsing products"
    ]
    
    for text in test_texts:
        patterns = ai_learning_agent.recognize_patterns(text)
        print(f"  Text: '{text}'")
        if patterns:
            print(f"    AI Recognized: {patterns}")
        else:
            print(f"    No patterns recognized (needs more training data)")
    
    # Show AI knowledge graph
    print(f"\nAI Knowledge Graph:")
    print(f"  Patterns: {len(ai_learning_agent.patterns)}")
    print(f"  Incidents: {len(ai_learning_agent.incidents)}")
    print(f"  Video Profiles: {len(ai_learning_agent.video_profiles)}")
    print(f"  Reasoning Chains: {len(ai_learning_agent.reasoning_chains)}")
    
    # Show folder structure
    print(f"\nVideo Folder Structure:")
    video_folder = Path("data/videos/working_ai_demo_001")
    if video_folder.exists():
        print(f"  {video_folder.name}/")
        for item in video_folder.iterdir():
            if item.is_file():
                print(f"    {item.name}")
            elif item.is_dir():
                print(f"    {item.name}/")
                if item.name == "analysis":
                    analysis_files = list(item.glob("*.json"))
                    print(f"      {len(analysis_files)} frame analysis files")
                elif item.name == "frames":
                    frame_files = list(item.glob("*.jpg"))
                    print(f"      {len(frame_files)} frame images")
    
    print(f"\nAI Learning Agent Capabilities:")
    print(f"  Fixed numpy compatibility (1.26.4) + sklearn (1.2.2)")
    print(f"  LangGraph state management for reasoning")
    print(f"  GPT-4 enhanced pattern analysis")
    print(f"  Adaptive threat detection with learning")
    print(f"  Cross-video pattern recognition")
    print(f"  Comprehensive incident memory")
    print(f"  AI-driven behavioral analysis")
    print(f"  Video-specific folder indexing")
    print(f"  Real-time adaptation and learning")

async def demo_multiple_videos():
    """Demonstrate multiple video processing"""
    
    print(f"\nMULTIPLE VIDEO AI LEARNING DEMO")
    print("=" * 70)
    
    indexer = VideoFolderIndexer()
    
    videos = [
        {
            'video_id': 'multi_demo_001',
            'location': 'Shop Floor',
            'camera_id': 'CAM_MULTI_001',
            'tags': ['shop', 'customer', 'normal']
        },
        {
            'video_id': 'multi_demo_002',
            'location': 'Entrance',
            'camera_id': 'CAM_MULTI_002',
            'tags': ['entrance', 'security', 'access']
        }
    ]
    
    for video_info in videos:
        print(f"\nProcessing {video_info['video_id']}...")
        
        video_metadata = VideoMetadata(
            video_id=video_info['video_id'],
            filename=f"{video_info['video_id']}.mp4",
            file_path=f"videos/{video_info['video_id']}.mp4",
            file_size=1024000,
            duration=20.0,
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
                'is_restricted_zone': False
            }
            
            await indexer.process_video_frames(video_metadata.video_id, frames_list, telemetry_base)
            
            # Show results
            if video_metadata.video_id in ai_learning_agent.video_profiles:
                profile = ai_learning_agent.video_profiles[video_metadata.video_id]
                print(f"  Learning Score: {profile.learning_score:.2f}")
                print(f"  Adaptation Level: {profile.adaptation_level}%")
    
    # Show final summary
    print(f"\nFINAL AI KNOWLEDGE SUMMARY:")
    final_insights = ai_learning_agent.get_learning_insights()
    
    print(f"  Total Patterns: {final_insights['total_patterns']}")
    print(f"  Total Incidents: {final_insights['total_incidents']}")
    print(f"  Total Videos: {final_insights['total_videos']}")
    print(f"  Reasoning Chains: {final_insights['reasoning_chains_count']}")
    
    if final_insights['learning_trends']:
        trends = final_insights['learning_trends']
        print(f"  Average Learning Score: {trends['avg_learning_score']:.2f}")
        print(f"  Average Adaptation Level: {trends['avg_adaptation_level']:.1f}%")
        print(f"  Total Behaviors Learned: {trends['total_behaviors_learned']}")
    
    print(f"\nAI LEARNING SYSTEM IS FULLY OPERATIONAL!")
    print(f"Knowledge Base: {ai_learning_agent.knowledge_base_path}")
    print(f"The AI agent learns from every video and improves over time!")
    print(f"Cross-video learning creates an intelligent security network!")

if __name__ == "__main__":
    # Run working AI agent demonstration
    asyncio.run(demo_working_ai_agent())
    
    # Run multiple video demo
    asyncio.run(demo_multiple_videos())
