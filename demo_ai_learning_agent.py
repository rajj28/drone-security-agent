#!/usr/bin/env python3

"""
demo_ai_learning_agent.py - Demonstration of the Advanced AI Learning Agent

This script shows how the AI learning agent using LangGraph:
1. Uses sophisticated reasoning with LangGraph
2. Learns patterns from video analysis
3. Remembers incidents and behaviors across videos
4. Provides adaptive threat detection
5. Builds comprehensive knowledge with AI reasoning
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.video_folder_indexer import VideoFolderIndexer, VideoMetadata
from src.ai_learning_agent import ai_learning_agent

async def demo_ai_learning_capabilities():
    """Demonstrate AI learning agent capabilities"""
    
    print("ADVANCED AI LEARNING AGENT DEMONSTRATION")
    print("=" * 70)
    print("Powered by LangGraph + GPT-4 + sklearn")
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
    
    if insights['top_patterns']:
        top_pattern = insights['top_patterns'][0]
        print(f"  Top Pattern: {top_pattern['description']} (frequency: {top_pattern['frequency']})")
    
    # Process a new video with AI learning
    print(f"\nProcessing New Video with AI Learning...")
    
    indexer = VideoFolderIndexer()
    
    # Create video metadata
    video_metadata = VideoMetadata(
        video_id="ai_demo_001",
        filename="advanced_theft_demo.mp4",
        file_path="demo_videos/advanced_theft_demo.mp4",
        file_size=3072000,
        duration=60.0,
        fps=30.0,
        resolution=(1920, 1080),
        codec="h264",
        created_at=datetime.utcnow(),
        location="High-End Electronics Store",
        camera_id="CAM_AI_001",
        tags=["theft", "electronics", "sophisticated", "ai_detected"]
    )
    
    # Create video folder
    video_folder = indexer.create_video_folder(video_metadata)
    print(f"Created video folder: {video_folder}")
    
    # Simulate frame processing with AI learning
    frames_dir = Path("data/theft_analysis_frames")
    if frames_dir.exists():
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))[:4]  # Use 4 frames for demo
        
        frames_list = []
        for i, frame_file in enumerate(frame_files):
            frame_id = f"{video_metadata.video_id}_frame_{i+1:03d}"
            timestamp = i * 3.0  # 3 second intervals
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
        
        print(f"Completed AI-powered analysis")
        
        # Show AI learning results
        print(f"\nAI Learning Results for {video_metadata.video_id}:")
        
        # Get AI analysis from the agent
        if video_metadata.video_id in ai_learning_agent.video_profiles:
            profile = ai_learning_agent.video_profiles[video_metadata.video_id]
            print(f"  Learning Score: {profile.learning_score:.2f}")
            print(f"  Adaptation Level: {profile.adaptation_level}%")
            print(f"  Patterns Detected: {profile.patterns_detected}")
            print(f"  Unique Behaviors: {list(profile.unique_behaviors)}")
            print(f"  Threat Frames: {profile.threat_frames}/{profile.total_frames}")
        
        # Show AI reasoning chain
        if video_metadata.video_id in ai_learning_agent.reasoning_chains:
            reasoning_chain = ai_learning_agent.reasoning_chains[video_metadata.video_id]
            print(f"\nAI Reasoning Chain:")
            for i, step in enumerate(reasoning_chain, 1):
                print(f"  {i}. {step}")
        
        # Show adaptive analysis for a frame
        if frames_list:
            first_frame_id = frames_list[0][0]
            frame_analysis = indexer.get_video_analysis(video_metadata.video_id, first_frame_id)
            
            if 'adaptive_confidence' in frame_analysis.get('analysis_result', {}):
                adaptive_result = frame_analysis['analysis_result']
                print(f"\nAI-Enhanced Analysis for {first_frame_id}:")
                print(f"  Original Confidence: {adaptive_result.get('original_analysis', {}).get('gpt4o_enhanced', {}).get('confidence', 0):.2f}")
                print(f"  AI Adaptive Confidence: {adaptive_result.get('adaptive_confidence', 0):.2f}")
                print(f"  Recognized Patterns: {adaptive_result.get('recognized_patterns', [])}")
                print(f"  Similar Videos: {adaptive_result.get('similar_videos', [])}")
                print(f"  AI Reasoning: {adaptive_result.get('ai_reasoning', 'N/A')}")
        
        # Show updated AI knowledge base
        print(f"\nUpdated AI Knowledge Base:")
        updated_insights = ai_learning_agent.get_learning_insights()
        
        print(f"  Total Patterns: {updated_insights['total_patterns']} (+{updated_insights['total_patterns'] - insights['total_patterns']})")
        print(f"  Total Incidents: {updated_insights['total_incidents']} (+{updated_insights['total_incidents'] - insights['total_incidents']})")
        print(f"  Total Videos: {updated_insights['total_videos']} (+{updated_insights['total_videos'] - insights['total_videos']})")
        print(f"  Reasoning Chains: {updated_insights['reasoning_chains_count']}")
        
        if updated_insights['top_patterns']:
            latest_pattern = updated_insights['top_patterns'][0]
            print(f"  Latest Pattern: {latest_pattern['description']} (confidence: {latest_pattern['confidence']:.2f})")
        
        # Show AI learning trends
        if 'learning_trends' in updated_insights:
            trends = updated_insights['learning_trends']
            print(f"  Average Learning Score: {trends['avg_learning_score']:.2f}")
            print(f"  Average Adaptation Level: {trends['avg_adaptation_level']:.1f}%")
            print(f"  Total Behaviors Learned: {trends['total_behaviors_learned']}")
    
    # Demonstrate AI pattern recognition
    print(f"\nAI Pattern Recognition Demo:")
    test_texts = [
        "Person reaching for pocket and concealing phone while looking around nervously",
        "Individual scanning the area before quickly grabbing merchandise",
        "Group of people gathering near exit in coordinated suspicious manner",
        "Normal customer peacefully browsing products with no suspicious behavior"
    ]
    
    for text in test_texts:
        patterns = ai_learning_agent.recognize_patterns(text)
        print(f"  Text: '{text}'")
        print(f"    AI Recognized: {patterns}")
    
    # Show AI knowledge graph connections
    print(f"\nAI Knowledge Graph Connections:")
    for pattern, videos in list(ai_learning_agent.reasoning_chains.items())[:3]:
        print(f"  {pattern}: {len(videos)} reasoning steps")
    
    print(f"\nAI Learning Agent Benefits:")
    print(f"  LangGraph-powered reasoning chains")
    print(f"  GPT-4 enhanced pattern analysis")
    print(f"  Sophisticated threat assessment")
    print(f"  Cross-video learning with AI")
    print(f"  Adaptive confidence scoring")
    print(f"  Comprehensive incident memory")
    print(f"  AI-driven behavioral analysis")
    print(f"  Contextual threat reasoning")

async def demo_cross_video_ai_learning():
    """Demonstrate cross-video AI learning"""
    
    print(f"\nCROSS-VIDEO AI LEARNING DEMONSTRATION")
    print("=" * 70)
    
    # Process multiple videos to show AI cross-video learning
    indexer = VideoFolderIndexer()
    
    videos = [
        {
            'video_id': 'ai_shop_cam_001',
            'location': 'Main Shop Floor',
            'camera_id': 'CAM_AI_001',
            'tags': ['shop', 'customer', 'ai_monitored']
        },
        {
            'video_id': 'ai_entrance_cam_002',
            'location': 'Main Entrance',
            'camera_id': 'CAM_AI_002',
            'tags': ['entrance', 'security', 'ai_protected']
        },
        {
            'video_id': 'ai_storage_cam_003',
            'location': 'Restricted Storage Area',
            'camera_id': 'CAM_AI_003',
            'tags': ['storage', 'restricted', 'ai_secured']
        }
    ]
    
    # Process each video with AI
    for video_info in videos:
        print(f"\nAI Processing {video_info['video_id']}...")
        
        video_metadata = VideoMetadata(
            video_id=video_info['video_id'],
            filename=f"{video_info['video_id']}.mp4",
            file_path=f"videos/{video_info['video_id']}.mp4",
            file_size=2048000,
            duration=45.0,
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
                timestamp = i * 2.0
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
            
            # Show AI learning profile
            if video_metadata.video_id in ai_learning_agent.video_profiles:
                profile = ai_learning_agent.video_profiles[video_metadata.video_id]
                print(f"  AI Learning Score: {profile.learning_score:.2f}")
                print(f"  AI Adaptation Level: {profile.adaptation_level}%")
                
                # Show AI reasoning
                if video_metadata.video_id in ai_learning_agent.reasoning_chains:
                    reasoning = ai_learning_agent.reasoning_chains[video_metadata.video_id]
                    print(f"  AI Reasoning Steps: {len(reasoning)}")
    
    # Show AI cross-video insights
    print(f"\nAI Cross-Video Learning Insights:")
    
    # Show video similarities using AI
    videos = list(ai_learning_agent.video_profiles.keys())
    for i, video1 in enumerate(videos):
        for video2 in videos[i+1:]:
            similarity = ai_learning_agent._find_similar_videos(video1)
            if video2 in similarity:
                print(f"  {video1} ↔ {video2}: AI-detected similarity")
    
    # Show AI shared patterns
    pattern_videos = defaultdict(list)
    for pattern in ai_learning_agent.patterns.values():
        for incident in ai_learning_agent.incidents.values():
            if any(p in incident.description.lower() for p in pattern.description.lower().split()):
                pattern_videos[pattern.pattern_id].append(incident.video_id)
    
    print(f"\nAI-Identified Shared Patterns:")
    for pattern_id, video_list in pattern_videos.items():
        if len(video_list) > 1:
            pattern = ai_learning_agent.patterns[pattern_id]
            print(f"  {pattern.description}: {len(set(video_list))} videos")
    
    # Final AI knowledge summary
    print(f"\nFinal AI Knowledge Summary:")
    final_insights = ai_learning_agent.get_learning_insights()
    
    print(f"  Total Patterns: {final_insights['total_patterns']}")
    print(f"  Total Incidents: {final_insights['total_incidents']}")
    print(f"  Total Videos: {final_insights['total_videos']}")
    print(f"  Reasoning Chains: {final_insights['reasoning_chains_count']}")
    
    if 'adaptation_summary' in final_insights:
        adaptation = final_insights['adaptation_summary']
        print(f"  High Adaptation Videos: {adaptation['high_adaptation_videos']}")
        print(f"  Medium Adaptation Videos: {adaptation['medium_adaptation_videos']}")
        print(f"  Low Adaptation Videos: {adaptation['low_adaptation_videos']}")
    
    # Show AI reasoning chain sample
    if ai_learning_agent.reasoning_chains:
        sample_video = list(ai_learning_agent.reasoning_chains.keys())[0]
        reasoning = ai_learning_agent.reasoning_chains[sample_video]
        print(f"\nSample AI Reasoning Chain for {sample_video}:")
        for i, step in enumerate(reasoning[:3], 1):  # Show first 3 steps
            print(f"  {i}. {step}")

if __name__ == "__main__":
    from collections import defaultdict
    
    # Run AI learning demonstration
    asyncio.run(demo_ai_learning_capabilities())
    
    # Run cross-video AI learning
    asyncio.run(demo_cross_video_ai_learning())
    
    print(f"\nAI LEARNING AGENT DEMONSTRATION COMPLETE!")
    print(f"AI Knowledge Base: {ai_learning_agent.knowledge_base_path}")
    print(f"The AI agent is now smarter with LangGraph reasoning!")
    print(f"Each video analysis improves the agent's intelligence!")
    print(f"Cross-video learning creates a security knowledge network!")
