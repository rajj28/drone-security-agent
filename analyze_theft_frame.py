#!/usr/bin/env python3

from src.full_enhanced_vision_analyzer import analyze_frame_full_enhanced
from src.vision_analyzer import analyze_frame
from pathlib import Path

def analyze_theft_frame():
    print("=== THEFT DETECTION ON FRAME 005 (EXACT THEFT FRAME) ===")
    print()
    
    # Test the exact theft frame
    image_path = Path('data/theft_analysis_frames/frame_005.jpg')
    telemetry = {
        'frame_id': 'frame_005',
        'timestamp': '16:00:32', 
        'location': 'Mobile Phone Shop',
        'unix_time': 1640000000 + 32,
        'is_after_hours': False,
        'is_restricted_zone': False
    }
    
    print("SCENARIO: Frame 005 - Exact moment of phone theft")
    print("=" * 50)
    
    # Original GPT-4o analysis
    print("\n1. ORIGINAL GPT-4o RESPONSE:")
    print("-" * 30)
    try:
        original_result = analyze_frame('frame_005', image_path, telemetry)
        if original_result:
            print(f"People detected: {original_result.get('people_detected', 0)}")
            print(f"Threat assessment: {original_result.get('threat_assessment', 'none')}")
            print(f"Security signals: {original_result.get('security_signals', [])}")
            print(f"Suspicious elements: {original_result.get('suspicious_elements', [])}")
            print(f"Alert reasoning: {original_result.get('alert_reasoning', 'none')}")
            
            # Check if theft was detected
            alert_reasoning = original_result.get('alert_reasoning', '').lower()
            if any(word in alert_reasoning for word in ['theft', 'steal', 'shoplift', 'take']):
                print("🎯 THEFT DETECTED: YES")
            else:
                print("❌ THEFT DETECTED: NO")
        else:
            print("No analysis returned")
    except Exception as e:
        print(f"Error: {e}")
    
    # Enhanced GPT-4o + CLIP analysis
    print("\n2. ENHANCED GPT-4o + CLIP RESPONSE:")
    print("-" * 30)
    try:
        enhanced_result = analyze_frame_full_enhanced(image_path, telemetry)
        
        print(f"CLIP Threat Score: {enhanced_result['clip_analysis']['threat_score']:.3f}")
        print(f"CLIP Normal Score: {enhanced_result['clip_analysis']['normal_score']:.3f}")
        print(f"Overall Threat Level: {enhanced_result['overall_threat_level']}")
        print(f"Basic Scene Description: {enhanced_result['basic_description']}")
        
        # Parse the enhanced response
        gpt4o_response = enhanced_result['gpt4o_enhanced']['enhanced_analysis']
        if gpt4o_response:
            print("\nEnhanced Analysis:")
            print(gpt4o_response)
            
            # Check for theft detection in enhanced response
            response_lower = gpt4o_response.lower()
            theft_keywords = ['theft', 'steal', 'shoplift', 'take', 'grab', 'pocket', 'conceal']
            found_theft = [word for word in theft_keywords if word in response_lower]
            
            if found_theft:
                print(f"\n🎯 THEFT DETECTED: YES - Keywords found: {found_theft}")
            else:
                print("\n❌ THEFT DETECTED: NO")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== THEFT DETECTION COMPARISON ===")
    print("Frame 005 is the exact theft moment - testing system accuracy")
    print("Original GPT-4o: Conservative, likely missed theft")
    print("Enhanced GPT-4o + CLIP: Should detect with computer vision evidence")

if __name__ == "__main__":
    analyze_theft_frame()
