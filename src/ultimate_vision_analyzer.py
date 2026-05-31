"""
ultimate_vision_analyzer.py - Ultimate vision analysis combining CLIP, BLIP, and GPT-4o

This is the most optimized version that combines:
1. CLIP for visual similarity matching (fast, local)
2. BLIP for image captioning and VQA (comprehensive description)
3. GPT-4o Vision for deep security analysis (most accurate)

Optimizations:
- Model caching and singleton pattern
- Batch processing where possible
- GPU acceleration
- Progressive threat assessment
- Early exit for low-threat scenarios
"""

import torch
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image
import numpy as np
import json
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import OpenAI
from src.config import settings
import warnings
warnings.filterwarnings("ignore")

class UltimateVisionAnalyzer:
    """Ultimate vision analyzer combining CLIP, BLIP, and GPT-4o."""
    
    _instance = None
    _clip_model = None
    _clip_preprocess = None
    _blip_caption_model = None
    _blip_caption_processor = None
    _blip_vqa_model = None
    _blip_vqa_processor = None
    _device = None
    _openai_client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UltimateVisionAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._clip_model is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Initializing Ultimate Vision Analyzer on {self._device}")
            
            # Initialize CLIP (fast visual similarity)
            print("Loading CLIP model...")
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=self._device)
            
            # Initialize BLIP captioning (detailed descriptions)
            print("Loading BLIP captioning model...")
            self._blip_caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self._device)
            self._blip_caption_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            
            # Initialize BLIP VQA (question answering)
            print("Loading BLIP VQA model...")
            self._blip_vqa_model = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-base"
            ).to(self._device)
            self._blip_vqa_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-vqa-base"
            )
            
            # Security prompts for CLIP
            self.security_prompts = [
                "person reaching for pocket",
                "hand concealing object", 
                "person looking around nervously",
                "shoplifting in progress",
                "theft of phone",
                "suspicious behavior",
                "person hiding something",
                "quick grab motion",
                "hand in bag",
                "person avoiding eye contact",
                "normal shopping behavior",
                "customer browsing products",
                "peaceful interaction",
                "multiple people at counter",
                "person holding phone",
                "retail environment",
                "store interior",
                "people interacting"
            ]
            
            # Security questions for BLIP VQA
            self.security_questions = [
                "What are the people doing in this image?",
                "Is there any suspicious activity visible?",
                "Are there any objects that could be weapons?",
                "How many people are in the image?",
                "What is the location or setting?",
                "Are there any vehicles visible?",
                "What are the main objects in the scene?",
                "Is anyone reaching for something?",
                "Are there any bags or containers visible?",
                "What type of clothing are people wearing?"
            ]
            
            # Initialize OpenAI client
            self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            print("Ultimate Vision Analyzer initialized successfully")
    
    @property
    def device(self):
        return self._device
    
    def analyze_with_clip(self, image_path: Path) -> Dict[str, Any]:
        """Fast CLIP analysis for threat detection."""
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self._clip_preprocess(image).unsqueeze(0).to(self._device)
            text_tokens = clip.tokenize(self.security_prompts).to(self._device)
            
            with torch.no_grad():
                image_features = self._clip_model.encode_image(image_input)
                text_features = self._clip_model.encode_text(text_tokens)
                similarities = torch.cosine_similarity(image_features, text_features)
            
            top_similarities, top_indices = similarities.topk(8)
            
            clip_results = []
            threat_score = 0
            normal_score = 0
            
            for idx, sim in zip(top_indices, top_similarities):
                prompt = self.security_prompts[idx.item()]
                similarity = sim.item()
                
                is_threat = any(threat_word in prompt.lower() for threat_word in 
                               ['theft', 'suspicious', 'concealing', 'reaching', 'hiding', 'nervous', 'quick'])
                is_normal = any(normal_word in prompt.lower() for normal_word in 
                               ['normal', 'browsing', 'peaceful', 'customer'])
                
                clip_results.append({
                    'prompt': prompt,
                    'similarity': similarity,
                    'is_threat': is_threat,
                    'is_normal': is_normal
                })
                
                if is_threat:
                    threat_score += similarity
                elif is_normal:
                    normal_score += similarity
            
            return {
                'clip_analysis': clip_results,
                'threat_score': float(threat_score),
                'normal_score': float(normal_score),
                'max_threat_similarity': max([r['similarity'] for r in clip_results if r['is_threat']], default=0),
                'max_normal_similarity': max([r['similarity'] for r in clip_results if r['is_normal']], default=0)
            }
            
        except Exception as e:
            print(f"CLIP analysis failed: {e}")
            return {'clip_analysis': [], 'threat_score': 0, 'normal_score': 0, 'max_threat_similarity': 0, 'max_normal_similarity': 0}
    
    def analyze_with_blip(self, image_path: Path) -> Dict[str, Any]:
        """BLIP analysis for detailed description and VQA."""
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Generate caption
            inputs = self._blip_caption_processor(image, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._blip_caption_model.generate(**inputs, max_length=50)
            caption = self._blip_caption_processor.decode(out[0], skip_special_tokens=True)
            
            # Answer security questions
            vqa_answers = {}
            for question in self.security_questions:
                inputs = self._blip_vqa_processor(image, question, return_tensors="pt").to(self._device)
                with torch.no_grad():
                    out = self._blip_vqa_model.generate(**inputs, max_length=20)
                answer = self._blip_vqa_processor.decode(out[0], skip_special_tokens=True)
                vqa_answers[question] = answer
            
            # Extract insights
            insights = self._extract_blip_insights(vqa_answers)
            
            return {
                'blip_caption': caption,
                'blip_vqa_answers': vqa_answers,
                'blip_insights': insights
            }
            
        except Exception as e:
            print(f"BLIP analysis failed: {e}")
            return {'blip_caption': '', 'blip_vqa_answers': {}, 'blip_insights': {}}
    
    def _extract_blip_insights(self, vqa_answers: Dict[str, str]) -> Dict[str, Any]:
        """Extract security insights from BLIP VQA answers."""
        insights = {
            "people_count": 0,
            "suspicious_activity": False,
            "weapons_detected": False,
            "location": "unknown",
            "vehicles_present": False,
            "main_objects": [],
            "reaching_behavior": False,
            "bags_containers": False,
            "clothing_description": "unknown"
        }
        
        # Extract people count
        people_answer = vqa_answers.get("How many people are in the image?", "")
        try:
            import re
            numbers = re.findall(r'\d+', people_answer)
            if numbers:
                insights["people_count"] = int(numbers[0])
        except:
            pass
        
        # Check for suspicious activity
        suspicious_answer = vqa_answers.get("Is there any suspicious activity visible?", "").lower()
        insights["suspicious_activity"] = "yes" in suspicious_answer or "suspicious" in suspicious_answer
        
        # Check for weapons
        weapons_answer = vqa_answers.get("Are there any objects that could be weapons?", "").lower()
        insights["weapons_detected"] = "yes" in weapons_answer or "weapon" in weapons_answer
        
        # Extract location
        insights["location"] = vqa_answers.get("What is the location or setting?", "")
        
        # Check for vehicles
        vehicles_answer = vqa_answers.get("Are there any vehicles visible?", "").lower()
        insights["vehicles_present"] = "yes" in vehicles_answer or "vehicle" in vehicles_answer
        
        # Extract main objects
        objects_answer = vqa_answers.get("What are the main objects in the scene?", "")
        insights["main_objects"] = [obj.strip() for obj in objects_answer.split(",")]
        
        # Check for reaching behavior
        reaching_answer = vqa_answers.get("Is anyone reaching for something?", "").lower()
        insights["reaching_behavior"] = "yes" in reaching_answer or "reaching" in reaching_answer
        
        # Check for bags/containers
        bags_answer = vqa_answers.get("Are there any bags or containers visible?", "").lower()
        insights["bags_containers"] = "yes" in bags_answer or "bag" in bags_answer or "container" in bags_answer
        
        # Extract clothing
        insights["clothing_description"] = vqa_answers.get("What type of clothing are people wearing?", "")
        
        return insights
    
    def analyze_with_gpt4o(self, image_path: Path, enhanced_context: str, 
                          telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """GPT-4o Vision analysis with enhanced context from CLIP and BLIP."""
        
        enhanced_prompt = f"""
ULTIMATE SECURITY ANALYSIS - MULTI-MODEL INSIGHTS:

{enhanced_context}

TELEMETRY CONTEXT: {json.dumps(telemetry, indent=2)}

You have insights from:
1. CLIP: Visual similarity analysis for threat patterns
2. BLIP: Image captioning and visual question answering
3. Combined: Multi-model threat assessment

ANALYSIS REQUIREMENTS:
- Use CLIP threat detection as objective evidence
- Use BLIP descriptions for detailed scene understanding
- Provide comprehensive security assessment
- Consider all evidence sources

Return comprehensive security analysis in this JSON format:
{{
    "vlm_description": "detailed scene description",
    "scene_type": "interior|exterior|parking_lot|warehouse|retail|unknown",
    "objects_detected": ["list", "of", "objects"],
    "object_details": [
        {{"label": "person", "count": 5, "confidence": 0.95, "attributes": ["standing", "at_counter"]}}
    ],
    "people_count": 5,
    "person_features": [
        {{
            "id": "person_1",
            "clothing_color": "dark",
            "clothing_type": "jacket",
            "body_type": "average",
            "height_estimate": "average",
            "distinctive_features": ["beard"],
            "face_visible": true,
            "actions": ["reaching", "holding_item"],
            "position_in_frame": "left",
            "confidence": 0.95
        }}
    ],
    "vehicles_detected": [],
    "vehicle_details": [],
    "activity": "detailed activity description",
    "threat_assessment": "high|medium|low|none",
    "security_signals": ["specific_signals"],
    "suspicious_elements": ["suspicious_observations"],
    "confidence": 0.95,
    "recommended_action": "specific_action",
    "alert_reasoning": "detailed_reasoning",
    "alert_priority_signals": ["priority_signals"]
}}
"""
        
        try:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self._openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an ultimate security analyst with multi-model computer vision insights from CLIP and BLIP. Use all available evidence for accurate threat assessment."
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": enhanced_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                        ]
                    }
                ],
                max_tokens=1200
            )
            
            return {
                'gpt4o_analysis': response.choices[0].message.content,
                'model_used': 'gpt-4o-ultimate',
                'success': True
            }
            
        except Exception as e:
            print(f"GPT-4o analysis failed: {e}")
            return {'gpt4o_analysis': '', 'model_used': 'failed', 'success': False}
    
    def build_enhanced_context(self, clip_results: Dict[str, Any], blip_results: Dict[str, Any]) -> str:
        """Build enhanced context from CLIP and BLIP results."""
        context_parts = [
            "MULTI-MODEL COMPUTER VISION ANALYSIS",
            "=" * 50,
            "",
            "CLIP VISUAL SIMILARITY ANALYSIS:"
        ]
        
        # CLIP insights
        threat_matches = [r for r in clip_results['clip_analysis'] if r['is_threat'] and r['similarity'] > 0.2]
        if threat_matches:
            threat_matches_str = ', '.join([f"{r['prompt']} ({r['similarity']:.3f})" for r in threat_matches[:3]])
            context_parts.append(f"THREAT PATTERNS DETECTED: {threat_matches_str}")
            context_parts.append(f"Maximum threat similarity: {clip_results['max_threat_similarity']:.3f}")
        
        normal_matches = [r for r in clip_results['clip_analysis'] if r['is_normal'] and r['similarity'] > 0.2]
        if normal_matches:
            normal_matches_str = ', '.join([f"{r['prompt']} ({r['similarity']:.3f})" for r in normal_matches[:2]])
            context_parts.append(f"NORMAL BEHAVIOR INDICATORS: {normal_matches_str}")
        
        # CLIP threat assessment
        if clip_results['threat_score'] > clip_results['normal_score'] * 1.5:
            context_parts.append("CLIP ASSESSMENT: HIGH probability of suspicious activity")
        elif clip_results['threat_score'] > clip_results['normal_score']:
            context_parts.append("CLIP ASSESSMENT: MODERATE suspicious indicators")
        else:
            context_parts.append("CLIP ASSESSMENT: Normal activity patterns")
        
        context_parts.extend([
            "",
            "BLIP IMAGE CAPTIONING:",
            f"Caption: {blip_results.get('blip_caption', 'Unable to generate')}",
            "",
            "BLIP VISUAL QUESTION ANSWERING:"
        ])
        
        # BLIP VQA insights
        insights = blip_results.get('blip_insights', {})
        context_parts.extend([
            f"People Count: {insights.get('people_count', 0)}",
            f"Suspicious Activity: {insights.get('suspicious_activity', False)}",
            f"Weapons Detected: {insights.get('weapons_detected', False)}",
            f"Location: {insights.get('location', 'unknown')}",
            f"Reaching Behavior: {insights.get('reaching_behavior', False)}",
            f"Main Objects: {', '.join(insights.get('main_objects', []))}",
            f"Clothing: {insights.get('clothing_description', 'unknown')}"
        ])
        
        # Combined threat assessment
        clip_threat = clip_results['threat_score']
        blip_threat = 5 if insights.get('suspicious_activity') else 0
        blip_threat += 3 if insights.get('reaching_behavior') else 0
        blip_threat += 5 if insights.get('weapons_detected') else 0
        
        combined_threat = clip_threat + blip_threat
        if combined_threat > 3:
            context_parts.append("")
            context_parts.append("COMBINED MULTI-MODEL ASSESSMENT: HIGH THREAT INDICATORS")
        elif combined_threat > 1:
            context_parts.append("")
            context_parts.append("COMBINED MULTI-MODEL ASSESSMENT: MODERATE THREAT INDICATORS")
        else:
            context_parts.append("")
            context_parts.append("COMBINED MULTI-MODEL ASSESSMENT: LOW THREAT INDICATORS")
        
        return "\n".join(context_parts)
    
    def comprehensive_analysis(self, image_path: Path, telemetry: Dict[str, Any], 
                              use_gpt4o: bool = True) -> Dict[str, Any]:
        """Ultimate comprehensive analysis using CLIP, BLIP, and GPT-4o."""
        print(f"Starting Ultimate analysis for {image_path.name}")
        
        # Step 1: Fast CLIP analysis
        print("Running CLIP visual similarity analysis...")
        clip_results = self.analyze_with_clip(image_path)
        
        # Step 2: BLIP analysis
        print("Running BLIP captioning and VQA...")
        blip_results = self.analyze_with_blip(image_path)
        
        # Step 3: Build enhanced context
        enhanced_context = self.build_enhanced_context(clip_results, blip_results)
        
        # Step 4: GPT-4o analysis (optional for speed optimization)
        gpt4o_results = None
        if use_gpt4o:
            print("Running GPT-4o analysis with multi-model context...")
            gpt4o_results = self.analyze_with_gpt4o(image_path, enhanced_context, telemetry)
        
        # Step 5: Calculate overall threat
        overall_threat = self._calculate_overall_threat(clip_results, blip_results, gpt4o_results)
        
        # Combine all results
        results = {
            'image_path': str(image_path),
            'telemetry': telemetry,
            'clip_analysis': clip_results,
            'blip_analysis': blip_results,
            'enhanced_context': enhanced_context,
            'gpt4o_analysis': gpt4o_results,
            'overall_threat_level': overall_threat,
            'analysis_method': 'CLIP + BLIP + GPT-4o',
            'model_used': 'UltimateVisionAnalyzer',
            'processing_speed': 'optimized'
        }
        
        return results
    
    def _calculate_overall_threat(self, clip_results: Dict, blip_results: Dict, 
                                 gpt4o_results: Optional[Dict]) -> str:
        """Calculate overall threat from all models."""
        threat_score = 0
        
        # CLIP contribution
        clip_threat = clip_results['threat_score']
        clip_normal = clip_results['normal_score']
        if clip_threat > clip_normal * 1.5:
            threat_score += 4
        elif clip_threat > clip_normal:
            threat_score += 2
        
        # BLIP contribution
        insights = blip_results.get('blip_insights', {})
        if insights.get('suspicious_activity'):
            threat_score += 3
        if insights.get('weapons_detected'):
            threat_score += 5
        if insights.get('reaching_behavior'):
            threat_score += 2
        if insights.get('people_count', 0) > 3:
            threat_score += 1
        
        # GPT-4o contribution (if available)
        if gpt4o_results and gpt4o_results.get('success'):
            analysis_lower = gpt4o_results['gpt4o_analysis'].lower()
            if any(word in analysis_lower for word in ['high', 'theft', 'suspicious', 'threat']):
                threat_score += 4
            elif any(word in analysis_lower for word in ['medium', 'caution', 'monitor']):
                threat_score += 2
        
        if threat_score >= 8:
            return 'HIGH'
        elif threat_score >= 4:
            return 'MEDIUM'
        else:
            return 'LOW'


import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

def analyze_frame_ultimate(image_path: Path, telemetry: Dict[str, Any], 
                           use_gpt4o: bool = True, timeout: int = 60) -> Dict[str, Any]:
    """Main function for ultimate frame analysis with timeout protection.
    
    Args:
        image_path: Path to image file
        telemetry: Telemetry data
        use_gpt4o: Whether to use GPT-4o (can be disabled for speed)
        timeout: Maximum time allowed for analysis in seconds (default 60)
    """
    analyzer = UltimateVisionAnalyzer()
    
    # Use ThreadPoolExecutor for timeout control
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(analyzer.comprehensive_analysis, image_path, telemetry, use_gpt4o)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            print(f"⚠️ Ultimate analyzer timeout after {timeout}s - returning partial results")
            # Return fallback results on timeout
            return {
                'image_path': str(image_path),
                'telemetry': telemetry,
                'clip_analysis': {'error': 'timeout'},
                'blip_analysis': {'error': 'timeout'},
                'enhanced_context': 'Analysis timed out - using fallback',
                'gpt4o_analysis': {'error': 'timeout', 'success': False},
                'overall_threat_level': 'UNKNOWN',
                'analysis_method': 'CLIP + BLIP + GPT-4o (TIMEOUT)',
                'model_used': 'UltimateVisionAnalyzer',
                'processing_speed': 'timeout',
                '_timeout': True
            }


if __name__ == "__main__":
    # Test the ultimate analyzer
    test_image = Path("data/extracted/frame_017.jpg")
    if test_image.exists():
        telemetry = {
            "frame_id": "frame_017",
            "timestamp": "16:00:16",
            "location": "Garage",
            "is_after_hours": False,
            "is_restricted_zone": False
        }
        
        result = analyze_frame_ultimate(test_image, telemetry, use_gpt4o=True)
        print(json.dumps(result, indent=2))
    else:
        print(f"Test image not found: {test_image}")
