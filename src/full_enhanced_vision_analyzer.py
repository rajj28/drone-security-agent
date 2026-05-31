"""
full_enhanced_vision_analyzer.py - Complete enhanced vision analysis using CLIP and GPT-4o

Combines CLIP for visual similarity matching with GPT-4o for enhanced security analysis.
This version uses CLIP (working) and provides BLIP alternative for future implementation.
"""

import torch
import clip
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import base64
from openai import OpenAI
from src.config import settings
import warnings
warnings.filterwarnings("ignore")

class FullEnhancedVisionAnalyzer:
    """Complete enhanced vision analyzer using CLIP and GPT-4o."""
    
    _instance = None
    _clip_model = None
    _clip_preprocess = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FullEnhancedVisionAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._clip_model is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Initializing Full Enhanced Vision Analyzer on {self._device}")
            
            # Initialize CLIP (only once)
            print("Loading CLIP model...")
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=self._device)
            print("CLIP model loaded successfully")
            
            # Security-specific text prompts for CLIP
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
            
            # Initialize OpenAI client
            self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            print("Full Enhanced Vision Analyzer initialized successfully")
    
    @property
    def device(self):
        return self._device
    
    @property
    def clip_model(self):
        return self._clip_model
    
    @property
    def clip_preprocess(self):
        return self._clip_preprocess
    
        
    def analyze_with_clip(self, image_path: Path) -> Dict[str, Any]:
        """Analyze image using CLIP for security-related object detection."""
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize security prompts
            text_tokens = clip.tokenize(self.security_prompts).to(self.device)
            
            # Get features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Calculate similarities
                similarities = torch.cosine_similarity(image_features, text_features)
                
            # Get top matches
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
    
    def generate_basic_image_description(self, image_path: Path) -> str:
        """Generate basic image description using CLIP similarity."""
        try:
            # Basic descriptions for similarity matching
            basic_descriptions = [
                "people standing in a shop",
                "people at a counter",
                "retail store environment",
                "multiple people interacting",
                "person holding objects",
                "store interior with customers",
                "people near display case",
                "counter transaction in progress"
            ]
            
            image = Image.open(image_path).convert("RGB")
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_tokens = clip.tokenize(basic_descriptions).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)
                similarities = torch.cosine_similarity(image_features, text_features)
                
            best_match_idx = similarities.argmax().item()
            return basic_descriptions[best_match_idx]
            
        except Exception as e:
            print(f"Basic description generation failed: {e}")
            return "people in indoor environment"
    
    def enhance_gpt4o_context(self, clip_results: Dict[str, Any], basic_description: str) -> str:
        """Create enhanced context for GPT-4o based on CLIP analysis."""
        context_parts = []
        
        # Add CLIP insights
        if clip_results['clip_analysis']:
            context_parts.append("CLIP VISUAL SIMILARITY ANALYSIS:")
            
            threat_matches = [r for r in clip_results['clip_analysis'] if r['is_threat'] and r['similarity'] > 0.2]
            if threat_matches:
                threat_matches_str = ', '.join([f"{r['prompt']} ({r['similarity']:.3f})" for r in threat_matches[:3]])
                context_parts.append(f"THREAT PATTERNS DETECTED: {threat_matches_str}")
                context_parts.append(f"Maximum threat similarity: {clip_results['max_threat_similarity']:.3f}")
            
            normal_matches = [r for r in clip_results['clip_analysis'] if r['is_normal'] and r['similarity'] > 0.2]
            if normal_matches:
                normal_matches_str = ', '.join([f"{r['prompt']} ({r['similarity']:.3f})" for r in normal_matches[:2]])
                context_parts.append(f"NORMAL BEHAVIOR INDICATORS: {normal_matches_str}")
                context_parts.append(f"Maximum normal similarity: {clip_results['max_normal_similarity']:.3f}")
        
        # Add basic description
        context_parts.append(f"BASIC SCENE DESCRIPTION: {basic_description}")
        
        # Add threat assessment based on scores
        threat_score = clip_results['threat_score']
        normal_score = clip_results['normal_score']
        
        if threat_score > normal_score * 1.5:
            context_parts.append("CLIP ASSESSMENT: High probability of suspicious activity")
        elif threat_score > normal_score:
            context_parts.append("CLIP ASSESSMENT: Moderate suspicious indicators")
        else:
            context_parts.append("CLIP ASSESSMENT: Normal activity patterns")
        
        return "\n".join(context_parts)
    
    def analyze_with_enhanced_gpt4o(self, image_path: Path, enhanced_context: str, 
                                   telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze with GPT-4o using enhanced context from CLIP."""
        
        enhanced_prompt = f"""
ENHANCED SECURITY ANALYSIS WITH CLIP COMPUTER VISION:

{enhanced_context}

TELEMETRY CONTEXT: {json.dumps(telemetry, indent=2)}

CRITICAL MISSION: You have CLIP computer vision insights providing objective similarity analysis.
CLIP has identified visual patterns and compared them to known threat and normal behaviors.

ANALYSIS REQUIREMENTS:
1. Use CLIP threat detection results as strong evidence
2. If CLIP shows high threat similarity, treat as credible threat indicator
3. Consider both threat and normal behavior patterns identified by CLIP
4. Provide detailed security assessment based on combined evidence

SECURITY FOCUS:
- Theft detection and shoplifting patterns
- Suspicious behavior identification
- Multiple person coordination analysis
- Retail environment security assessment

Return comprehensive security analysis in this JSON format:
{{
    "enhanced_threat_assessment": "high/medium/low/none",
    "clip_evidence": {{
        "threat_patterns_found": ["specific CLIP matches"],
        "normal_patterns_found": ["specific CLIP matches"],
        "threat_confidence": 0.95
    }},
    "behavioral_analysis": "detailed behavior description using CLIP insights",
    "theft_probability": "high/medium/low/none",
    "security_recommendations": ["specific actions based on CLIP evidence"],
    "confidence": 0.95,
    "people_detected": 4,
    "suspicious_activities": ["specific activities supported by CLIP"],
    "detailed_reasoning": "comprehensive explanation incorporating CLIP analysis"
}}
"""
        
        try:
            # Encode image
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an enhanced security analyst with CLIP computer vision insights. Use CLIP similarity analysis to make accurate threat assessments."
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
                'enhanced_analysis': response.choices[0].message.content,
                'model_used': 'gpt-4o-clip-enhanced',
                'context_used': enhanced_context,
                'success': True
            }
            
        except Exception as e:
            print(f"Enhanced GPT-4o analysis failed: {e}")
            return {'enhanced_analysis': '', 'model_used': 'failed', 'context_used': enhanced_context, 'success': False}
    
    def comprehensive_analysis(self, image_path: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis using CLIP and GPT-4o."""
        print(f"Starting comprehensive CLIP-enhanced analysis for {image_path.name}")
        
        # Step 1: CLIP Analysis
        print("Running CLIP visual similarity analysis...")
        clip_results = self.analyze_with_clip(image_path)
        
        # Step 2: Basic Description Generation
        print("Generating basic scene description...")
        basic_description = self.generate_basic_image_description(image_path)
        
        # Step 3: Enhanced GPT-4o Analysis
        print("Running CLIP-enhanced GPT-4o analysis...")
        enhanced_context = self.enhance_gpt4o_context(clip_results, basic_description)
        gpt4o_results = self.analyze_with_enhanced_gpt4o(image_path, enhanced_context, telemetry)
        
        # Step 4: Combine results
        combined_results = {
            'image_path': str(image_path),
            'telemetry': telemetry,
            'clip_analysis': clip_results,
            'basic_description': basic_description,
            'gpt4o_enhanced': gpt4o_results,
            'overall_threat_level': self._calculate_overall_threat(clip_results, gpt4o_results),
            'analysis_timestamp': str(Path().cwd()),
            'enhancement_method': 'CLIP + GPT-4o'
        }
        
        return combined_results
    
    def _calculate_overall_threat(self, clip_results: Dict, gpt4o_results: Dict) -> str:
        """Calculate overall threat level from CLIP and GPT-4o analyses."""
        threat_score = 0
        
        # CLIP contribution
        clip_threat = clip_results['threat_score']
        clip_normal = clip_results['normal_score']
        
        if clip_threat > clip_normal * 1.5:
            threat_score += 4
        elif clip_threat > clip_normal:
            threat_score += 2
        
        # GPT-4o contribution (if available)
        if gpt4o_results.get('success') and gpt4o_results.get('enhanced_analysis'):
            analysis_lower = gpt4o_results['enhanced_analysis'].lower()
            if any(word in analysis_lower for word in ['high', 'theft', 'suspicious', 'threat']):
                threat_score += 4
            elif any(word in analysis_lower for word in ['medium', 'caution', 'monitor']):
                threat_score += 2
        
        if threat_score >= 6:
            return 'HIGH'
        elif threat_score >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'

def analyze_frame_full_enhanced(image_path: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Main function for full enhanced frame analysis using CLIP."""
    analyzer = FullEnhancedVisionAnalyzer()
    return analyzer.comprehensive_analysis(image_path, telemetry)
