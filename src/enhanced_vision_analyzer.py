"""
enhanced_vision_analyzer.py - Hybrid vision analysis using CLIP, BLIP, and GPT-4o

Combines open-source models for better visual understanding with GPT-4o for reasoning.
CLIP provides object detection and similarity matching.
BLIP provides detailed image captioning.
GPT-4o provides security threat analysis and reasoning.
"""

import torch
import clip
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from openai import OpenAI
from src.config import settings
import warnings
warnings.filterwarnings("ignore")

class EnhancedVisionAnalyzer:
    """Hybrid vision analyzer combining CLIP, BLIP, and GPT-4o."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Enhanced Vision Analyzer on {self.device}")
        
        # Initialize CLIP
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Initialize BLIP
        print("Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        
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
            "peaceful interaction"
        ]
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
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
            top_indices = similarities.topk(5).indices
            top_similarities = similarities.topk(5).values
            
            clip_results = []
            threat_score = 0
            
            for idx, sim in zip(top_indices, top_similarities):
                prompt = self.security_prompts[idx]
                similarity = sim.item()
                
                clip_results.append({
                    'prompt': prompt,
                    'similarity': similarity,
                    'is_threat': any(threat_word in prompt.lower() for threat_word in 
                                   ['theft', 'suspicious', 'concealing', 'reaching', 'hiding', 'nervous'])
                })
                
                if clip_results[-1]['is_threat']:
                    threat_score += similarity
            
            return {
                'clip_analysis': clip_results,
                'threat_score': threat_score.item(),
                'max_threat_similarity': max([r['similarity'] for r in clip_results if r['is_threat']], default=0)
            }
            
        except Exception as e:
            print(f"CLIP analysis failed: {e}")
            return {'clip_analysis': [], 'threat_score': 0, 'max_threat_similarity': 0}
    
    def analyze_with_blip(self, image_path: Path) -> Dict[str, Any]:
        """Analyze image using BLIP for detailed captioning."""
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Generate caption
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Generate conditional caption with security focus
            security_prompt = "a security camera view of"
            inputs = self.blip_processor(image, security_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
                security_caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return {
                'general_caption': caption,
                'security_caption': security_caption,
                'caption_length': len(caption.split())
            }
            
        except Exception as e:
            print(f"BLIP analysis failed: {e}")
            return {'general_caption': '', 'security_caption': '', 'caption_length': 0}
    
    def enhance_gpt4o_context(self, clip_results: Dict[str, Any], blip_results: Dict[str, Any]) -> str:
        """Create enhanced context for GPT-4o based on CLIP and BLIP analysis."""
        context_parts = []
        
        # Add CLIP insights
        if clip_results['clip_analysis']:
            context_parts.append("VISUAL SIMILARITY ANALYSIS:")
            threat_prompts = [r for r in clip_results['clip_analysis'] if r['is_threat']]
            if threat_prompts:
                context_parts.append(f"High similarity to threat patterns: {', '.join([r['prompt'] for r in threat_prompts[:3]])}")
                context_parts.append(f"Maximum threat similarity: {clip_results['max_threat_similarity']:.3f}")
            
            normal_prompts = [r for r in clip_results['clip_analysis'] if not r['is_threat']]
            if normal_prompts:
                context_parts.append(f"Normal behavior indicators: {', '.join([r['prompt'] for r in normal_prompts[:2]])}")
        
        # Add BLIP insights
        if blip_results['general_caption']:
            context_parts.append(f"IMAGE DESCRIPTION: {blip_results['general_caption']}")
        
        if blip_results['security_caption']:
            context_parts.append(f"SECURITY-FOCUSED DESCRIPTION: {blip_results['security_caption']}")
        
        return "\n".join(context_parts)
    
    def analyze_with_enhanced_gpt4o(self, image_path: Path, enhanced_context: str, 
                                   telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze with GPT-4o using enhanced context from CLIP and BLIP."""
        
        enhanced_prompt = f"""
ENHANCED SECURITY ANALYSIS WITH COMPUTER VISION INSIGHTS:

{enhanced_context}

TELEMETRY CONTEXT: {json.dumps(telemetry, indent=2)}

CRITICAL MISSION: You have enhanced visual analysis from CLIP (similarity matching) and BLIP (detailed captioning).
Use these insights to make ACCURATE threat assessments. The computer vision models have already identified
potential threat patterns - your job is to validate and provide detailed security analysis.

If CLIP shows high similarity to threat patterns, treat this as STRONG EVIDENCE of suspicious activity.
If BLIP describes activities that match theft indicators, validate these observations.

FOCUS ON: Theft detection, suspicious behavior, security threats, criminal activity.

Return detailed security analysis in this JSON format:
{{
    "enhanced_threat_assessment": "high/medium/low/none",
    "visual_evidence": ["specific observations from CLIP/BLIP"],
    "behavioral_analysis": "detailed behavior description",
    "theft_probability": "high/medium/low/none",
    "security_recommendations": ["specific actions"],
    "confidence": 0.95,
    "people_detected": 4,
    "suspicious_activities": ["specific activities"],
    "detailed_reasoning": "comprehensive explanation"
}}
"""
        
        try:
            # Encode image
            with open(image_path, "rb") as image_file:
                import base64
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an enhanced security analyst with access to CLIP and BLIP computer vision insights. Use these to make accurate threat assessments."
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": enhanced_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            return {
                'enhanced_analysis': response.choices[0].message.content,
                'model_used': 'gpt-4o-enhanced',
                'context_used': enhanced_context
            }
            
        except Exception as e:
            print(f"Enhanced GPT-4o analysis failed: {e}")
            return {'enhanced_analysis': '', 'model_used': 'failed', 'context_used': enhanced_context}
    
    def comprehensive_analysis(self, image_path: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis using all three models."""
        print(f"Starting comprehensive analysis for {image_path.name}")
        
        # Step 1: CLIP Analysis
        print("Running CLIP analysis...")
        clip_results = self.analyze_with_clip(image_path)
        
        # Step 2: BLIP Analysis  
        print("Running BLIP analysis...")
        blip_results = self.analyze_with_blip(image_path)
        
        # Step 3: Enhanced GPT-4o Analysis
        print("Running enhanced GPT-4o analysis...")
        enhanced_context = self.enhance_gpt4o_context(clip_results, blip_results)
        gpt4o_results = self.analyze_with_enhanced_gpt4o(image_path, enhanced_context, telemetry)
        
        # Step 4: Combine results
        combined_results = {
            'image_path': str(image_path),
            'telemetry': telemetry,
            'clip_analysis': clip_results,
            'blip_analysis': blip_results,
            'gpt4o_enhanced': gpt4o_results,
            'overall_threat_level': self._calculate_overall_threat(clip_results, blip_results, gpt4o_results),
            'analysis_timestamp': str(Path().cwd())
        }
        
        return combined_results
    
    def _calculate_overall_threat(self, clip_results: Dict, blip_results: Dict, gpt4o_results: Dict) -> str:
        """Calculate overall threat level from all analyses."""
        threat_score = 0
        
        # CLIP contribution
        threat_score += clip_results['threat_score'] * 2
        
        # BLIP contribution (check for theft-related words)
        if blip_results['security_caption']:
            theft_words = ['theft', 'steal', 'pocket', 'conceal', 'suspicious', 'hide']
            if any(word in blip_results['security_caption'].lower() for word in theft_words):
                threat_score += 3
        
        # GPT-4o contribution (if available)
        if gpt4o_results['enhanced_analysis']:
            analysis_lower = gpt4o_results['enhanced_analysis'].lower()
            if any(word in analysis_lower for word in ['high', 'theft', 'suspicious', 'threat']):
                threat_score += 4
        
        if threat_score >= 5:
            return 'HIGH'
        elif threat_score >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'

def analyze_frame_enhanced(image_path: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Main function for enhanced frame analysis."""
    analyzer = EnhancedVisionAnalyzer()
    return analyzer.comprehensive_analysis(image_path, telemetry)
