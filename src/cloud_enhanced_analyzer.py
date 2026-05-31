"""
cloud_enhanced_analyzer.py - Cloud-based enhanced analysis using Hugging Face Inference API

This module uses Hugging Face's free Inference API to run CLIP and BLIP models in the cloud,
while keeping GPT-4o local. This avoids local model loading delays and timeouts.

Benefits:
- No local GPU required
- Fast model inference (cloud GPUs)
- Free tier available (30k requests/month)
- Scalable and reliable

Requirements:
- Hugging Face API token (free at huggingface.co/settings/tokens)
"""

import requests
import base64
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import OpenAI
from src.config import settings

# Hugging Face API configuration
HF_API_TOKEN = settings.HF_API_TOKEN
HF_API_URL = "https://api-inference.huggingface.co/models"

# Model endpoints
CLIP_MODEL = "openai/clip-vit-base-patch32"
BLIP_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
BLIP_VQA_MODEL = "Salesforce/blip-vqa-base"


class CloudEnhancedAnalyzer:
    """Cloud-based analyzer using Hugging Face Inference API + local GPT-4o."""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.hf_headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}"
        } if HF_API_TOKEN else {}
        self.request_count = 0
        
    def analyze_frame_cloud(self, image_path: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze frame using cloud CLIP + BLIP + local GPT-4o.
        
        Args:
            image_path: Path to frame image
            telemetry: Frame telemetry data
            
        Returns:
            Combined analysis results
        """
        start_time = time.time()
        print(f"[CLOUD] Starting cloud-enhanced analysis for {image_path.name}")
        
        # Load and encode image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode()
        
        results = {
            'image_path': str(image_path),
            'telemetry': telemetry,
            'clip_analysis': None,
            'blip_analysis': None,
            'gpt4o_analysis': None,
            'overall_threat_level': 'UNKNOWN',
            'analysis_method': 'Cloud CLIP + BLIP + GPT-4o',
            'model_used': 'CloudEnhancedAnalyzer',
            'processing_time_ms': 0
        }
        
        # Step 1: Cloud CLIP Analysis (parallel with BLIP)
        print("[CLOUD] Calling Hugging Face CLIP API...")
        try:
            clip_results = self._analyze_with_clip_cloud(image_b64)
            results['clip_analysis'] = clip_results
            print(f"[OK] CLIP analysis complete - Threat score: {clip_results.get('threat_score', 0)}")
        except Exception as e:
            print(f"[WARNING] CLIP API failed: {e}")
            results['clip_analysis'] = {'error': str(e), 'threat_score': 0}
        
        # Step 2: Cloud BLIP Analysis (parallel with CLIP)
        print("[CLOUD] Calling Hugging Face BLIP API...")
        try:
            blip_results = self._analyze_with_blip_cloud(image_b64, image_bytes)
            results['blip_analysis'] = blip_results
            print(f"[OK] BLIP analysis complete - Caption: {blip_results.get('caption', 'N/A')[:50]}...")
        except Exception as e:
            print(f"[WARNING] BLIP API failed: {e}")
            results['blip_analysis'] = {'error': str(e), 'caption': ''}
        
        # Step 3: Build enhanced context from CLIP + BLIP
        enhanced_context = self._build_enhanced_context(
            results['clip_analysis'], 
            results['blip_analysis']
        )
        results['enhanced_context'] = enhanced_context
        
        # Step 4: TWO-STAGE GPT-4o Analysis
        # Stage 1: Initial analysis with neutral prompt
        print("[AI] Stage 1: Initial GPT-4o analysis...")
        gpt4o_results_stage1 = None
        try:
            gpt4o_results_stage1 = self._analyze_with_gpt4o(image_path, enhanced_context, telemetry, stage=1)
            results['gpt4o_analysis_stage1'] = gpt4o_results_stage1
            
            # Check if suspicious keywords detected in stage 1
            suspicious_keywords = [
                "reaching", "concealing", "hiding", "grabbing", "palming", 
                "pocketing", "loitering", "lurking", "sneaking", "crouching",
                "stuffing", "taking", "unattended", "weapon", "fire", "assault"
            ]
            
            stage1_text = ""
            if gpt4o_results_stage1.get('parsed_data'):
                parsed = gpt4o_results_stage1['parsed_data']
                stage1_text = str(parsed.get('vlm_description', '')) + " " + str(parsed.get('person_features', ''))
            elif gpt4o_results_stage1.get('gpt4o_analysis'):
                stage1_text = str(gpt4o_results_stage1['gpt4o_analysis'])
            
            stage1_lower = stage1_text.lower()
            suspicious_detected = [kw for kw in suspicious_keywords if kw in stage1_lower]
            
            if suspicious_detected:
                print(f"[ALERT] Suspicious keywords detected in Stage 1: {suspicious_detected}")
                print("[AI] Stage 2: Running security-focused analysis...")
                
                # Stage 2: Security-focused analysis
                enhanced_context_stage2 = enhanced_context + f"\n\n[SECURITY ALERT] Stage 1 detected suspicious behaviors: {', '.join(suspicious_detected)}. Analyze specifically for theft/concealment behaviors."
                
                gpt4o_results = self._analyze_with_gpt4o(image_path, enhanced_context_stage2, telemetry, stage=2)
                results['gpt4o_analysis'] = gpt4o_results
                results['suspicious_keywords_detected'] = suspicious_detected
                print("[OK] Two-stage analysis complete with security focus")
            else:
                # No suspicious activity, use stage 1 results
                results['gpt4o_analysis'] = gpt4o_results_stage1
                print("[OK] Stage 1 sufficient - no suspicious activity detected")
                
        except Exception as e:
            print(f"[ERROR] GPT-4o analysis failed: {e}")
            results['gpt4o_analysis'] = {'success': False, 'error': str(e)}
        
        # Step 5: Calculate overall threat
        results['overall_threat_level'] = self._calculate_overall_threat(
            results['clip_analysis'],
            results['blip_analysis'],
            results['gpt4o_analysis']
        )
        
        results['processing_time_ms'] = int((time.time() - start_time) * 1000)
        print(f"[OK] Cloud-enhanced analysis complete in {results['processing_time_ms']}ms")
        
        return results
    
    def _analyze_with_clip_cloud(self, image_b64: str) -> Dict[str, Any]:
        """Analyze image using Hugging Face CLIP API."""
        if not HF_API_TOKEN:
            return {'error': 'No HF_API_TOKEN set', 'threat_score': 0, 'matches': []}
        
        # Comprehensive security prompts for CLIP - organized by threat category
        security_prompts = [
            # THEFT & CONCEALMENT BEHAVIORS
            "person reaching towards shelf or items",
            "hand concealing object or hiding something",
            "person putting item in pocket or bag",
            "person grabbing merchandise quickly",
            "person palming small items",
            "person stuffing items into clothing",
            
            # SUSPICIOUS BEHAVIORS
            "person looking around nervously or checking for cameras",
            "person loitering without shopping purpose",
            "person crouching or hiding behind shelves",
            "person waiting in corner without activity",
            "person sneaking or tiptoeing",
            
            # MOVEMENT & FLIGHT BEHAVIORS
            "person running in store",
            "person moving quickly or rushing",
            "person exiting rapidly without paying",
            "person looking back while leaving",
            
            # UNAUTHORIZED ACCESS
            "person behind counter or in restricted area",
            "person entering staff-only zone",
            "person climbing or jumping fence",
            "person forcing door or lock",
            
            # ENVIRONMENTAL THREATS
            "unattended bag or package left behind",
            "fire or smoke visible",
            "weapon or dangerous object visible",
            "crowd gathering or confrontation",
            "vehicle in pedestrian area",
            
            # NORMAL BASELINE (for contrast)
            "person shopping normally with cart",
            "customer examining products",
            "staff member working",
            "peaceful normal scene"
        ]
        
        # Threat category mapping for better classification
        threat_categories = {
            "theft_concealment": [
                "person reaching towards shelf or items",
                "hand concealing object or hiding something",
                "person putting item in pocket or bag",
                "person grabbing merchandise quickly",
                "person palming small items",
                "person stuffing items into clothing"
            ],
            "suspicious_behavior": [
                "person looking around nervously or checking for cameras",
                "person loitering without shopping purpose",
                "person crouching or hiding behind shelves",
                "person waiting in corner without activity",
                "person sneaking or tiptoeing"
            ],
            "flight_behavior": [
                "person running in store",
                "person moving quickly or rushing",
                "person exiting rapidly without paying",
                "person looking back while leaving"
            ],
            "unauthorized_access": [
                "person behind counter or in restricted area",
                "person entering staff-only zone",
                "person climbing or jumping fence",
                "person forcing door or lock"
            ],
            "environmental_threat": [
                "unattended bag or package left behind",
                "fire or smoke visible",
                "weapon or dangerous object visible",
                "crowd gathering or confrontation",
                "vehicle in pedestrian area"
            ]
        }
        
        try:
            # Call Hugging Face CLIP API
            response = requests.post(
                f"{HF_API_URL}/{CLIP_MODEL}",
                headers=self.hf_headers,
                json={
                    "inputs": {
                        "image": image_b64,
                        "candidate_labels": security_prompts
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Calculate threat scores by category
                category_scores = {cat: 0 for cat in threat_categories.keys()}
                all_matches = []
                suspicious_matches = []
                
                if isinstance(result, list):
                    for item in result:
                        label = item.get('label', '')
                        score = item.get('score', 0)
                        match_data = {'label': label, 'score': score}
                        all_matches.append(match_data)
                        
                        # Check if suspicious (score > 0.25 for broader detection)
                        is_suspicious = score > 0.25 and label not in [
                            "person shopping normally with cart",
                            "customer examining products",
                            "staff member working",
                            "peaceful normal scene"
                        ]
                        
                        if is_suspicious:
                            suspicious_matches.append(match_data)
                            
                            # Add to category score
                            for cat, labels in threat_categories.items():
                                if label in labels:
                                    category_scores[cat] += score * 100
                
                # Overall threat score
                total_threat = sum(category_scores.values())
                
                # Determine primary threat category
                primary_category = max(category_scores, key=category_scores.get) if category_scores else None
                
                return {
                    'threat_score': min(total_threat, 100),
                    'threat_categories': category_scores,
                    'primary_threat_category': primary_category,
                    'matches': all_matches,
                    'suspicious_matches': suspicious_matches,
                    'has_reaching': any('reaching' in m['label'] for m in suspicious_matches),
                    'has_concealing': any('concealing' in m['label'] or 'hiding' in m['label'] for m in suspicious_matches),
                    'has_loitering': any('loitering' in m['label'] or 'waiting' in m['label'] for m in suspicious_matches),
                    'top_matches': sorted(suspicious_matches, key=lambda x: x['score'], reverse=True)[:3]
                }
            else:
                return {
                    'error': f'API returned {response.status_code}',
                    'threat_score': 0,
                    'matches': []
                }
                
        except Exception as e:
            return {'error': str(e), 'threat_score': 0, 'matches': []}
    
    def _analyze_with_blip_cloud(self, image_b64: str, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze image using Hugging Face BLIP API."""
        if not HF_API_TOKEN:
            return {'error': 'No HF_API_TOKEN set', 'caption': '', 'insights': {}}
        
        try:
            # Get image caption
            caption_response = requests.post(
                f"{HF_API_URL}/{BLIP_CAPTION_MODEL}",
                headers=self.hf_headers,
                data=image_bytes,  # BLIP models usually take raw bytes
                timeout=30
            )
            
            caption = ""
            if caption_response.status_code == 200:
                caption_result = caption_response.json()
                if isinstance(caption_result, list) and len(caption_result) > 0:
                    caption = caption_result[0].get('generated_text', '')
                elif isinstance(caption_result, dict):
                    caption = caption_result.get('generated_text', '')
            
            # Extract insights from caption
            insights = self._extract_insights_from_caption(caption)
            
            return {
                'caption': caption,
                'insights': insights,
                'security_keywords_found': insights.get('security_keywords', [])
            }
            
        except Exception as e:
            return {'error': str(e), 'caption': '', 'insights': {}}
    
    def _extract_insights_from_caption(self, caption: str) -> Dict[str, Any]:
        """Extract security insights from BLIP caption."""
        caption_lower = caption.lower()
        
        # Security keywords
        security_keywords = [
            'stealing', 'theft', 'suspicious', 'unattended', 'crowd',
            'running', 'hiding', 'concealing', 'reaching', 'loitering'
        ]
        
        found_keywords = [kw for kw in security_keywords if kw in caption_lower]
        
        # Count people mentioned
        people_count = 0
        if 'person' in caption_lower or 'people' in caption_lower or 'man' in caption_lower or 'woman' in caption_lower:
            # Try to extract number
            import re
            numbers = re.findall(r'(\d+)\s*(?:person|people|men|women|man|woman)', caption_lower)
            if numbers:
                people_count = sum(int(n) for n in numbers)
            else:
                # Count occurrences of person-related words
                people_count = caption_lower.count('person') + caption_lower.count('man') + caption_lower.count('woman')
        
        return {
            'main_objects': [],  # Would need object detection
            'activity': caption,
            'location': 'unknown',
            'security_keywords': found_keywords,
            'people_count': people_count,
            'threat_indicators': len(found_keywords)
        }
    
    def _build_enhanced_context(self, clip_results: Dict, blip_results: Dict) -> str:
        """Build enhanced context from CLIP and BLIP results."""
        context_parts = []
        
        # Add CLIP findings
        if clip_results and not clip_results.get('error'):
            context_parts.append(f"CLIP Visual Analysis:")
            context_parts.append(f"- Threat Score: {clip_results.get('threat_score', 0):.1f}/100")
            suspicious = clip_results.get('suspicious_matches', [])
            if suspicious:
                context_parts.append("- Visual Matches:")
                for match in suspicious[:3]:  # Top 3 matches
                    context_parts.append(f"  * {match['label']}: {match['score']:.2f}")
        
        # Add BLIP caption
        if blip_results and not blip_results.get('error'):
            caption = blip_results.get('caption', '')
            if caption:
                context_parts.append(f"\nBLIP Image Caption: {caption}")
            
            insights = blip_results.get('insights', {})
            keywords = insights.get('security_keywords', [])
            if keywords:
                context_parts.append(f"- Security Keywords: {', '.join(keywords)}")
        
        return "\n".join(context_parts) if context_parts else "No enhanced context available."
    
    def _analyze_with_gpt4o(self, image_path: Path, enhanced_context: str, 
                           telemetry: Dict[str, Any], stage: int = 1) -> Dict[str, Any]:
        """Analyze with GPT-4o using enhanced context.
        
        Stage 1: Neutral observation to detect suspicious keywords
        Stage 2: Security-focused analysis for confirmed threats
        """
        try:
            # Encode image
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Stage-specific prompts
            if stage == 1:
                # STAGE 1: Neutral observation (to catch suspicious keywords)
                prompt = f"""You are observing a surveillance camera frame. Describe what you see objectively and factually.

Focus on:
1. Scene type and setting
2. Number of people and what they are doing
3. Any hand movements, reaching, or object interactions
4. People's positions and actions
5. Any unusual or noteworthy behaviors

Context from other analysis tools:
{enhanced_context}

Frame metadata:
- Frame ID: {telemetry.get('frame_id', 'unknown')}
- Time: {telemetry.get('timestamp', 'unknown')}
- Location: {telemetry.get('location', 'unknown')}

Provide detailed observation in JSON:
{{
  "vlm_description": "Detailed factual description",
  "scene_type": "retail|warehouse|parking|interior|exterior",
  "people_count": 0,
  "person_features": ["Person_1: [exactly what they are doing]", "Person_2: [actions]"],
  "objects_detected": ["list of visible objects"],
  "activity": "summary of overall activity",
  "security_signals": [],
  "suspicious_elements": [],
  "threat_level": "CLEAR",
  "threat_type": "clear",
  "reasoning": "Initial observation - no security assessment",
  "recommended_action": "Continue monitoring",
  "confidence": 0.9
}}

Be factual and descriptive. Note any reaching, concealing, or unusual movements."""
            else:
                # STAGE 2: Security-focused analysis
                prompt = f"""You are a SECURITY ANALYST reviewing surveillance footage for THREATS.

🚨 SECURITY ASSESSMENT REQUIRED 🚨

Previous observation flagged these behaviors: {enhanced_context}

THREAT CLASSIFICATION:
- CRITICAL: Weapon, fire, assault, forced entry, active theft
- HIGH: Confirmed trespassing, unauthorized access, repeated suspicious behavior
- MEDIUM: Suspicious but unconfirmed (reaching, loitering, concealing attempts)
- LOW: Minor concerns, rule violations
- CLEAR: No threat detected

IMPORTANT: The following behaviors are THEFT INDICATORS - flag as MEDIUM or HIGH:
- Reaching towards items without staff present
- Hand/arm movements toward pockets/bags while near merchandise
- Looking around nervously while handling items
- Crouching or hiding behind displays
- Rapid movements with items
- Putting items in clothing or bags

Context:
- After hours: {telemetry.get('is_after_hours', False)}
- Location: {telemetry.get('location', 'unknown')}
- Time: {telemetry.get('timestamp', 'unknown')}

Provide security assessment in JSON:
{{
  "threat_level": "CRITICAL|HIGH|MEDIUM|LOW|CLEAR",
  "threat_type": "theft_behavior|loitering|trespassing|suspicious_behavior|clear",
  "vlm_description": "Security-focused description",
  "scene_type": "retail|warehouse|parking|interior|exterior",
  "people_count": 0,
  "person_features": ["Person_1: [security-relevant actions]"],
  "security_signals": ["theft_indicator_1", "theft_indicator_2"],
  "suspicious_elements": ["specific concerns"],
  "reasoning": "Security assessment: why this is/isn't a threat",
  "recommended_action": "Specific security response",
  "confidence": 0.85
}}

Focus on THEFT and SECURITY THREATS. Do NOT dismiss reaching/concealing as 'normal shopping'."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.2
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON to extract threat fields
            import json
            try:
                # Extract JSON from response
                json_start = content.find('{')
                json_end = content.rfind('}')
                if json_start != -1 and json_end != -1:
                    json_str = content[json_start:json_end+1]
                    parsed = json.loads(json_str)
                    
                    return {
                        'success': True,
                        'gpt4o_analysis': content,
                        'model': 'gpt-4o',
                        'threat_level': parsed.get('threat_level', 'UNKNOWN'),
                        'threat_type': parsed.get('threat_type', 'unknown'),
                        'confidence': parsed.get('confidence', 0.0),
                        'parsed_data': parsed
                    }
            except Exception as e:
                print(f"[WARNING] Could not parse GPT-4o JSON: {e}")
            
            return {
                'success': True,
                'gpt4o_analysis': content,
                'model': 'gpt-4o',
                'threat_level': 'UNKNOWN',
                'threat_type': 'unknown'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'gpt4o_analysis': None
            }
    
    def _calculate_overall_threat(self, clip_results: Dict, blip_results: Dict, 
                                 gpt4o_results: Dict) -> str:
        """Calculate overall threat level from all analyses."""
        threat_scores = []
        
        # CLIP threat score
        if clip_results and not clip_results.get('error'):
            clip_score = clip_results.get('threat_score', 0)
            threat_scores.append(clip_score)
        
        # BLIP threat indicators
        if blip_results and not blip_results.get('error'):
            insights = blip_results.get('insights', {})
            keywords = insights.get('security_keywords', [])
            if keywords:
                threat_scores.append(len(keywords) * 15)  # 15 points per keyword
        
        # Parse GPT-4o security assessment if available
        if gpt4o_results and gpt4o_results.get('success'):
            content = gpt4o_results.get('gpt4o_analysis', '')
            if 'HIGH' in content.upper() or 'CRITICAL' in content.upper():
                threat_scores.append(80)
            elif 'MEDIUM' in content.upper():
                threat_scores.append(50)
            elif 'LOW' in content.upper():
                threat_scores.append(20)
        
        if not threat_scores:
            return 'UNKNOWN'
        
        avg_score = sum(threat_scores) / len(threat_scores)
        
        if avg_score >= 70:
            return 'HIGH'
        elif avg_score >= 40:
            return 'MEDIUM'
        else:
            return 'LOW'


def analyze_frame_cloud(image_path: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for cloud-enhanced analysis."""
    analyzer = CloudEnhancedAnalyzer()
    return analyzer.analyze_frame_cloud(image_path, telemetry)


# Simple test
if __name__ == "__main__":
    # Test with existing frame
    test_image = Path("data/extracted/frame_017.jpg")
    if test_image.exists():
        print("Testing cloud-enhanced analyzer...")
        print("Make sure to set HF_API_TOKEN environment variable!")
        print("Get free token at: https://huggingface.co/settings/tokens")
        print()
        
        telemetry = {
            "frame_id": "frame_017",
            "timestamp": "16:00:16",
            "location": "Garage",
            "is_after_hours": False,
            "is_restricted_zone": False
        }
        
        result = analyze_frame_cloud(test_image, telemetry)
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Test image not found: {test_image}")
