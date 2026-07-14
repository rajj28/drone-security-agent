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
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import settings
from src.gemini_client import generate_vision
from src.api_retry import quota_exhausted, is_quota_exhausted_error, call_with_retry

# Unified context (JSON + structured timeline + optional Mongo)
try:
    from src.unified_context import get_unified_context, get_vlm_prompt_context
    CONTEXT_AVAILABLE = True
except ImportError:
    CONTEXT_AVAILABLE = False
    print("[WARNING] Unified context not available")

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
        self.hf_headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}"
        } if HF_API_TOKEN else {}
        self.request_count = 0
        self.skip_hf = os.environ.get("SKIP_HF_APIS", "").lower() in ("1", "true", "yes")
        self.single_gpt_stage = os.environ.get("GPT_SINGLE_STAGE", "true").lower() in (
            "1",
            "true",
            "yes",
        )

    def _hf_post(self, url: str, label: str, **kwargs) -> requests.Response:
        """HF inference with backoff (429/503/409)."""

        def _do() -> requests.Response:
            self.request_count += 1
            return requests.post(url, headers=self.hf_headers, timeout=60, **kwargs)

        return call_with_retry(_do, label=label)
        
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
        
        telemetry = dict(telemetry)
        frame_id = telemetry.get('frame_id', 'unknown')

        frame_history = ""
        if CONTEXT_AVAILABLE:
            try:
                unified = get_unified_context()
                telemetry = unified.ensure_telemetry_session_id(telemetry)
                frame_history = get_vlm_prompt_context(frame_id, window=5)
                print(f"[CLOUD] Session {telemetry.get('session_id')} | context loaded ({len(frame_history)} chars)")
                if frame_history:
                    print(f"[CLOUD] Context preview: {frame_history[:200]}...")
            except Exception as e:
                print(f"[WARNING] Failed to load unified context: {e}")
        
        # Load and encode image (with optional robust preprocessing)
        from src.frame_preprocessor import (
            assess_image_quality,
            preprocess_frame_image,
            preprocessing_enabled,
            quality_prompt_hints,
        )

        if preprocessing_enabled():
            image_bytes, image_quality = preprocess_frame_image(image_path)
        else:
            image_quality = assess_image_quality(image_path)
            image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode()
        quality_hints = quality_prompt_hints(image_quality)
        
        results = {
            'image_path': str(image_path),
            'telemetry': telemetry,
            'clip_analysis': None,
            'blip_analysis': None,
            'gpt4o_analysis': None,
            'overall_threat_level': 'UNKNOWN',
            'analysis_method': 'Cloud CLIP + BLIP + Gemini',
            'model_used': 'CloudEnhancedAnalyzer',
            'processing_time_ms': 0,
            'session_context': frame_history,
            'image_quality': image_quality,
        }
        
        if self.skip_hf or quota_exhausted():
            print("[CLOUD] Skipping HF CLIP/BLIP (SKIP_HF_APIS or quota guard)")
            results["clip_analysis"] = {"skipped": True, "threat_score": 0, "matches": []}
            results["blip_analysis"] = {"skipped": True, "caption": "", "insights": {}}
        else:
            print("[CLOUD] Calling Hugging Face CLIP API...")
            try:
                clip_results = self._analyze_with_clip_cloud(image_b64)
                results["clip_analysis"] = clip_results
                print(f"[OK] CLIP complete - threat score: {clip_results.get('threat_score', 0)}")
            except Exception as e:
                print(f"[WARNING] CLIP API failed: {e}")
                results["clip_analysis"] = {"error": str(e), "threat_score": 0}

            print("[CLOUD] Calling Hugging Face BLIP API...")
            try:
                blip_results = self._analyze_with_blip_cloud(image_b64, image_bytes)
                results["blip_analysis"] = blip_results
                print(f"[OK] BLIP complete - caption: {blip_results.get('caption', 'N/A')[:50]}...")
            except Exception as e:
                print(f"[WARNING] BLIP API failed: {e}")
                results["blip_analysis"] = {"error": str(e), "caption": ""}
        
        # Step 3: Build enhanced context from CLIP + BLIP
        enhanced_context = self._build_enhanced_context(
            results['clip_analysis'], 
            results['blip_analysis']
        )
        if quality_hints:
            enhanced_context += quality_hints
        results['enhanced_context'] = enhanced_context
        results['image_quality'] = image_quality
        
        # Step 4: TWO-STAGE Gemini VLM analysis
        print("[AI] Stage 1: Initial Gemini analysis...")
        gpt4o_results_stage1 = None
        try:
            gpt4o_results_stage1 = self._analyze_with_gemini(
                image_bytes, enhanced_context, telemetry, stage=1, frame_history=frame_history
            )
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
            
            if suspicious_detected and not self.single_gpt_stage:
                print(f"[ALERT] Suspicious keywords in Stage 1: {suspicious_detected}")
                print("[AI] Stage 2: security-focused analysis...")
                enhanced_context_stage2 = (
                    enhanced_context
                    + f"\n\n[SECURITY ALERT] Stage 1: {', '.join(suspicious_detected)}."
                )
                gpt4o_results = self._analyze_with_gemini(
                    image_bytes,
                    enhanced_context_stage2,
                    telemetry,
                    stage=2,
                    frame_history=frame_history,
                )
                results["gpt4o_analysis"] = gpt4o_results
                results["suspicious_keywords_detected"] = suspicious_detected
                print("[OK] Two-stage analysis complete")
            elif suspicious_detected and self.single_gpt_stage:
                results["gpt4o_analysis"] = gpt4o_results_stage1
                results["suspicious_keywords_detected"] = suspicious_detected
                print("[OK] Single-stage mode — skipping extra GPT call (rate limit guard)")
            else:
                # No suspicious activity, use stage 1 results
                results['gpt4o_analysis'] = gpt4o_results_stage1
                print("[OK] Stage 1 sufficient - no suspicious activity detected")
                
        except Exception as e:
            print(f"[ERROR] Gemini analysis failed: {e}")
            if is_quota_exhausted_error(e):
                from src.api_retry import mark_quota_exhausted

                mark_quota_exhausted()
            results["gpt4o_analysis"] = {"success": False, "error": str(e)}
        
        # Step 5: Calculate overall threat
        results['overall_threat_level'] = self._calculate_overall_threat(
            results['clip_analysis'],
            results['blip_analysis'],
            results['gpt4o_analysis']
        )
        
        results['processing_time_ms'] = int((time.time() - start_time) * 1000)
        print(f"[OK] Cloud-enhanced analysis complete in {results['processing_time_ms']}ms")
        
        # Context persistence is handled by vision_analyzer.analyze_all_frames via unified_context
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
            response = self._hf_post(
                f"{HF_API_URL}/{CLIP_MODEL}",
                "hf-clip",
                json={
                    "inputs": {
                        "image": image_b64,
                        "candidate_labels": security_prompts[:20],
                    }
                },
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
            caption_response = self._hf_post(
                f"{HF_API_URL}/{BLIP_CAPTION_MODEL}",
                "hf-blip",
                data=image_bytes,
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
    
    def _analyze_with_gemini(
        self,
        image: Any,
        enhanced_context: str,
        telemetry: Dict[str, Any],
        stage: int = 1,
        frame_history: str = "",
    ) -> Dict[str, Any]:
        """Analyze with Gemini VLM using enhanced context."""
        try:
            # Stage-specific prompts
            if stage == 1:
                # STAGE 1: Initial observation with context-aware analysis
                prompt = f"""You are a security observer analyzing surveillance footage. Be PRECISE and FACTUAL.

=== FRAME HISTORY (Previous Activity Context) ===
{frame_history if frame_history else "No previous frames - establishing baseline."}

=== CURRENT FRAME CONTEXT (CRITICAL for understanding) ===
- Frame ID: {telemetry.get('frame_id', 'unknown')}
- Location: {telemetry.get('location', 'unknown')}
- Time: {telemetry.get('timestamp', 'unknown')}
- After Hours: {telemetry.get('is_after_hours', False)}
- Zone Type: {telemetry.get('zone_type', 'unknown')}
- Restricted Zone: {telemetry.get('is_restricted_zone', False)}

Additional analysis context:
{enhanced_context}

=== OBSERVATION REQUIREMENTS ===
1. SCENE TYPE: retail, warehouse, parking, perimeter, staff_area, storage, public_area
2. PEOPLE COUNT: Count EXACTLY - say "2 people" not "a few people"
3. PEOPLE POSITIONS: Where is each person? (behind counter, in aisle, near door, at register)
4. BODY LANGUAGE: What are hands doing? (reaching, holding items, in pockets, behind back)
5. OBJECT INTERACTIONS: Touching anything? Staff section? Merchandise? Register?
6. BEHAVIORAL SIGNALS: Looking around nervously? Moving quickly? Hiding?

=== HIGH-PRIORITY OBSERVATIONS (Flag these) ===
- Person behind counter/staff section
- Person reaching toward register/cashier area
- Person in restricted/staff-only area
- Person handling merchandise in suspicious manner
- Concealment behaviors (hiding items, checking for observers)
- After-hours presence in restricted zones

Provide detailed observation in JSON:
{{
  "vlm_description": "Precise scene description including exact positions, actions, and any high-priority observations",
  "scene_type": "retail|warehouse|parking|interior|exterior|staff_area|storage",
  "people_count": 0,
  "person_features": ["Person 1: [exact location + body position + hand activity + what they're doing]", "Person 2: [same format]"],
  "objects_detected": ["specific visible objects"],
  "activity": "summary of what people are actually doing",
  "security_signals": ["reaching_toward_staff_section", "behind_counter", "concealing_items", "checking_surroundings"],
  "suspicious_elements": ["person_in_unauthorized_area", "hands_near_register"],
  "threat_level": "CLEAR",
  "threat_type": "clear",
  "reasoning": "Factual observation. Note any suspicious positions/actions relative to location context.",
  "recommended_action": "Continue monitoring",
  "confidence": 0.9
}}

BE PRECISE about positions and actions. Location context matters!"""
            else:
                # STAGE 2: Security-focused analysis with situation understanding
                prompt = f"""You are a SENIOR SECURITY ANALYST. Apply SITUATION UNDERSTANDING — the same action can be innocent or CRITICAL based on context.

SECURITY ASSESSMENT REQUIRED 
=== FRAME HISTORY (Previous Activity Context) ===
{frame_history if frame_history else "No previous frames - establishing baseline."}

=== PREVIOUS FRAME ANALYSIS ===
{enhanced_context}

=== CONTEXT-AWARE THREAT ASSESSMENT ===
CRITICAL THREAT INDICATORS (CRITICAL / HIGH):
- THEFT: Reaching toward drawers, register, safe, display cases; pocketing items, group distraction.
- INTRUSION: Scaling fences/walls/gates, unauthorized entry into restricted/staff zones, tampering with entries.
- WEAPONS & VIOLENCE: Brandishing weapons (guns, knives), physical fighting, assault, hostage situations.
- SAFETY HAZARDS: Visible fire, smoke, safety violations, blocked fire exits, slips/falls (person lying down/injured).
- VANDALISM: Tampering with locks, prying doors/windows, graffiti, property damage.
- VEHICLES: Tailgating through security gates, blocking access roads/exits, unauthorized idling in restricted zones.

THREAT LEVELS:
- CRITICAL:
  * Weapons, fire, smoke, active physical altercations/violence.
  * Reaching or opening cash register, safe, high-value storage, or drawers below counter.
  * Intruder scaling fences/gates or forced entry in progress (lock prying, door forcing).
  * Medical emergency (person collapsed/unconscious on ground).
  * After-hours presence in restricted staff/vault zones.

- HIGH:
  * Person trespassing in restricted/staff-only zones.
  * Concealing items under clothing or in bags in a retail/warehouse area.
  * Attempting to open locked doors, gates, or containers.
  * Coordinated suspicious behavior (e.g. distraction tactics).
  * Loitering inside restricted/fenced areas.

- MEDIUM:
  * Hovering near valuables, cash register, or entry gates without purpose.
  * Repeatedly looking around or checking for cameras/observers (scouting).
  * Group crowding staff or entry gates with unclear purpose.
  * Vehicle parked in unauthorized area or tailgating a security gate.

- LOW:
  * Unfamiliar person in public area.
  * Minor rule violations or loitering in non-restricted public spaces.

- CLEAR:
  * Normal authorized activity (shopping, walking, working).
  * Appropriate social distance and actions relative to location.

=== KEY CONTEXT RULES ===
1. WEAPONS/FIRE/SAFETY: Any indication = CRITICAL immediately.
2. BOUNDARIES/INTRUSIONS: Scaling fence/gate or behind counter = HIGH minimum.
3. RETAIL/STORAGE: Reaching drawers/register/valuables = CRITICAL.
4. AFTER HOURS: Escalate any detected threat level by +1 level (e.g., MEDIUM becomes HIGH).

=== FRAME CONTEXT ===
- Location: {telemetry.get('location', 'unknown')}
- After Hours: {telemetry.get('is_after_hours', False)}
- Restricted Zone: {telemetry.get('is_restricted_zone', False)}
- Time: {telemetry.get('timestamp', 'unknown')}

Provide security assessment in JSON:
{{
  "threat_level": "CRITICAL|HIGH|MEDIUM|LOW|CLEAR",
  "threat_type": "theft_behavior|loitering|trespassing|unauthorized_access|suspicious_behavior|weapons_violence|fire_safety_hazard|forced_entry|clear",
  "vlm_description": "Security-focused description with exact positions and actions",
  "scene_type": "retail|warehouse|parking|interior|exterior|staff_area|storage|perimeter|gate",
  "people_count": 0,
  "person_features": ["Person 1: [position + action + security relevance]"],
  "security_signals": ["reaching_staff_section", "unauthorized_area_access", "concealment_behavior", "scaling_boundary", "tampering_lock", "smoke_detected", "weapon_visible"],
  "suspicious_elements": ["specific concerns with location context"],
  "reasoning": "EXPLAIN THE SITUATION: Person is [action] at [location] which is [threat level] because [specific contextual reason]. Connect action to location!",
  "recommended_action": "Specific security response",
  "confidence": 0.85
}}

IMPORTANT: Consider WHERE and WHEN the action is happening. Context is everything!"""

            gemini = generate_vision(
                prompt,
                image,
                max_output_tokens=4096,
                temperature=0.2,
            )
            if not gemini.get("success"):
                return {
                    "success": False,
                    "error": gemini.get("error", "Gemini vision failed"),
                    "gpt4o_analysis": None,
                }

            content = gemini["text"]
            model_used = gemini.get("model_used", settings.GEMINI_MODEL)
            
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
                        "success": True,
                        "gpt4o_analysis": content,
                        "model": model_used,
                        "threat_level": parsed.get("threat_level", "UNKNOWN"),
                        "threat_type": parsed.get("threat_type", "unknown"),
                        "confidence": parsed.get("confidence", 0.0),
                        "parsed_data": parsed,
                    }
            except Exception as e:
                print(f"[WARNING] Could not parse Gemini JSON: {e}")

            return {
                "success": True,
                "gpt4o_analysis": content,
                "model": model_used,
                "threat_level": "UNKNOWN",
                "threat_type": "unknown",
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
            t_level = str(gpt4o_results.get('threat_level', '')).upper()
            if 'CRITICAL' in t_level or 'HIGH' in t_level or 'ELEVATED' in t_level:
                threat_scores.append(80)
            elif 'MEDIUM' in t_level:
                threat_scores.append(50)
            elif 'LOW' in t_level:
                threat_scores.append(20)
            else:
                content = gpt4o_results.get('gpt4o_analysis', '')
                if 'HIGH' in content.upper() or 'CRITICAL' in content.upper() or 'ELEVATED' in content.upper():
                    threat_scores.append(80)
                elif 'MEDIUM' in content.upper():
                    threat_scores.append(50)
                elif 'LOW' in content.upper():
                    threat_scores.append(20)
        
        if not threat_scores:
            return 'UNKNOWN'
        
        avg_score = sum(threat_scores) / len(threat_scores)
        
        # RIGOROUS THRESHOLDS - Only significant threats get elevated status
        if avg_score >= 80:  # Was 70 - now requires stronger evidence
            return 'HIGH'
        elif avg_score >= 55:  # Was 40 - medium requires more proof
            return 'MEDIUM'
        elif avg_score >= 25:  # NEW - LOW tier
            return 'LOW'
        else:
            return 'CLEAR'  # Was LOW - now CLEAR for minimal scores


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
