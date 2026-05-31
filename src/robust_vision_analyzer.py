"""
robust_vision_analyzer.py — Enhanced vision analyzer with robust error handling and unknown video support.

- Handles various video qualities and conditions
- Fallback strategies for poor quality frames
- Adaptive prompting based on video content
- Enhanced person detection with multiple confidence levels
- Production-ready error handling and logging
"""

import base64
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from openai import OpenAI
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustVisionAnalyzer:
    """Enhanced vision analyzer with robust handling of unknown videos."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.processed_count = 0
        self.error_count = 0
        self.fallback_count = 0
        
    def _preprocess_image(self, image_path: Path) -> Optional[Image.Image]:
        """Preprocess image for better analysis with various enhancements."""
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Check image quality and apply enhancements
            img_array = np.array(img)
            
            # Calculate image quality metrics
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            logger.info(f"Image quality - Blur: {blur_score:.2f}, Brightness: {brightness:.2f}, Contrast: {contrast:.2f}")
            
            # Apply enhancements based on quality
            if blur_score < 100:  # Blurry image
                logger.info("Applying sharpening filter")
                img = img.filter(ImageFilter.SHARPEN)
                self.fallback_count += 1
            
            if brightness < 50:  # Dark image
                logger.info("Applying brightness enhancement")
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(1.5)
                self.fallback_count += 1
            
            if contrast < 30:  # Low contrast
                logger.info("Applying contrast enhancement")
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.5)
                self.fallback_count += 1
            
            # Resize if too large (to reduce API costs and improve processing)
            if img.size[0] > 1920 or img.size[1] > 1080:
                img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {img.size}")
            
            return img
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def _get_adaptive_prompt(self, telemetry: Dict[str, Any], image_quality: Dict[str, float]) -> str:
        """Generate adaptive prompt based on telemetry and image quality."""
        base_prompt = (
            "You are an expert drone security analyst AI. Analyze this CCTV frame and provide a detailed security assessment. "
            "Focus on security-relevant evidence, not generic descriptions. "
        )
        
        # Add quality-specific instructions
        if image_quality.get('blur_score', 0) < 100:
            base_prompt += "Note: This image appears to be blurry. Focus on clearly visible elements and indicate uncertainty. "
        
        if image_quality.get('brightness', 128) < 50:
            base_prompt += "Note: This image appears dark. Pay special attention to silhouettes and light sources. "
        
        if image_quality.get('contrast', 50) < 30:
            base_prompt += "Note: This image has low contrast. Be conservative in your analysis. "
        
        # Add context-specific instructions
        location = telemetry.get('location', '').lower()
        if 'restricted' in location:
            base_prompt += "This is a restricted area. Pay extra attention to unauthorized access. "
        elif 'entrance' in location or 'gate' in location:
            base_prompt += "This is an entrance/gate area. Focus on access control and visitor behavior. "
        elif 'parking' in location or 'garage' in location:
            base_prompt += "This is a parking/garage area. Focus on vehicle security and suspicious behavior. "
        
        # Time-based instructions
        time_str = telemetry.get('timestamp', '')
        if time_str:
            try:
                hour = int(time_str.split(':')[0])
                if hour < 6 or hour > 20:
                    base_prompt += "This is after hours. Be alert for unusual activity. "
            except:
                pass
        
        base_prompt += "Always respond in valid JSON format only."
        
        return base_prompt
    
    def _get_enhanced_user_prompt(self, telemetry: Dict[str, Any]) -> str:
        """Get enhanced user prompt with detailed person tracking focus."""
        return (
            f"Analyze this security camera frame with focus on detailed person identification and tracking. "
            f"Telemetry context: {json.dumps(telemetry, indent=2)}. "
            f"Return rich, structured security metadata that supports later alert reasoning and person tracking. "
            f"For each person, provide detailed attributes for tracking across frames. "
            f"Use the exact JSON format below and include only fields you can support from the image.\n"
            f"{{\n"
            f"  'vlm_description': 'detailed description of the scene',\n"
            f"  'scene_type': 'parking_lot|road|entrance|warehouse|residential|interior|unknown|outdoor|indoor',\n"
            f"  'objects_detected': ['list', 'of', 'object labels'],\n"
            f"  'object_details': [\n"
            f"    {{'label': 'person', 'count': 1, 'confidence': 0.9, 'attributes': ['standing', 'near_vehicle']}},\n"
            f"    {{'label': 'vehicle', 'count': 1, 'confidence': 0.85, 'attributes': ['sedan', 'parked']}}\n"
            f"  ],\n"
            f"  'people_count': 0,\n"
            f"  'person_features': [\n"
            f"    {{\n"
            f"      'id': 'person_1',\n"
            f"      'clothing_color': 'red|blue|green|black|white|gray|yellow|orange|purple|pink|brown|unknown',\n"
            f"      'clothing_type': 'shirt|jacket|coat|dress|uniform|hoodie|t-shirt|sweater|unknown',\n"
            f"      'body_type': 'slim|average|heavy|athletic|unknown',\n"
            f"      'height_estimate': 'short|average|tall|unknown',\n"
            f"      'distinctive_features': ['cap|hat|glasses|beard|mask|backpack|bag|phone|umbrella'],\n"
            f"      'face_visible': true|false,\n"
            f"      'actions': ['walking|standing|running|sitting|crouching|carrying_object'],\n"
            f"      'position_in_frame': 'left|center|right|top_left|top_right|bottom_left|bottom_right',\n"
            f"      'confidence': 0.9,\n"
            f"      'tracking_notes': 'notes for cross-frame identification'\n"
            f"    }}\n"
            f"  ],\n"
            f"  'vehicles_detected': [],\n"
            f"  'vehicle_details': [\n"
            f"    {{\n"
            f"      'type': 'sedan|suv|truck|van|motorcycle|bus|bicycle|unknown',\n"
            f"      'color': 'red|blue|green|black|white|gray|silver|yellow|orange|unknown',\n"
            f"      'position': 'parked|moving|stopped|unknown',\n"
            f"      'direction': 'north|south|east|west|unknown',\n"
            f"      'occupancy': 'driver_only|multiple|empty|unknown',\n"
            f"      'plate_visible': true|false,\n"
            f"      'plate_number': 'if_visible',\n"
            f"      'make_model': 'if_identifiable',\n"
            f"      'condition': 'good|damaged|unknown'\n"
            f"    }}\n"
            f"  ],\n"
            f"  'activity': 'detailed description of activity with person and vehicle details',\n"
            f"  'security_signals': ['after_hours_presence', 'restricted_zone_vehicle', 'unauthorized_access', 'suspicious_behavior', 'loitering', 'vandalism', 'theft'],\n"
            f"  'suspicious_elements': ['list of suspicious observations'],\n"
            f"  'threat_assessment': 'none|low|medium|high|critical',\n"
            f"  'confidence': 0.95,\n"
            f"  'recommended_action': 'specific recommended action',\n"
            f"  'alert_reasoning': 'detailed reasoning for threat level',\n"
            f"  'alert_priority_signals': ['person_after_hours', 'vehicle_loitering', 'unauthorized_access', 'suspicious_package'],\n"
            f"  'image_quality_assessment': 'clear|blurry|dark|low_contrast|noisy',\n"
            f"  'analysis_confidence': 0.9,\n"
            f"  'uncertainty_notes': 'any uncertainties in the analysis'\n"
            f"}}"
        )
    
    def _analyze_with_fallback(self, image_path: Path, system_prompt: str, user_prompt: str, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image with multiple fallback strategies."""
        
        # Try original image first
        try:
            result = self._single_analysis_attempt(image_path, system_prompt, user_prompt)
            if result and self._validate_analysis(result):
                return result
        except Exception as e:
            logger.warning(f"Primary analysis failed: {e}")
        
        # Fallback 1: Enhanced image
        logger.info("Trying fallback 1: Enhanced image")
        try:
            enhanced_img = self._preprocess_image(image_path)
            if enhanced_img:
                # Save enhanced version temporarily
                enhanced_path = image_path.parent / f"enhanced_{image_path.name}"
                enhanced_img.save(enhanced_path)
                
                result = self._single_analysis_attempt(enhanced_path, system_prompt, user_prompt)
                if result and self._validate_analysis(result):
                    enhanced_path.unlink()  # Clean up
                    return result
                
                enhanced_path.unlink()  # Clean up
        except Exception as e:
            logger.warning(f"Fallback 1 failed: {e}")
        
        # Fallback 2: Simplified prompt
        logger.info("Trying fallback 2: Simplified prompt")
        try:
            simplified_prompt = (
                "Analyze this security image and provide basic JSON: "
                "{\"objects_detected\": [], \"people_count\": 0, \"threat_assessment\": \"none\"}"
            )
            result = self._single_analysis_attempt(image_path, system_prompt, simplified_prompt)
            if result:
                # Add default values for missing fields
                return self._add_default_values(result, telemetry)
        except Exception as e:
            logger.warning(f"Fallback 2 failed: {e}")
        
        # Fallback 3: Default analysis
        logger.info("Using fallback 3: Default analysis")
        return self._create_default_analysis(telemetry, image_path.name)
    
    def _single_analysis_attempt(self, image_path: Path, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Single attempt at image analysis."""
        try:
            # Encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Make API call
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
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
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from API")
            
            # Parse JSON
            try:
                analysis = json.loads(content)
                return analysis
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    return analysis
                else:
                    raise ValueError("No valid JSON found in response")
                    
        except Exception as e:
            logger.error(f"Analysis attempt failed: {e}")
            return None
    
    def _validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Validate that analysis meets minimum requirements."""
        required_fields = ['objects_detected', 'people_count', 'threat_assessment']
        
        for field in required_fields:
            if field not in analysis:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Check for reasonable values
        if analysis.get('people_count', 0) < 0:
            logger.warning("Invalid people count")
            return False
        
        if analysis.get('threat_assessment') not in ['none', 'low', 'medium', 'high', 'critical']:
            logger.warning(f"Invalid threat assessment: {analysis.get('threat_assessment')}")
            return False
        
        return True
    
    def _add_default_values(self, analysis: Dict[str, Any], telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Add default values for missing fields."""
        defaults = {
            'vlm_description': 'Basic security analysis',
            'scene_type': 'unknown',
            'object_details': [],
            'person_features': [],
            'vehicles_detected': [],
            'vehicle_details': [],
            'activity': 'No specific activity detected',
            'security_signals': [],
            'suspicious_elements': [],
            'confidence': 0.5,
            'recommended_action': 'Continue monitoring',
            'alert_reasoning': 'Limited analysis due to image quality',
            'alert_priority_signals': [],
            'image_quality_assessment': 'poor',
            'analysis_confidence': 0.3,
            'uncertainty_notes': 'Analysis performed with fallback method'
        }
        
        for key, value in defaults.items():
            if key not in analysis:
                analysis[key] = value
        
        return analysis
    
    def _create_default_analysis(self, telemetry: Dict[str, Any], frame_name: str) -> Dict[str, Any]:
        """Create default analysis when all else fails."""
        return {
            'vlm_description': f'Unable to analyze frame {frame_name} due to technical issues',
            'scene_type': 'unknown',
            'objects_detected': [],
            'object_details': [],
            'people_count': 0,
            'person_features': [],
            'vehicles_detected': [],
            'vehicle_details': [],
            'activity': 'No activity detected - analysis failed',
            'security_signals': [],
            'suspicious_elements': ['analysis_failure'],
            'threat_assessment': 'none',
            'confidence': 0.0,
            'recommended_action': 'Manual review required',
            'alert_reasoning': 'Automated analysis failed - requires human review',
            'alert_priority_signals': [],
            'image_quality_assessment': 'failed',
            'analysis_confidence': 0.0,
            'uncertainty_notes': 'Complete analysis failure - technical issues'
        }
    
    def analyze_frame(self, frame_path: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single frame with robust error handling."""
        try:
            # Calculate image quality
            img_array = np.array(Image.open(frame_path))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            image_quality = {
                'blur_score': cv2.Laplacian(gray, cv2.CV_64F).var(),
                'brightness': np.mean(gray),
                'contrast': np.std(gray)
            }
            
            # Get adaptive prompts
            system_prompt = self._get_adaptive_prompt(telemetry, image_quality)
            user_prompt = self._get_enhanced_user_prompt(telemetry)
            
            # Perform analysis with fallbacks
            analysis = self._analyze_with_fallback(frame_path, system_prompt, user_prompt, telemetry)
            
            # Add metadata
            analysis['frame_id'] = frame_path.stem
            analysis['timestamp'] = telemetry.get('timestamp', '')
            analysis['location'] = telemetry.get('location', '')
            analysis['processing_metadata'] = {
                'image_quality': image_quality,
                'processing_time': time.time(),
                'fallback_used': self.fallback_count > 0,
                'analysis_method': 'robust_vision_analyzer'
            }
            
            self.processed_count += 1
            logger.info(f"Successfully analyzed {frame_path.name} (Processed: {self.processed_count})")
            
            return analysis
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Frame analysis failed for {frame_path.name}: {e}")
            return self._create_default_analysis(telemetry, frame_path.name)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'fallback_count': self.fallback_count,
            'success_rate': (self.processed_count / max(1, self.processed_count + self.error_count)) * 100
        }

# Global analyzer instance
analyzer = RobustVisionAnalyzer()

def analyze_frame_robust(frame_path: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single frame with robust handling."""
    return analyzer.analyze_frame(frame_path, telemetry)

def run_robust_analysis():
    """Run robust analysis on all frames."""
    logger.info("🔍 Starting robust vision analysis...")
    
    # Get all analysis files
    analysis_files = sorted(settings.ANALYSIS_DIR.glob("frame_*_analysis.json"))
    telemetry_files = sorted(settings.TELEMETRY_DIR.glob("frame_*_telemetry.json"))
    
    logger.info(f"Found {len(analysis_files)} analysis files")
    
    processed_count = 0
    for analysis_file in analysis_files:
        frame_id = analysis_file.stem.replace("_analysis", "")
        
        try:
            # Load telemetry
            telemetry_file = settings.TELEMETRY_DIR / f"{frame_id}_telemetry.json"
            with open(telemetry_file, 'r', encoding='utf-8') as f:
                telemetry = json.load(f)
            
            # Get frame path
            frame_path = settings.EXTRACTED_DIR / f"{frame_id}.jpg"
            if not frame_path.exists():
                logger.warning(f"Frame file not found: {frame_path}")
                continue
            
            # Analyze frame
            analysis = analyze_frame_robust(frame_path, telemetry)
            
            # Save analysis
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            processed_count += 1
            logger.info(f"✅ Processed {frame_id}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {frame_id}: {e}")
    
    # Print statistics
    stats = analyzer.get_processing_stats()
    logger.info(f"\n📊 Robust Analysis Statistics:")
    logger.info(f"   Frames processed: {stats['processed_count']}")
    logger.info(f"   Errors: {stats['error_count']}")
    logger.info(f"   Fallbacks used: {stats['fallback_count']}")
    logger.info(f"   Success rate: {stats['success_rate']:.1f}%")
    
    return stats

if __name__ == "__main__":
    run_robust_analysis()
