"""
lightweight_vision_enhancer.py - Lightweight vision enhancement using OpenCV and PIL

Provides basic computer vision capabilities to enhance GPT-4o analysis without heavy model downloads.
Uses OpenCV for object detection, motion analysis, and basic visual features.
"""

import cv2
import numpy as np
from PIL import Image, ImageStat
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
import base64
from openai import OpenAI
from src.config import settings

class LightweightVisionEnhancer:
    """Lightweight vision enhancement using OpenCV and basic computer vision."""
    
    def __init__(self):
        print("Initializing Lightweight Vision Enhancer...")
        
        # Load pre-trained models (lightweight versions)
        try:
            # Load face detection model
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Load person detection model (HOG)
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            print("OpenCV models loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load OpenCV models: {e}")
            self.face_cascade = None
            self.hog = None
        
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Security-specific analysis patterns
        self.security_patterns = {
            'hand_positions': ['hand_near_pocket', 'hand_in_pocket', 'hand_concealing', 'reaching_motion'],
            'body_language': ['turned_away', 'hunched_posture', 'nervous_movement', 'avoiding_gaze'],
            'movement_patterns': ['quick_movements', 'sudden_direction_change', 'loitering', 'circling'],
            'interaction_patterns': ['blocking_view', 'distraction_behavior', 'coordinated_movement']
        }
    
    def analyze_image_features(self, image_path: Path) -> Dict[str, Any]:
        """Extract basic visual features using OpenCV."""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {'error': 'Could not load image'}
            
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            features = {
                'image_dimensions': image.shape,
                'brightness': np.mean(gray),
                'contrast': np.std(gray),
                'dominant_colors': self._get_dominant_colors(hsv),
                'edge_density': self._calculate_edge_density(gray),
                'motion_blur': self._detect_motion_blur(gray)
            }
            
            # Detect people and faces
            if self.hog is not None:
                features['people_detected'] = self._detect_people(image)
            
            if self.face_cascade is not None:
                features['faces_detected'] = self._detect_faces(gray)
            
            # Analyze regions of interest
            features['regions_of_interest'] = self._analyze_regions(image)
            
            return features
            
        except Exception as e:
            return {'error': f'Feature extraction failed: {e}'}
    
    def _get_dominant_colors(self, hsv_image: np.ndarray, k: int = 3) -> List[str]:
        """Get dominant colors from HSV image."""
        try:
            # Reshape image to list of pixels
            pixels = hsv_image.reshape(-1, 3)
            
            # Simple color analysis (avoiding heavy K-means)
            h_mean = np.mean(pixels[:, 0])
            s_mean = np.mean(pixels[:, 1])
            v_mean = np.mean(pixels[:, 2])
            
            colors = []
            if h_mean < 10 or h_mean > 170:
                colors.append('red')
            elif 10 <= h_mean < 25:
                colors.append('orange')
            elif 25 <= h_mean < 35:
                colors.append('yellow')
            elif 35 <= h_mean < 85:
                colors.append('green')
            elif 85 <= h_mean < 125:
                colors.append('blue')
            elif 125 <= h_mean < 155:
                colors.append('purple')
            else:
                colors.append('pink')
            
            if s_mean < 50:
                colors.append('desaturated')
            if v_mean < 100:
                colors.append('dark')
            elif v_mean > 200:
                colors.append('bright')
            
            return colors[:k]
            
        except Exception:
            return ['unknown']
    
    def _calculate_edge_density(self, gray_image: np.ndarray) -> float:
        """Calculate edge density in the image."""
        try:
            edges = cv2.Canny(gray_image, 50, 150)
            return np.mean(edges) / 255.0
        except Exception:
            return 0.0
    
    def _detect_motion_blur(self, gray_image: np.ndarray) -> float:
        """Detect motion blur using Laplacian variance."""
        try:
            return cv2.Laplacian(gray_image, cv2.CV_64F).var()
        except Exception:
            return 0.0
    
    def _detect_people(self, image: np.ndarray) -> int:
        """Detect people using HOG detector."""
        try:
            # Resize for better detection
            height, width = image.shape[:2]
            if width > 640:
                scale = 640 / width
                image = cv2.resize(image, (640, int(height * scale)))
            
            # Detect people
            boxes, weights = self.hog.detectMultiScale(image, winStride=(8, 8))
            return len(boxes)
        except Exception:
            return 0
    
    def _detect_faces(self, gray_image: np.ndarray) -> int:
        """Detect faces using Haar cascade."""
        try:
            faces = self.face_cascade.detectMultiScale(gray_image, 1.1, 4)
            return len(faces)
        except Exception:
            return 0
    
    def _analyze_regions(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze different regions of the image."""
        try:
            height, width = image.shape[:2]
            
            # Define regions
            regions = {
                'upper_third': image[:height//3, :],
                'middle_third': image[height//3:2*height//3, :],
                'lower_third': image[2*height//3:, :],
                'left_half': image[:, :width//2],
                'right_half': image[:, width//2:],
                'center': image[height//4:3*height//4, width//4:3*width//4]
            }
            
            region_analysis = {}
            for region_name, region_img in regions.items():
                gray_region = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
                region_analysis[region_name] = {
                    'brightness': np.mean(gray_region),
                    'activity_level': np.std(gray_region),
                    'edge_density': self._calculate_edge_density(gray_region)
                }
            
            return region_analysis
            
        except Exception:
            return {}
    
    def generate_enhanced_prompt(self, image_features: Dict[str, Any], telemetry: Dict[str, Any]) -> str:
        """Generate enhanced prompt for GPT-4o using visual features."""
        
        prompt_parts = [
            "ENHANCED COMPUTER VISION ANALYSIS:",
            f"Image Analysis Results: {json.dumps(image_features, indent=2)}",
            f"Telemetry Data: {json.dumps(telemetry, indent=2)}",
            "",
            "SECURITY THREAT ASSESSMENT MISSION:",
            "You have computer vision insights to enhance your analysis. Use these features:",
            "",
            "VISUAL EVIDENCE TO CONSIDER:",
        ]
        
        # Add specific insights based on features
        if 'people_detected' in image_features:
            people_count = image_features['people_detected']
            prompt_parts.append(f"- People detected by computer vision: {people_count}")
            
            if people_count >= 2:
                prompt_parts.append("- MULTIPLE PERSONS: Higher theft risk - look for coordinated behavior")
        
        if 'faces_detected' in image_features:
            faces_count = image_features['faces_detected']
            prompt_parts.append(f"- Faces detected: {faces_count}")
            
            if faces_count > 0:
                prompt_parts.append("- FACES VISIBLE: Check for eye contact, nervous expressions")
        
        if 'regions_of_interest' in image_features:
            regions = image_features['regions_of_interest']
            
            # Check for activity in different regions
            if 'center' in regions and regions['center']['activity_level'] > 50:
                prompt_parts.append("- HIGH CENTER ACTIVITY: Main action area - monitor closely")
            
            if 'lower_third' in regions and regions['lower_third']['activity_level'] > 40:
                prompt_parts.append("- LOWER AREA ACTIVITY: Possible hand movements/concealment")
        
        # Add lighting and quality insights
        if 'brightness' in image_features:
            brightness = image_features['brightness']
            if brightness < 100:
                prompt_parts.append("- LOW LIGHTING: May conceal suspicious activities")
            elif brightness > 200:
                prompt_parts.append("- BRIGHT LIGHTING: Good visibility for detailed analysis")
        
        if 'motion_blur' in image_features and image_features['motion_blur'] < 100:
            prompt_parts.append("- MOTION BLUR DETECTED: Rapid movements may indicate suspicious activity")
        
        prompt_parts.extend([
            "",
            "CRITICAL FOCUS AREAS:",
            "- Theft detection: Look for hand movements, concealment, quick actions",
            "- Suspicious behavior: Check for nervousness, avoidance, positioning",
            "- Group dynamics: Look for distraction techniques, coordinated movements",
            "- Security threats: Identify any criminal activity patterns",
            "",
            "Provide detailed security analysis with specific threat indicators.",
            "Return JSON with threat assessment, behavioral analysis, and recommendations."
        ])
        
        return "\n".join(prompt_parts)
    
    def analyze_with_enhanced_gpt4o(self, image_path: Path, enhanced_prompt: str) -> Dict[str, Any]:
        """Analyze image with GPT-4o using enhanced visual context."""
        
        try:
            # Encode image
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an enhanced security analyst with computer vision insights. Use visual features to make accurate threat assessments."
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
                'model_used': 'gpt-4o-cv-enhanced',
                'success': True
            }
            
        except Exception as e:
            return {
                'enhanced_analysis': '',
                'model_used': 'failed',
                'success': False,
                'error': str(e)
            }
    
    def comprehensive_analysis(self, image_path: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis using computer vision enhancement."""
        print(f"Starting enhanced analysis for {image_path.name}")
        
        # Step 1: Extract visual features
        print("Extracting computer vision features...")
        visual_features = self.analyze_image_features(image_path)
        
        # Step 2: Generate enhanced prompt
        enhanced_prompt = self.generate_enhanced_prompt(visual_features, telemetry)
        
        # Step 3: Analyze with enhanced GPT-4o
        print("Running enhanced GPT-4o analysis...")
        gpt4o_results = self.analyze_with_enhanced_gpt4o(image_path, enhanced_prompt)
        
        # Step 4: Combine results
        combined_results = {
            'image_path': str(image_path),
            'telemetry': telemetry,
            'visual_features': visual_features,
            'enhanced_analysis': gpt4o_results,
            'analysis_success': gpt4o_results.get('success', False)
        }
        
        return combined_results

def analyze_frame_lightweight_enhanced(image_path: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Main function for lightweight enhanced frame analysis."""
    enhancer = LightweightVisionEnhancer()
    return enhancer.comprehensive_analysis(image_path, telemetry)
