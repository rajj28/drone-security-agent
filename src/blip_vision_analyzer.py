"""
blip_vision_analyzer.py - BLIP-based vision analysis for security

Uses Salesforce BLIP (Bootstrapping Language-Image Pre-training) for:
- Image captioning and description generation
- Visual Question Answering (VQA) for security queries
- Object detection and scene understanding
- Enhanced context for GPT-4o analysis
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings("ignore")

class BLIPVisionAnalyzer:
    """BLIP-based vision analyzer for security applications."""
    
    _instance = None
    _caption_model = None
    _caption_processor = None
    _vqa_model = None
    _vqa_processor = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BLIPVisionAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._caption_model is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Initializing BLIP Vision Analyzer on {self._device}")
            
            # Initialize BLIP for image captioning
            print("Loading BLIP captioning model...")
            self._caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self._device)
            self._caption_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            print("BLIP captioning model loaded successfully")
            
            # Initialize BLIP for VQA
            print("Loading BLIP VQA model...")
            self._vqa_model = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-base"
            ).to(self._device)
            self._vqa_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-vqa-base"
            )
            print("BLIP VQA model loaded successfully")
            
            # Security-specific questions for VQA
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
            
            print("BLIP Vision Analyzer initialized successfully")
    
    @property
    def device(self):
        return self._device
    
    def generate_caption(self, image_path: Path) -> str:
        """Generate image caption using BLIP."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self._caption_processor(image, return_tensors="pt").to(self._device)
            
            with torch.no_grad():
                out = self._caption_model.generate(**inputs, max_length=50)
            
            caption = self._caption_processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            print(f"Caption generation failed: {e}")
            return "Unable to generate caption"
    
    def answer_security_questions(self, image_path: Path) -> Dict[str, str]:
        """Answer security-specific questions using BLIP VQA."""
        try:
            image = Image.open(image_path).convert("RGB")
            answers = {}
            
            for question in self.security_questions:
                inputs = self._vqa_processor(image, question, return_tensors="pt").to(self._device)
                
                with torch.no_grad():
                    out = self._vqa_model.generate(**inputs, max_length=20)
                
                answer = self._vqa_processor.decode(out[0], skip_special_tokens=True)
                answers[question] = answer
            
            return answers
            
        except Exception as e:
            print(f"VQA failed: {e}")
            return {q: "Unable to answer" for q in self.security_questions}
    
    def extract_security_insights(self, vqa_answers: Dict[str, str]) -> Dict[str, Any]:
        """Extract security-relevant insights from VQA answers."""
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
            # Extract number from answer
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
        location_answer = vqa_answers.get("What is the location or setting?", "")
        insights["location"] = location_answer
        
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
        clothing_answer = vqa_answers.get("What type of clothing are people wearing?", "")
        insights["clothing_description"] = clothing_answer
        
        return insights
    
    def comprehensive_blip_analysis(self, image_path: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive BLIP-based analysis."""
        print(f"Running BLIP comprehensive analysis for {image_path.name}")
        
        # Step 1: Generate caption
        print("Generating image caption...")
        caption = self.generate_caption(image_path)
        
        # Step 2: Answer security questions
        print("Answering security questions...")
        vqa_answers = self.answer_security_questions(image_path)
        
        # Step 3: Extract security insights
        print("Extracting security insights...")
        security_insights = self.extract_security_insights(vqa_answers)
        
        # Step 4: Calculate threat level
        threat_level = self._calculate_blip_threat(security_insights)
        
        # Combine results
        results = {
            "image_path": str(image_path),
            "telemetry": telemetry,
            "blip_caption": caption,
            "blip_vqa_answers": vqa_answers,
            "blip_security_insights": security_insights,
            "blip_threat_level": threat_level,
            "analysis_method": "BLIP",
            "model_used": "Salesforce/BLIP"
        }
        
        return results
    
    def _calculate_blip_threat(self, insights: Dict[str, Any]) -> str:
        """Calculate threat level based on BLIP insights."""
        threat_score = 0
        
        if insights["suspicious_activity"]:
            threat_score += 3
        
        if insights["weapons_detected"]:
            threat_score += 5
        
        if insights["reaching_behavior"]:
            threat_score += 2
        
        if insights["people_count"] > 3:
            threat_score += 1
        
        if insights["bags_containers"]:
            threat_score += 1
        
        if threat_score >= 5:
            return "HIGH"
        elif threat_score >= 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_enhanced_context_for_gpt4o(self, image_path: Path) -> str:
        """Generate enhanced context for GPT-4o using BLIP analysis."""
        caption = self.generate_caption(image_path)
        vqa_answers = self.answer_security_questions(image_path)
        insights = self.extract_security_insights(vqa_answers)
        
        context_parts = [
            "BLIP COMPUTER VISION ANALYSIS:",
            f"Image Caption: {caption}",
            "",
            "Security Question Answers:"
        ]
        
        for question, answer in vqa_answers.items():
            context_parts.append(f"  Q: {question}")
            context_parts.append(f"  A: {answer}")
        
        context_parts.extend([
            "",
            "Extracted Security Insights:",
            f"  People Count: {insights['people_count']}",
            f"  Suspicious Activity: {insights['suspicious_activity']}",
            f"  Weapons Detected: {insights['weapons_detected']}",
            f"  Location: {insights['location']}",
            f"  Reaching Behavior: {insights['reaching_behavior']}",
            f"  BLIP Threat Assessment: {self._calculate_blip_threat(insights)}"
        ])
        
        return "\n".join(context_parts)


def analyze_with_blip(image_path: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Main function for BLIP-based analysis."""
    analyzer = BLIPVisionAnalyzer()
    return analyzer.comprehensive_blip_analysis(image_path, telemetry)


if __name__ == "__main__":
    # Test BLIP analyzer
    test_image = Path("data/extracted/frame_017.jpg")
    if test_image.exists():
        telemetry = {
            "frame_id": "frame_017",
            "timestamp": "16:00:16",
            "location": "Garage",
            "is_after_hours": False,
            "is_restricted_zone": False
        }
        
        result = analyze_with_blip(test_image, telemetry)
        print(json.dumps(result, indent=2))
    else:
        print(f"Test image not found: {test_image}")
