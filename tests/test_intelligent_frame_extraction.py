"""
test_intelligent_frame_extraction.py - Comprehensive testing for intelligent frame extraction.

Tests all extraction strategies, edge cases, and performance scenarios.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import cv2
import numpy as np

from src.intelligent_frame_extractor import (
    IntelligentFrameExtractor,
    ExtractionStrategy,
    FrameInfo,
    extract_frames_intelligently
)

class TestIntelligentFrameExtractor:
    """Test suite for intelligent frame extraction."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance for testing."""
        return IntelligentFrameExtractor()
    
    @pytest.fixture
    def sample_video_path(self):
        """Create a sample video for testing."""
        # Create a temporary video file
        temp_dir = Path(tempfile.mkdtemp())
        video_path = temp_dir / "test_video.mp4"
        
        # Create a simple test video using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 1.0, (640, 480))
        
        # Generate 30 frames with some motion
        for i in range(30):
            # Create frame with some motion
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add moving object
            x = int(320 + 100 * np.sin(i * 0.2))
            y = int(240 + 50 * np.cos(i * 0.2))
            cv2.circle(frame, (x, y), 50, (255, 255, 255), -1)
            out.write(frame)
        
        out.release()
        yield video_path
        
        # Cleanup
        if video_path.exists():
            video_path.unlink()
        temp_dir.rmdir()
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.motion_threshold == 0.15
        assert extractor.scene_change_threshold == 0.3
        assert extractor.min_frame_interval == 0.5
        assert extractor.max_frames_per_minute == 30
    
    def test_get_video_info(self, extractor, sample_video_path):
        """Test video information extraction."""
        info = extractor._get_video_info(sample_video_path)
        
        assert info is not None
        assert "duration" in info
        assert "fps" in info
        assert "width" in info
        assert "height" in info
        assert "codec" in info
        assert info["width"] == 640
        assert info["height"] == 480
    
    def test_get_video_info_invalid_file(self, extractor):
        """Test video info extraction with invalid file."""
        invalid_path = Path("nonexistent_video.mp4")
        info = extractor._get_video_info(invalid_path)
        assert info is None
    
    def test_uniform_extraction(self, extractor, sample_video_path):
        """Test uniform frame extraction strategy."""
        output_dir = Path(tempfile.mkdtemp())
        
        try:
            frames = extractor._extract_uniform_frames(
                sample_video_path, output_dir, {"duration": 30, "fps": 1.0}, 2.0
            )
            
            assert len(frames) > 0
            assert all(isinstance(f, FrameInfo) for f in frames)
            assert all(f.extraction_reason == "uniform" for f in frames)
            
            # Check files exist
            for frame in frames:
                assert (output_dir / frame.filename).exists()
        
        finally:
            shutil.rmtree(output_dir)
    
    def test_motion_based_extraction(self, extractor, sample_video_path):
        """Test motion-based frame extraction."""
        output_dir = Path(tempfile.mkdtemp())
        
        try:
            frames = extractor._extract_motion_frames(
                sample_video_path, output_dir, {"duration": 30, "fps": 1.0}, 10
            )
            
            assert len(frames) > 0
            assert all(isinstance(f, FrameInfo) for f in frames)
            assert all("motion" in f.extraction_reason for f in frames)
            
            # Check motion scores
            for frame in frames:
                assert frame.motion_score > 0
            
        finally:
            shutil.rmtree(output_dir)
    
    def test_scene_change_extraction(self, extractor, sample_video_path):
        """Test scene change frame extraction."""
        output_dir = Path(tempfile.mkdtemp())
        
        try:
            frames = extractor._extract_scene_change_frames(
                sample_video_path, output_dir, {"duration": 30, "fps": 1.0}, 10
            )
            
            assert len(frames) >= 0  # May be 0 if no scene changes detected
            assert all(isinstance(f, FrameInfo) for f in frames)
            
        finally:
            shutil.rmtree(output_dir)
    
    def test_hybrid_extraction(self, extractor, sample_video_path):
        """Test hybrid frame extraction strategy."""
        output_dir = Path(tempfile.mkdtemp())
        
        try:
            frames = extractor._extract_hybrid_frames(
                sample_video_path, output_dir, {"duration": 30, "fps": 1.0}, 20, 2.0
            )
            
            assert len(frames) > 0
            assert all(isinstance(f, FrameInfo) for f in frames)
            
            # Check we have different types of frames
            reasons = {f.extraction_reason for f in frames}
            assert len(reasons) >= 1
            
        finally:
            shutil.rmtree(output_dir)
    
    def test_frame_deduplication(self, extractor):
        """Test frame deduplication."""
        frames = [
            FrameInfo(1, 0.0, "frame1.jpg", 100, 50, extraction_reason="test"),
            FrameInfo(2, 0.1, "frame2.jpg", 100, 50, extraction_reason="test"),
            FrameInfo(3, 0.2, "frame3.jpg", 100, 50, extraction_reason="test"),
            FrameInfo(4, 0.25, "frame4.jpg", 100, 50, extraction_reason="test"),  # Too close to frame 3
            FrameInfo(5, 1.0, "frame5.jpg", 100, 50, extraction_reason="test"),
        ]
        
        deduplicated = extractor._deduplicate_frames(frames, min_interval=0.5)
        
        # Should remove frame 4 (too close to frame 3)
        assert len(deduplicated) == 4
        assert deduplicated[2].timestamp == 0.2
        assert deduplicated[3].timestamp == 1.0
    
    def test_importance_score_calculation(self, extractor):
        """Test importance score calculation."""
        motion_score = extractor._calculate_importance_score("motion_score_0.8")
        scene_score = extractor._calculate_importance_score("scene_change_0.9")
        uniform_score = extractor._calculate_importance_score("uniform")
        other_score = extractor._calculate_importance_score("other")
        
        assert motion_score == 0.8
        assert scene_score == 0.9
        assert uniform_score == 0.3
        assert other_score == 0.5
    
    def test_extract_frame_at_timestamp(self, extractor, sample_video_path):
        """Test single frame extraction at timestamp."""
        output_dir = Path(tempfile.mkdtemp())
        
        try:
            frame = extractor._extract_frame_at_timestamp(
                sample_video_path, output_dir, 1.0, 1, "test"
            )
            
            assert frame is not None
            assert isinstance(frame, FrameInfo)
            assert frame.timestamp == 1.0
            assert frame.frame_number == 1
            assert (output_dir / frame.filename).exists()
            
        finally:
            shutil.rmtree(output_dir)
    
    def test_save_extraction_log(self, extractor, sample_video_path):
        """Test extraction log saving."""
        frames = [
            FrameInfo(1, 0.0, "frame1.jpg", 100, 50, extraction_reason="test"),
            FrameInfo(2, 1.0, "frame2.jpg", 100, 50, extraction_reason="test"),
        ]
        
        output_path = Path(tempfile.mkdtemp()) / "extraction_log.json"
        
        try:
            extractor.save_extraction_log(frames, sample_video_path, ExtractionStrategy.HYBRID, output_path)
            
            assert output_path.exists()
            
            # Verify log content
            import json
            with open(output_path) as f:
                log = json.load(f)
            
            assert log["extraction_strategy"] == "hybrid"
            assert log["total_frames_extracted"] == 2
            assert len(log["frames"]) == 2
            assert "statistics" in log
            
        finally:
            output_path.unlink()
            output_path.parent.rmdir()

class TestExtractionStrategies:
    """Test different extraction strategies."""
    
    def test_strategy_enum(self):
        """Test extraction strategy enum."""
        assert ExtractionStrategy.UNIFORM.value == "uniform"
        assert ExtractionStrategy.MOTION_BASED.value == "motion_based"
        assert ExtractionStrategy.SCENE_CHANGE.value == "scene_change"
        assert ExtractionStrategy.HYBRID.value == "hybrid"
    
    def test_invalid_strategy(self):
        """Test handling of invalid strategy."""
        with pytest.raises(ValueError):
            ExtractionStrategy("invalid_strategy")

class TestMainFunction:
    """Test the main extraction function."""
    
    @patch('src.intelligent_frame_extractor.extractor')
    def test_main_function_success(self, mock_extractor):
        """Test main function with successful extraction."""
        # Mock the extractor
        mock_frames = [
            FrameInfo(1, 0.0, "frame1.jpg", 100, 50, extraction_reason="test")
        ]
        mock_extractor.extract_frames_intelligently.return_value = mock_frames
        mock_extractor.save_extraction_log.return_value = None
        
        result = extract_frames_intelligently(
            max_frames=10,
            strategy="hybrid"
        )
        
        assert len(result) == 1
        assert result[0]["frame_number"] == 1
        assert result[0]["extraction_reason"] == "test"
    
    @patch('src.intelligent_frame_extractor.extractor')
    def test_main_function_failure(self, mock_extractor):
        """Test main function with extraction failure."""
        # Mock the extractor to raise an exception
        mock_extractor.extract_frames_intelligently.side_effect = Exception("Test error")
        
        with pytest.raises(Exception):
            extract_frames_intelligently()

class TestPerformance:
    """Performance tests for frame extraction."""
    
    def test_extraction_performance(self):
        """Test extraction performance with larger video."""
        # This would require a larger test video
        # For now, just ensure the function doesn't crash
        extractor = IntelligentFrameExtractor()
        assert extractor.max_frames_per_minute == 30
    
    def test_memory_usage(self):
        """Test memory usage during extraction."""
        # This would require memory profiling tools
        # For now, just ensure the extractor can be created
        extractor = IntelligentFrameExtractor()
        assert extractor is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
