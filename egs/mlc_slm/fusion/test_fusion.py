#!/usr/bin/env python3
"""
Simple test script for ASR Speaker Fusion module
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asr_speaker_fusion import ASRSpeakerFusion, WordSegment, SpeakerSegment

def test_basic_fusion():
    """Test basic fusion functionality"""
    print("Testing basic fusion functionality...")
    
    # Create test data
    words = [
        WordSegment("Hello", 0.0, 0.5, 0.95),
        WordSegment("world", 0.5, 1.0, 0.92),
        WordSegment("how", 1.0, 1.3, 0.88),
        WordSegment("are", 1.3, 1.5, 0.90),
        WordSegment("you", 1.5, 1.8, 0.87),
        WordSegment("today", 1.8, 2.2, 0.89),
    ]
    
    speakers = [
        SpeakerSegment("speaker_1", 0.0, 1.2, 0.98),
        SpeakerSegment("speaker_2", 1.2, 2.5, 0.97),
    ]
    
    # Create fusion instance
    fusion = ASRSpeakerFusion()
    
    # Test fusion
    fused_words = fusion.fuse(words, speakers, parallel=False)
    
    # Display results
    print(f"\nFused {len(fused_words)} words:")
    for i, word in enumerate(fused_words):
        print(f"  {i+1}. '{word.word}' ({word.start_time:.1f}s-{word.end_time:.1f}s) -> {word.speaker_id} ({word.fusion_method})")
    
    # Get statistics
    stats = fusion.get_fusion_statistics(fused_words)
    print(f"\nFusion statistics:")
    print(f"  Total words: {stats['total_words']}")
    print(f"  Overlap fused: {stats['overlap_fused']} ({stats['overlap_percentage']:.1f}%)")
    print(f"  Distance fused: {stats['distance_fused']} ({stats['distance_percentage']:.1f}%)")
    print(f"  Speakers: {list(stats['speaker_distribution'].keys())}")
    
    return True

def test_edge_cases():
    """Test edge cases"""
    print("\nTesting edge cases...")
    
    # Test with no speakers
    words = [WordSegment("test", 0.0, 1.0, 0.9)]
    speakers = []
    
    fusion = ASRSpeakerFusion()
    try:
        result = fusion.fuse(words, speakers)
        print("  ✓ Handled empty speakers list")
    except Exception as e:
        print(f"  ✗ Failed to handle empty speakers: {e}")
        return False
    
    # Test with no words
    words = []
    speakers = [SpeakerSegment("speaker_1", 0.0, 1.0, 0.9)]
    
    try:
        result = fusion.fuse(words, speakers)
        print("  ✓ Handled empty words list")
    except Exception as e:
        print(f"  ✗ Failed to handle empty words: {e}")
        return False
    
    return True

def test_validation():
    """Test input validation"""
    print("\nTesting input validation...")
    
    # Test invalid time range
    try:
        invalid_word = WordSegment("test", 1.0, 0.5, 0.9)  # start > end
        print("  ✗ Should have failed for invalid time range")
        return False
    except ValueError:
        print("  ✓ Correctly validated time range")
    
    try:
        invalid_speaker = SpeakerSegment("test", 2.0, 1.0, 0.9)  # start > end
        print("  ✗ Should have failed for invalid time range")
        return False
    except ValueError:
        print("  ✓ Correctly validated speaker time range")
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("ASR Speaker Fusion - Basic Test Suite")
    print("=" * 50)
    
    success = True
    
    # Run tests
    success &= test_basic_fusion()
    success &= test_edge_cases()
    success &= test_validation()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print("=" * 50)
    
    sys.exit(0 if success else 1)
