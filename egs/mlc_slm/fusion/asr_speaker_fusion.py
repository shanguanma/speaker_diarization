"""
ASR Speaker Fusion Module

This module implements the fusion of ASR word-level timestamps with speaker labels.
The algorithm works as follows:
1. If a word segment overlaps with at least one speaker segment, associate the word
   with the speaker that has the biggest temporal overlap
2. Otherwise, if the word segment doesn't overlap with any speaker segment, associate
   it with the speaker that has the smallest temporal distance based on segment boundaries

Author: AI Assistant
Date: 2024
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WordSegment:
    """Represents a word segment from ASR output"""
    word: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None
    
    def __post_init__(self):
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be less than end_time")


@dataclass
class SpeakerSegment:
    """Represents a speaker segment from speaker diarization output"""
    speaker_id: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None
    
    def __post_init__(self):
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be less than end_time")


@dataclass
class FusedWord:
    """Represents a word with associated speaker information"""
    word: str
    start_time: float
    end_time: float
    speaker_id: str
    confidence: Optional[float] = None
    fusion_method: str = "overlap"  # 'overlap' or 'distance'


class ASRSpeakerFusion:
    """
    Main class for fusing ASR word-level timestamps with speaker labels
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize the fusion module
        
        Args:
            n_workers: Number of worker processes/threads for parallel processing
                       If None, uses CPU count
        """
        self.n_workers = n_workers or mp.cpu_count()
        logger.info(f"Initializing ASR Speaker Fusion with {self.n_workers} workers")
    
    def calculate_overlap(self, word: WordSegment, speaker: SpeakerSegment) -> float:
        """
        Calculate temporal overlap between a word and speaker segment
        
        Args:
            word: Word segment
            speaker: Speaker segment
            
        Returns:
            Overlap duration in seconds
        """
        overlap_start = max(word.start_time, speaker.start_time)
        overlap_end = min(word.end_time, speaker.end_time)
        return max(0, overlap_end - overlap_start)
    
    def calculate_temporal_distance(self, word: WordSegment, speaker: SpeakerSegment) -> float:
        """
        Calculate temporal distance between a word and speaker segment
        
        Args:
            word: Word segment
            speaker: Speaker segment
            
        Returns:
            Temporal distance in seconds
        """
        if word.end_time < speaker.start_time:
            return speaker.start_time - word.end_time
        elif word.start_time > speaker.end_time:
            return word.start_time - speaker.end_time
        else:
            return 0.0
    
    def find_best_speaker_by_overlap(self, word: WordSegment, speakers: List[SpeakerSegment]) -> Optional[Tuple[str, float]]:
        """
        Find the speaker with the biggest temporal overlap with the word
        
        Args:
            word: Word segment
            speakers: List of speaker segments
            
        Returns:
            Tuple of (speaker_id, overlap_duration) or None if no overlap
        """
        best_speaker = None
        best_overlap = 0.0
        
        for speaker in speakers:
            overlap = self.calculate_overlap(word, speaker)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker.speaker_id
        
        return (best_speaker, best_overlap) if best_speaker else None
    
    def find_best_speaker_by_distance(self, word: WordSegment, speakers: List[SpeakerSegment]) -> Tuple[str, float]:
        """
        Find the speaker with the smallest temporal distance to the word
        
        Args:
            word: Word segment
            speakers: List of speaker segments
            
        Returns:
            Tuple of (speaker_id, distance)
        """
        best_speaker = None
        best_distance = float('inf')
        
        for speaker in speakers:
            distance = self.calculate_temporal_distance(word, speaker)
            if distance < best_distance:
                best_distance = distance
                best_speaker = speaker.speaker_id
        
        return best_speaker, best_distance
    
    def fuse_single_word(self, word: WordSegment, speakers: List[SpeakerSegment]) -> FusedWord:
        """
        Fuse a single word with speaker information
        
        Args:
            word: Word segment
            speakers: List of speaker segments
            
        Returns:
            Fused word with speaker information
        """
        # First, try to find overlap
        overlap_result = self.find_best_speaker_by_overlap(word, speakers)
        
        if overlap_result and overlap_result[1] > 0:
            # Word overlaps with at least one speaker segment
            speaker_id, overlap = overlap_result
            fusion_method = 'overlap'
        else:
            # No overlap, find closest speaker by temporal distance
            speaker_id, distance = self.find_best_speaker_by_distance(word, speakers)
            fusion_method = 'distance'
        
        return FusedWord(
            word=word.word,
            start_time=word.start_time,
            end_time=word.end_time,
            speaker_id=speaker_id,
            confidence=word.confidence,
            fusion_method=fusion_method
        )
    
    def fuse_words_sequential(self, words: List[WordSegment], speakers: List[SpeakerSegment]) -> List[FusedWord]:
        """
        Fuse words with speakers using sequential processing
        
        Args:
            words: List of word segments
            speakers: List of speaker segments
            
        Returns:
            List of fused words
        """
        logger.info(f"Processing {len(words)} words sequentially")
        start_time = time.time()
        
        fused_words = []
        for word in words:
            fused_word = self.fuse_single_word(word, speakers)
            fused_words.append(fused_word)
        
        processing_time = time.time() - start_time
        logger.info(f"Sequential processing completed in {processing_time:.4f} seconds")
        
        return fused_words
    
    def fuse_words_parallel(self, words: List[WordSegment], speakers: List[SpeakerSegment], 
                           use_processes: bool = True) -> List[FusedWord]:
        """
        Fuse words with speakers using parallel processing
        
        Args:
            words: List of word segments
            speakers: List of speaker segments
            use_processes: If True, use ProcessPoolExecutor, else ThreadPoolExecutor
            
        Returns:
            List of fused words
        """
        logger.info(f"Processing {len(words)} words in parallel using {'processes' if use_processes else 'threads'}")
        start_time = time.time()
        
        # Create partial function with speakers fixed
        fuse_func = partial(self.fuse_single_word, speakers=speakers)
        
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.n_workers) as executor:
            fused_words = list(executor.map(fuse_func, words))
        
        processing_time = time.time() - start_time
        logger.info(f"Parallel processing completed in {processing_time:.4f} seconds")
        
        return fused_words
    
    def fuse(self, words: List[WordSegment], speakers: List[SpeakerSegment], 
             parallel: bool = True, use_processes: bool = True) -> List[FusedWord]:
        """
        Main fusion method
        
        Args:
            words: List of word segments
            speakers: List of speaker segments
            parallel: Whether to use parallel processing
            use_processes: If parallel=True, whether to use processes or threads
            
        Returns:
            List of fused words
        """
        if not words:
            logger.warning("No words provided for fusion")
            return []
        
        if not speakers:
            logger.warning("No speakers provided for fusion")
            return []
        
        logger.info(f"Starting fusion: {len(words)} words, {len(speakers)} speaker segments")
        
        if parallel:
            return self.fuse_words_parallel(words, speakers, use_processes)
        else:
            return self.fuse_words_sequential(words, speakers)
    
    def get_fusion_statistics(self, fused_words: List[FusedWord]) -> Dict:
        """
        Get statistics about the fusion results
        
        Args:
            fused_words: List of fused words
            
        Returns:
            Dictionary containing fusion statistics
        """
        if not fused_words:
            return {}
        
        total_words = len(fused_words)
        overlap_fused = sum(1 for w in fused_words if w.fusion_method == 'overlap')
        distance_fused = sum(1 for w in fused_words if w.fusion_method == 'distance')
        
        speaker_counts = {}
        for word in fused_words:
            speaker_counts[word.speaker_id] = speaker_counts.get(word.speaker_id, 0) + 1
        
        return {
            'total_words': total_words,
            'overlap_fused': overlap_fused,
            'distance_fused': distance_fused,
            'overlap_percentage': (overlap_fused / total_words) * 100,
            'distance_percentage': (distance_fused / total_words) * 100,
            'speaker_distribution': speaker_counts,
            'unique_speakers': len(speaker_counts)
        }


def create_sample_data() -> Tuple[List[WordSegment], List[SpeakerSegment]]:
    """
    Create sample data for testing
    
    Returns:
        Tuple of (words, speakers) for testing
    """
    # Sample word segments (ASR output)
    words = [
        WordSegment("Hello", 0.0, 0.5, 0.95),
        WordSegment("world", 0.5, 1.0, 0.92),
        WordSegment("how", 1.0, 1.3, 0.88),
        WordSegment("are", 1.3, 1.5, 0.90),
        WordSegment("you", 1.5, 1.8, 0.87),
        WordSegment("today", 1.8, 2.2, 0.89),
    ]
    
    # Sample speaker segments (speaker diarization output)
    speakers = [
        SpeakerSegment("speaker_1", 0.0, 1.2, 0.98),
        SpeakerSegment("speaker_2", 1.2, 2.5, 0.97),
    ]
    
    return words, speakers


def run_performance_test():
    """
    Run performance comparison test between sequential and parallel processing
    """
    print("=" * 60)
    print("ASR Speaker Fusion Performance Test")
    print("=" * 60)
    
    # Create sample data
    words, speakers = create_sample_data()
    
    # Create fusion instance
    fusion = ASRSpeakerFusion()
    
    # Test sequential processing
    print("\n1. Sequential Processing Test:")
    fused_sequential = fusion.fuse(words, speakers, parallel=False)
    
    # Test parallel processing with threads
    print("\n2. Parallel Processing Test (Threads):")
    fused_parallel_threads = fusion.fuse(words, speakers, parallel=True, use_processes=False)
    
    # Test parallel processing with processes
    print("\n3. Parallel Processing Test (Processes):")
    fused_parallel_processes = fusion.fuse(words, speakers, parallel=True, use_processes=True)
    
    # Verify results are the same
    print("\n4. Result Verification:")
    sequential_speakers = [w.speaker_id for w in fused_sequential]
    thread_speakers = [w.speaker_id for w in fused_parallel_threads]
    process_speakers = [w.speaker_id for w in fused_parallel_processes]
    
    print(f"Sequential == Threads: {sequential_speakers == thread_speakers}")
    print(f"Sequential == Processes: {sequential_speakers == process_speakers}")
    
    # Display fusion statistics
    print("\n5. Fusion Statistics:")
    stats = fusion.get_fusion_statistics(fused_sequential)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Display detailed results
    print("\n6. Detailed Fusion Results:")
    for i, word in enumerate(fused_sequential):
        print(f"  Word {i+1}: '{word.word}' -> {word.speaker_id} ({word.fusion_method})")
    
    print("\n" + "=" * 60)


def run_scalability_test():
    """
    Run scalability test with larger datasets
    """
    print("\n" + "=" * 60)
    print("Scalability Test with Larger Datasets")
    print("=" * 60)
    
    # Create larger datasets
    np.random.seed(42)
    
    # Generate 1000 word segments
    n_words = 1000
    words = []
    current_time = 0.0
    
    for i in range(n_words):
        word_duration = np.random.uniform(0.1, 0.8)
        word = WordSegment(
            word=f"word_{i}",
            start_time=current_time,
            end_time=current_time + word_duration,
            confidence=np.random.uniform(0.7, 1.0)
        )
        words.append(word)
        current_time += word_duration + np.random.uniform(0.05, 0.2)
    
    # Generate 50 speaker segments
    n_speakers = 50
    speakers = []
    current_time = 0.0
    
    for i in range(n_speakers):
        speaker_duration = np.random.uniform(2.0, 8.0)
        speaker = SpeakerSegment(
            speaker_id=f"speaker_{i % 5}",  # 5 unique speakers
            start_time=current_time,
            end_time=current_time + speaker_duration,
            confidence=np.random.uniform(0.8, 1.0)
        )
        speakers.append(speaker)
        current_time += speaker_duration + np.random.uniform(0.5, 2.0)
    
    print(f"Generated dataset: {len(words)} words, {len(speakers)} speaker segments")
    
    # Test different processing methods
    fusion = ASRSpeakerFusion()
    
    # Sequential
    start_time = time.time()
    fused_seq = fusion.fuse(words, speakers, parallel=False)
    seq_time = time.time() - start_time
    
    # Parallel with threads
    start_time = time.time()
    fused_threads = fusion.fuse(words, speakers, parallel=True, use_processes=False)
    thread_time = time.time() - start_time
    
    # Parallel with processes
    start_time = time.time()
    fused_processes = fusion.fuse(words, speakers, parallel=True, use_processes=True)
    process_time = time.time() - start_time
    
    print(f"\nPerformance Results:")
    print(f"  Sequential:     {seq_time:.4f}s")
    print(f"  Threads:        {thread_time:.4f}s")
    print(f"  Processes:      {process_time:.4f}s")
    print(f"  Speedup (threads): {seq_time/thread_time:.2f}x")
    print(f"  Speedup (processes): {seq_time/process_time:.2f}x")
    
    # Verify results
    seq_speakers = [w.speaker_id for w in fused_seq]
    thread_speakers = [w.speaker_id for w in fused_threads]
    process_speakers = [w.speaker_id for w in fused_processes]
    
    print(f"\nResult Verification:")
    print(f"  Sequential == Threads: {seq_speakers == thread_speakers}")
    print(f"  Sequential == Processes: {seq_speakers == process_speakers}")
    
    # Display statistics
    stats = fusion.get_fusion_statistics(fused_seq)
    print(f"\nFusion Statistics:")
    print(f"  Total words: {stats['total_words']}")
    print(f"  Overlap fused: {stats['overlap_fused']} ({stats['overlap_percentage']:.1f}%)")
    print(f"  Distance fused: {stats['distance_fused']} ({stats['distance_percentage']:.1f}%)")
    print(f"  Unique speakers: {stats['unique_speakers']}")


if __name__ == "__main__":
    """
    Main execution block for testing and demonstration
    """
    try:
        # Run basic performance test
        run_performance_test()
        
        # Run scalability test
        run_scalability_test()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


