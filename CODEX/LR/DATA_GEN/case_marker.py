#!/usr/bin/env python3
"""
Case Marker Transition Model
Trains on EMILLE corpus to learn case marker transition probabilities
Applies to HUTB sentences for classification
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import pickle
from math import log
import re
import os
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CaseMarkerTransitionModel:
    def __init__(self, smoothing_alpha: float = 0.1):
        """
        Initialize Case Marker Transition Model
        
        Args:
            smoothing_alpha: Laplace smoothing parameter
        """
        self.smoothing_alpha = smoothing_alpha
        
        # Hindi case markers (from paper)
        self.case_markers = {
            'ne': 'ERG',  # ergative
            'ko': 'ACC/DAT',  # accusative/dative
            'se': 'INS',  # instrumental
            'ka': 'GEN',  # genitive
            'ki': 'GEN',  # genitive (feminine)
            'ke': 'GEN',  # genitive (plural/honorific)
            'me': 'LOC',  # locative (in Hindi script: में)
            'par': 'LOC', # locative (on)
            'tak': 'LOC', # locative (until)
            'men': 'LOC', # alternate spelling of में
            'mein': 'LOC', # another alternate
            'में': 'LOC',  # Hindi script version
            'पर': 'LOC',  # Hindi script version
            'से': 'INS',  # Hindi script version
            'को': 'ACC/DAT', # Hindi script version
            'ने': 'ERG',  # Hindi script version
            'का': 'GEN',  # Hindi script version
            'की': 'GEN',  # Hindi script version
            'के': 'GEN'   # Hindi script version
        }
        
        # Statistics
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.marker_counts = defaultdict(int)
        self.total_transitions = 0
        self.total_sentences = 0
        
        # Additional statistics
        self.position_counts = defaultdict(lambda: defaultdict(int))  # Position preferences
        self.distance_counts = defaultdict(list)  # Distance between same markers
        
    def preprocess_sentence(self, sentence: str) -> List[str]:
        """Clean and tokenize Hindi sentence"""
        # Remove sentence-final punctuation
        sentence = re.sub(r'[।॥\.\!\?]+\s*$', '', sentence.strip())
        
        # Normalize whitespace
        sentence = re.sub(r'\s+', ' ', sentence)
        
        # Split on spaces
        words = sentence.split()
        
        return words
    
    def extract_case_markers(self, words: List[str]) -> List[Tuple[int, str]]:
        """
        Extract case markers and their positions
        Returns: List of (position, marker) tuples
        """
        markers = []
        
        for i, word in enumerate(words):
            # Check if word is a case marker
            if word.lower() in self.case_markers:
                markers.append((i, word.lower()))
        
        return markers
    
    def train(self, sentences: List[str], sample_size: Optional[int] = None):
        """
        Train the model on EMILLE corpus
        
        Args:
            sentences: List of Hindi sentences
            sample_size: If provided, randomly sample this many sentences
        """
        logger.info(f"Starting training on {len(sentences)} sentences...")
        
        # Sample if requested
        if sample_size and sample_size < len(sentences):
            import random
            indices = random.sample(range(len(sentences)), sample_size)
            sentences = [sentences[i] for i in indices]
            logger.info(f"Sampled {sample_size} sentences for training")
        
        for sent_idx, sentence in enumerate(tqdm(sentences, desc="Training")):
            if sent_idx % 10000 == 0:
                logger.info(f"Processed {sent_idx} sentences...")
            
            words = self.preprocess_sentence(sentence)
            if not words:
                continue
            
            self.total_sentences += 1
            markers = self.extract_case_markers(words)
            
            if len(markers) < 2:
                continue
            
            # Collect transition statistics
            for i in range(len(markers) - 1):
                pos1, marker1 = markers[i]
                pos2, marker2 = markers[i + 1]
                
                # Transition counts
                self.transition_counts[marker1][marker2] += 1
                self.marker_counts[marker1] += 1
                self.total_transitions += 1
                
                # Position statistics (relative position in sentence)
                rel_pos1 = pos1 / len(words)
                self.position_counts[marker1][int(rel_pos1 * 10)] += 1
                
                # Distance between markers
                distance = pos2 - pos1
                self.distance_counts[(marker1, marker2)].append(distance)
        
        # Add final marker counts
        for markers_list in self.extract_case_markers_from_sentences(sentences[-1000:]):
            if markers_list:
                _, last_marker = markers_list[-1]
                self.marker_counts[last_marker] += 1
        
        logger.info(f"Training complete. Processed {self.total_sentences} sentences")
        logger.info(f"Found {self.total_transitions} case marker transitions")
        
        # Compute statistics
        self._compute_statistics()
    
    def extract_case_markers_from_sentences(self, sentences: List[str]) -> List[List[Tuple[int, str]]]:
        """Helper to extract markers from multiple sentences"""
        all_markers = []
        for sent in sentences:
            words = self.preprocess_sentence(sent)
            markers = self.extract_case_markers(words)
            all_markers.append(markers)
        return all_markers
    
    def _compute_statistics(self):
        """Compute derived statistics"""
        # Compute average distances
        self.avg_distances = {}
        for pair, distances in self.distance_counts.items():
            self.avg_distances[pair] = np.mean(distances) if distances else 0
        
        # Most common transitions
        logger.info("\nMost common case marker transitions:")
        all_transitions = []
        for m1, transitions in self.transition_counts.items():
            for m2, count in transitions.items():
                all_transitions.append((f"{m1}→{m2}", count))
        
        all_transitions.sort(key=lambda x: x[1], reverse=True)
        for trans, count in all_transitions[:10]:
            logger.info(f"  {trans}: {count} times")
    
    def get_transition_probability(self, marker1: str, marker2: str) -> float:
        """
        Get probability of marker2 following marker1
        Uses Laplace smoothing
        """
        marker1 = marker1.lower()
        marker2 = marker2.lower()
        
        # Only consider known markers
        if marker1 not in self.case_markers or marker2 not in self.case_markers:
            return 1e-6
        
        count = self.transition_counts[marker1][marker2]
        total = self.marker_counts[marker1]
        
        if total == 0:
            return 1.0 / len(self.case_markers)
        
        # Laplace smoothing
        prob = (count + self.smoothing_alpha) / (total + self.smoothing_alpha * len(self.case_markers))
        
        return prob
    
    def score_sentence(self, sentence: str, return_details: bool = False) -> float:
        """
        Score sentence based on case marker transitions
        Higher score = more natural transitions
        """
        words = self.preprocess_sentence(sentence)
        if not words:
            return 0.0
        
        markers = self.extract_case_markers(words)
        if len(markers) < 2:
            return 0.0  # No transitions to score
        
        scores = []
        details = []
        
        for i in range(len(markers) - 1):
            pos1, marker1 = markers[i]
            pos2, marker2 = markers[i + 1]
            
            # Transition probability
            trans_prob = self.get_transition_probability(marker1, marker2)
            score = log(trans_prob)
            
            # Distance penalty (optional)
            distance = pos2 - pos1
            expected_dist = self.avg_distances.get((marker1, marker2), 5)
            distance_penalty = -abs(distance - expected_dist) / 10
            
            total_score = score + distance_penalty
            scores.append(total_score)
            
            if return_details:
                details.append({
                    'transition': f"{marker1}→{marker2}",
                    'probability': trans_prob,
                    'score': score,
                    'distance': distance,
                    'distance_penalty': distance_penalty,
                    'total_score': total_score
                })
        
        # Average score
        final_score = np.mean(scores) if scores else 0.0
        
        if return_details:
            return final_score, details
        return final_score
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'case_markers': self.case_markers,
            'transition_counts': dict(self.transition_counts),
            'marker_counts': dict(self.marker_counts),
            'total_transitions': self.total_transitions,
            'total_sentences': self.total_sentences,
            'avg_distances': self.avg_distances,
            'smoothing_alpha': self.smoothing_alpha
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.case_markers = model_data['case_markers']
        self.transition_counts = defaultdict(lambda: defaultdict(int), model_data['transition_counts'])
        self.marker_counts = defaultdict(int, model_data['marker_counts'])
        self.total_transitions = model_data['total_transitions']
        self.total_sentences = model_data['total_sentences']
        self.avg_distances = model_data['avg_distances']
        self.smoothing_alpha = model_data['smoothing_alpha']
        
        logger.info(f"Model loaded from {filepath}")


def load_emille_corpus(emille_file_path: str, 
                      encoding: str = 'utf-8',
                      max_sentences: Optional[int] = 1000000) -> List[str]:
    """
    Load EMILLE corpus from tab-separated file
    Format: source_id<TAB>line_number<TAB>hindi_text
    """
    sentences = []
    
    logger.info(f"Loading EMILLE from {emille_file_path}")
    
    try:
        with open(emille_file_path, 'r', encoding=encoding) as f:
            for line in tqdm(f, desc="Loading EMILLE sentences"):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 3:
                    hindi_text = parts[2]
                    hindi_text = hindi_text.replace('~', '')
                    hindi_text = hindi_text.strip()
                    
                    words = hindi_text.split()
                    if len(words) > 3 and len(words) < 100:
                        sentences.append(hindi_text)
                        
                        if max_sentences and len(sentences) >= max_sentences:
                            logger.info(f"Reached max sentences limit: {max_sentences}")
                            break
                            
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise
    
    logger.info(f"Loaded {len(sentences)} sentences from EMILLE")
    return sentences


def process_hutb_with_case_marker_model(hutb_csv: str, 
                                       model_path: str,
                                       output_csv: str = 'hutb_case_marker_scores.csv'):
    """
    Score HUTB sentences using trained case marker model
    """
    # Load model
    cmm = CaseMarkerTransitionModel()
    cmm.load_model(model_path)
    
    # Load HUTB data
    logger.info(f"Loading HUTB sentences from {hutb_csv}")
    df = pd.read_csv(hutb_csv)
    
    # Score sentences
    logger.info("Scoring sentences...")
    scores = []
    marker_counts = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        score = cmm.score_sentence(row['Sentences'])
        scores.append(score)
        
        # Also count markers for statistics
        words = cmm.preprocess_sentence(row['Sentences'])
        markers = cmm.extract_case_markers(words)
        marker_counts.append(len(markers))
    
    df['case_marker_score'] = scores
    df['num_case_markers'] = marker_counts
    
    # Calculate pairwise features
    logger.info("Calculating pairwise features...")
    df['base_id'] = df['Sentence ID'].str.replace(r'\.\d+$', '', regex=True)
    
    results = []
    for base_id, group in df.groupby('base_id'):
        ref_row = group[group['Sentence ID'].str.endswith('.0')]
        if len(ref_row) == 0:
            continue
        
        ref_score = ref_row['case_marker_score'].iloc[0]
        
        for idx, row in group.iterrows():
            results.append({
                'sentence_id': row['Sentence ID'],
                'sentence': row['Sentences'],
                'case_marker_score': row['case_marker_score'],
                'num_case_markers': row['num_case_markers'],
                'score_diff_from_ref': row['case_marker_score'] - ref_score,
                'is_reference': row['Sentence ID'].endswith('.0')
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    
    # Print statistics
    logger.info("\nStatistics:")
    ref_scores = results_df[results_df['is_reference']]['case_marker_score']
    var_scores = results_df[~results_df['is_reference']]['case_marker_score']
    
    print(f"Reference sentences - Mean: {ref_scores.mean():.4f}, Std: {ref_scores.std():.4f}")
    print(f"Variant sentences - Mean: {var_scores.mean():.4f}, Std: {var_scores.std():.4f}")
    print(f"Average difference: {(ref_scores.mean() - var_scores.mean()):.4f}")
    
    # Analyze specific transitions
    print("\nExample problematic transitions in variants:")
    for _, row in results_df[~results_df['is_reference']].sort_values('case_marker_score').head(5).iterrows():
        sent = row['sentence']
        score, details = cmm.score_sentence(sent, return_details=True)
        print(f"\nSentence (score={score:.4f}): {sent[:100]}...")
        if details:
            worst_transition = min(details, key=lambda x: x['total_score'])
            print(f"  Worst transition: {worst_transition['transition']} (prob={worst_transition['probability']:.4f})")
    
    return results_df


if __name__ == "__main__":
    # Configuration
    EMILLE_FILE = "Written_Data.txt"
    MODEL_PATH = "emille_case_marker_model.pkl"
    HUTB_CSV = "hutb-sentences.csv"
    OUTPUT_CSV = "hutb_case_marker_scores.csv"
    
    # Step 1: Train on EMILLE (run once)
    if not os.path.exists(MODEL_PATH):
        logger.info("Training Case Marker Transition Model on EMILLE...")
        
        # Load EMILLE
        emille_sentences = load_emille_corpus(EMILLE_FILE, max_sentences=1000000)
        
        # Train model
        cmm = CaseMarkerTransitionModel(smoothing_alpha=0.1)
        cmm.train(emille_sentences)
        cmm.save_model(MODEL_PATH)
        
        # Show learned patterns
        print("\nTop case marker transitions learned:")
        transitions = []
        for m1, trans in cmm.transition_counts.items():
            for m2, count in trans.items():
                prob = cmm.get_transition_probability(m1, m2)
                transitions.append((f"{m1}→{m2}", count, prob))
        
        transitions.sort(key=lambda x: x[1], reverse=True)
        for trans, count, prob in transitions[:20]:
            print(f"  {trans}: {count} times (p={prob:.4f})")
    
    # Step 2: Apply to HUTB
    logger.info("\nApplying Case Marker Model to HUTB...")
    results = process_hutb_with_case_marker_model(HUTB_CSV, MODEL_PATH, OUTPUT_CSV)
    
    logger.info(f"\nScores saved to {OUTPUT_CSV}")
    logger.info("Processing complete!")