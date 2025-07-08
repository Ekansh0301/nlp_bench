#!/usr/bin/env python3
"""
Robust Positional Language Model implementation
Trains on EMILLE corpus (single file, tab-separated format)
Applies to HUTB sentences (CSV format)
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
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustPositionalLanguageModel:
    def __init__(self, 
                 n_position_buckets: int = 10,
                 min_word_freq: int = 5,
                 smoothing_alpha: float = 0.1,
                 use_pos_tags: bool = False):
        """
        Robust Positional Language Model with multiple improvements
        
        Args:
            n_position_buckets: Number of position buckets
            min_word_freq: Minimum word frequency to be included in vocab
            smoothing_alpha: Laplace smoothing parameter
            use_pos_tags: Whether to use POS tags in addition to words
        """
        self.n_buckets = n_position_buckets
        self.min_word_freq = min_word_freq
        self.smoothing_alpha = smoothing_alpha
        self.use_pos_tags = use_pos_tags
        
        # Core statistics
        self.word_position_counts = defaultdict(lambda: defaultdict(int))
        self.word_counts = defaultdict(int)
        self.position_totals = defaultdict(int)
        self.total_sentences = 0
        
        # Additional statistics for robustness
        self.position_length_stats = defaultdict(list)  # Track sentence lengths per position
        self.word_cooccurrence = defaultdict(lambda: defaultdict(int))  # Bigram counts
        self.pos_position_counts = defaultdict(lambda: defaultdict(int)) if use_pos_tags else None
        
        # Vocabulary and OOV handling
        self.vocab = set()
        self.oov_token = '<OOV>'
        self.filtered_vocab = set()  # After frequency filtering
        
    def preprocess_sentence(self, sentence: str) -> List[str]:
        """Enhanced preprocessing for Hindi sentences"""
        # Remove sentence-final punctuation
        sentence = re.sub(r'[।॥\.\!\?]+\s*$', '', sentence.strip())
        
        # Remove other punctuation but keep Hindi characters
        sentence = re.sub(r'[,;:\'"()\[\]{}]', ' ', sentence)
        
        # Normalize whitespace
        sentence = re.sub(r'\s+', ' ', sentence)
        
        # Split on spaces
        words = sentence.split()
        
        # Filter out empty tokens
        words = [w for w in words if w.strip()]
        
        return words
    
    def train(self, sentences: List[str], 
              sample_size: Optional[int] = None):
        """
        Train the model with robust statistics collection
        
        Args:
            sentences: List of sentences
            sample_size: If provided, randomly sample this many sentences
        """
        logger.info(f"Starting training on {len(sentences)} sentences...")
        
        # Sample if requested
        if sample_size and sample_size < len(sentences):
            import random
            indices = random.sample(range(len(sentences)), sample_size)
            sentences = [sentences[i] for i in indices]
            logger.info(f"Sampled {sample_size} sentences for training")
        
        # First pass: collect word frequencies
        logger.info("First pass: collecting word frequencies...")
        word_freq_counter = Counter()
        
        for sentence in tqdm(sentences, desc="Counting words"):
            words = self.preprocess_sentence(sentence)
            word_freq_counter.update(words)
        
        # Filter vocabulary by frequency
        self.vocab = set(word_freq_counter.keys())
        self.filtered_vocab = {word for word, freq in word_freq_counter.items() 
                              if freq >= self.min_word_freq}
        self.filtered_vocab.add(self.oov_token)
        
        logger.info(f"Vocabulary size: {len(self.vocab)} -> {len(self.filtered_vocab)} after filtering")
        
        # Second pass: collect position statistics
        logger.info("Second pass: collecting position statistics...")
        
        for sent_idx, sentence in enumerate(tqdm(sentences, desc="Training")):
            words = self.preprocess_sentence(sentence)
            if not words or len(words) < 2:  # Skip very short sentences
                continue
                
            sent_length = len(words)
            self.total_sentences += 1
            
            for i, word in enumerate(words):
                # Map OOV words
                word_key = word if word in self.filtered_vocab else self.oov_token
                
                # Calculate position bucket
                relative_pos = i / sent_length
                pos_bucket = min(int(relative_pos * self.n_buckets), self.n_buckets - 1)
                
                # Update counts
                self.word_position_counts[word_key][pos_bucket] += 1
                self.word_counts[word_key] += 1
                self.position_totals[pos_bucket] += 1
                
                # Track sentence length distribution per position
                self.position_length_stats[pos_bucket].append(sent_length)
                
                # Collect bigram statistics
                if i > 0:
                    prev_word = words[i-1] if words[i-1] in self.filtered_vocab else self.oov_token
                    self.word_cooccurrence[prev_word][word_key] += 1
        
        logger.info(f"Training complete. Processed {self.total_sentences} sentences")
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute derived statistics for better probability estimation"""
        # Compute average sentence lengths per position bucket
        self.avg_sent_length_per_position = {}
        for pos_bucket, lengths in self.position_length_stats.items():
            self.avg_sent_length_per_position[pos_bucket] = np.mean(lengths) if lengths else 0
        
        # Compute position priors
        total_positions = sum(self.position_totals.values())
        self.position_priors = {
            pos: count / total_positions 
            for pos, count in self.position_totals.items()
        }
    
    def get_position_probability(self, word: str, position_bucket: int) -> float:
        """
        Get probability with multiple smoothing strategies
        """
        # Map OOV words
        word_key = word if word in self.filtered_vocab else self.oov_token
        
        # Base count
        count = self.word_position_counts[word_key][position_bucket]
        position_total = self.position_totals[position_bucket]
        
        # Handle empty positions
        if position_total == 0:
            return 1.0 / len(self.filtered_vocab)
        
        # Method 1: Laplace smoothing
        laplace_prob = (count + self.smoothing_alpha) / (position_total + self.smoothing_alpha * len(self.filtered_vocab))
        
        # Method 2: Interpolation with position prior
        position_prior = self.position_priors.get(position_bucket, 1.0 / self.n_buckets)
        word_prob = self.word_counts[word_key] / sum(self.word_counts.values())
        
        # Weighted combination
        prob = 0.8 * laplace_prob + 0.2 * (position_prior * word_prob)
        
        return max(prob, 1e-10)
    
    def score_sentence(self, sentence: str, return_details: bool = False) -> float:
        """
        Score sentence with optional detailed scoring breakdown
        """
        words = self.preprocess_sentence(sentence)
        if not words:
            return float('-inf')
        
        sent_length = len(words)
        scores = []
        details = []
        
        for i, word in enumerate(words):
            # Calculate position bucket
            relative_pos = i / sent_length
            pos_bucket = min(int(relative_pos * self.n_buckets), self.n_buckets - 1)
            
            # Get probability
            prob = self.get_position_probability(word, pos_bucket)
            score = log(prob)
            scores.append(score)
            
            if return_details:
                details.append({
                    'word': word,
                    'position': i,
                    'bucket': pos_bucket,
                    'probability': prob,
                    'score': score
                })
        
        # Aggregate score (normalized by length)
        total_score = sum(scores) / sent_length if sent_length > 0 else float('-inf')
        
        if return_details:
            return total_score, details
        return total_score
    
    def save_model(self, filepath: str):
        """Save model with all statistics"""
        model_data = {
            'config': {
                'n_buckets': self.n_buckets,
                'min_word_freq': self.min_word_freq,
                'smoothing_alpha': self.smoothing_alpha,
                'use_pos_tags': self.use_pos_tags
            },
            'stats': {
                'word_position_counts': dict(self.word_position_counts),
                'word_counts': dict(self.word_counts),
                'position_totals': dict(self.position_totals),
                'filtered_vocab': list(self.filtered_vocab),
                'total_sentences': self.total_sentences,
                'position_priors': self.position_priors,
                'avg_sent_length_per_position': self.avg_sent_length_per_position
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model with all statistics"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore config
        self.n_buckets = model_data['config']['n_buckets']
        self.min_word_freq = model_data['config']['min_word_freq']
        self.smoothing_alpha = model_data['config']['smoothing_alpha']
        self.use_pos_tags = model_data['config']['use_pos_tags']
        
        # Restore stats
        stats = model_data['stats']
        self.word_position_counts = defaultdict(lambda: defaultdict(int), stats['word_position_counts'])
        self.word_counts = defaultdict(int, stats['word_counts'])
        self.position_totals = defaultdict(int, stats['position_totals'])
        self.filtered_vocab = set(stats['filtered_vocab'])
        self.total_sentences = stats['total_sentences']
        self.position_priors = stats['position_priors']
        self.avg_sent_length_per_position = stats['avg_sent_length_per_position']
        
        logger.info(f"Model loaded from {filepath}")


def load_emille_corpus_single_file(emille_file_path: str, 
                                  encoding: str = 'utf-8',
                                  max_sentences: Optional[int] = 1000000) -> List[str]:
    """
    Load EMILLE corpus from a single tab-separated file
    Format: source_id<TAB>line_number<TAB>hindi_text
    """
    sentences = []
    lines_processed = 0
    errors = 0
    
    logger.info(f"Loading EMILLE from {emille_file_path}")
    
    try:
        with open(emille_file_path, 'r', encoding=encoding) as f:
            for line in tqdm(f, desc="Loading EMILLE sentences"):
                lines_processed += 1
                line = line.strip()
                
                if not line:
                    continue
                
                # Split by tab
                parts = line.split('\t')
                
                # Check if valid format (3 columns)
                if len(parts) >= 3:
                    # Extract Hindi text (3rd column)
                    hindi_text = parts[2]
                    
                    # Clean up common artifacts
                    hindi_text = hindi_text.replace('~', '')  # Remove ~ artifacts
                    hindi_text = hindi_text.strip()
                    
                    # Basic filtering
                    words = hindi_text.split()
                    if len(words) > 3 and len(words) < 100:  # Reasonable sentence length
                        sentences.append(hindi_text)
                        
                        if max_sentences and len(sentences) >= max_sentences:
                            logger.info(f"Reached max sentences limit: {max_sentences}")
                            break
                else:
                    errors += 1
                    if errors < 10:  # Show first few errors
                        logger.debug(f"Invalid line format: {line[:100]}...")
                        
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise
    
    logger.info(f"Processed {lines_processed} lines, extracted {len(sentences)} valid sentences ({errors} errors)")
    return sentences


def process_hutb_with_plm(hutb_csv: str, 
                         plm_model_path: str,
                         output_csv: str = 'hutb_plm_scores_robust.csv'):
    """
    Score HUTB sentences using trained PLM model
    """
    # Load model
    plm = RobustPositionalLanguageModel()
    plm.load_model(plm_model_path)
    
    # Load HUTB data
    logger.info(f"Loading HUTB sentences from {hutb_csv}")
    df = pd.read_csv(hutb_csv)
    
    # Score sentences with progress bar
    logger.info("Scoring sentences...")
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        score = plm.score_sentence(row['Sentences'])
        scores.append(score)
    
    df['positional_lm_score'] = scores
    
    # Calculate features for classification
    logger.info("Calculating pairwise features...")
    df['base_id'] = df['Sentence ID'].str.replace(r'\.\d+$', '', regex=True)
    
    results = []
    for base_id, group in df.groupby('base_id'):
        ref_row = group[group['Sentence ID'].str.endswith('.0')]
        if len(ref_row) == 0:
            continue
            
        ref_score = ref_row['positional_lm_score'].iloc[0]
        
        for idx, row in group.iterrows():
            results.append({
                'sentence_id': row['Sentence ID'],
                'sentence': row['Sentences'],
                'positional_lm_score': row['positional_lm_score'],
                'score_diff_from_ref': row['positional_lm_score'] - ref_score,
                'is_reference': row['Sentence ID'].endswith('.0')
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    
    # Print statistics
    logger.info("\nStatistics:")
    ref_scores = results_df[results_df['is_reference']]['positional_lm_score']
    var_scores = results_df[~results_df['is_reference']]['positional_lm_score']
    
    print(f"Reference sentences - Mean: {ref_scores.mean():.4f}, Std: {ref_scores.std():.4f}")
    print(f"Variant sentences - Mean: {var_scores.mean():.4f}, Std: {var_scores.std():.4f}")
    print(f"Average difference: {(ref_scores.mean() - var_scores.mean()):.4f}")
    
    # Show position preferences for common words
    print("\nPosition preferences for common Hindi words:")
    common_words = ['है', 'के', 'में', 'की', 'से', 'और', 'को', 'का', 'हैं', 'ने', 'यह', 'कि']
    
    for word in common_words:
        if word in plm.word_position_counts:
            positions = plm.word_position_counts[word]
            total = sum(positions.values())
            if total > 0:
                most_common = max(positions, key=positions.get)
                percentage = (positions[most_common] / total) * 100
                print(f"{word}: position {most_common}/{plm.n_buckets} ({percentage:.1f}% of occurrences)")
    
    return results_df


if __name__ == "__main__":
    # Configuration
    EMILLE_FILE = "Written_Data.txt"  # Update with full path
    MODEL_PATH = "emille_plm_robust.pkl"
    HUTB_CSV = "hutb-sentences.csv"
    OUTPUT_CSV = "hutb_plm_scores_robust.csv"
    
    # Step 1: Train PLM on EMILLE (run once)
    if not os.path.exists(MODEL_PATH):
        # Load EMILLE sentences
        logger.info("Loading EMILLE corpus...")
        emille_sentences = load_emille_corpus_single_file(
            EMILLE_FILE,
            max_sentences=1000000  # Use up to 1M sentences like the paper
        )
        
        logger.info(f"Loaded {len(emille_sentences)} sentences from EMILLE")
        
        # Show sample sentences
        logger.info("\nSample sentences:")
        for i in range(min(5, len(emille_sentences))):
            logger.info(f"{i+1}: {emille_sentences[i][:100]}...")
        
        # Quick statistics
        sentence_lengths = [len(s.split()) for s in emille_sentences[:10000]]
        logger.info(f"\nAverage sentence length (first 10k): {np.mean(sentence_lengths):.1f} words")
        logger.info(f"Min length: {min(sentence_lengths)}, Max length: {max(sentence_lengths)}")
        
        # Train PLM
        logger.info("\nTraining Positional Language Model...")
        plm = RobustPositionalLanguageModel(
            n_position_buckets=10,
            min_word_freq=5,      # Words must appear at least 5 times
            smoothing_alpha=0.1   # Laplace smoothing
        )
        
        plm.train(emille_sentences)
        plm.save_model(MODEL_PATH)
        
        # Print model statistics
        logger.info(f"\nModel Statistics:")
        logger.info(f"Vocabulary size (filtered): {len(plm.filtered_vocab)}")
        logger.info(f"Total sentences processed: {plm.total_sentences}")
        logger.info(f"Position bucket totals: {dict(plm.position_totals)}")

        # Show top words per position
        for pos in range(plm.n_buckets):
            logger.info(f"\nTop words at position {pos}:")
            word_counts_at_pos = [(word, plm.word_position_counts[word][pos]) 
                                for word, positions in plm.word_position_counts.items() 
                                if positions[pos] > 0]
            word_counts_at_pos.sort(key=lambda x: x[1], reverse=True)
            for word, count in word_counts_at_pos[:5]:
                logger.info(f"  {word}: {count} times")
    
    # Step 2: Apply to HUTB
    logger.info(f"\nApplying PLM to HUTB sentences...")
    results = process_hutb_with_plm(
        hutb_csv=HUTB_CSV,
        plm_model_path=MODEL_PATH,
        output_csv=OUTPUT_CSV
    )
    
    logger.info(f"\nScores saved to {OUTPUT_CSV}")
    logger.info("Processing complete!")