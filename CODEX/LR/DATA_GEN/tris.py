#!/usr/bin/env python3
"""
KenLM-based Trigram Surprisal Calculation for Hindi
Handles both EMILLE corpus format and HUTB CSV format
Fast alternative to SRILM
"""

import kenlm
import pandas as pd
import numpy as np
import re
from math import log
from tqdm import tqdm
import argparse
import os

class KenLMSurprisal:
    def __init__(self, model_path):
        """Load KenLM model"""
        print(f"Loading KenLM model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = kenlm.Model(model_path)
        print(f"✓ Model loaded: {self.model.order}-gram model")
        # vocab_size not available in Python API
        
    def preprocess_sentence(self, sentence):
        """
        Preprocess Hindi sentence - same as SRILM version
        """
        # Remove extra whitespace
        sentence = sentence.strip()
        
        # Handle Hindi punctuation - add spaces
        sentence = re.sub(r'([।,?!])', r' \1 ', sentence)
        
        # Handle other punctuation
        sentence = re.sub(r'([.";:\'\(\)\[\]])', r' \1 ', sentence)
        
        # Normalize whitespace
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        return sentence
        
    def calculate_sentence_surprisal(self, sentence):
        """
        Calculate sentence-level surprisal
        Following same methodology as SRILM version:
        - Sum of per-word surprisals
        - Excluding punctuation
        """
        # Preprocess
        processed_sentence = self.preprocess_sentence(sentence)
        words = processed_sentence.split()
        
        total_surprisal = 0.0
        words_counted = 0
        
        # Get per-word scores from KenLM
        # full_scores returns: (log10_prob, ngram_length, is_oov)
        scores = list(self.model.full_scores(processed_sentence))
        
        for i, (log10_prob, ngram_length, is_oov) in enumerate(scores):
            if i < len(words):
                word = words[i]
                
                # Skip punctuation as per paper
                if word in ['।', ',', '?', '!', '.', '(', ')', '[', ']', '"', "'", '<s>', '</s>']:
                    continue
                
                # Convert log10 to natural log for surprisal
                # Surprisal = -log(p) = -log10(p) * log(10)
                surprisal = -log10_prob * log(10)
                total_surprisal += surprisal
                words_counted += 1
                
                # Debug info for first sentence (optional)
                if hasattr(self, 'debug') and self.debug and i < 5:
                    print(f"  Word: {word}, log10_prob: {log10_prob:.4f}, "
                          f"surprisal: {surprisal:.4f}, is_oov: {is_oov}")
        
        return total_surprisal
    
    def prepare_emille_corpus(self, emille_path, output_path, max_sentences=1000000):
        """
        Prepare EMILLE corpus (if needed)
        Handles format: doc_id<TAB>sent_num<TAB>text
        """
        print(f"\nPreparing EMILLE corpus...")
        print(f"Input: {emille_path}")
        
        sentence_count = 0
        
        with open(emille_path, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    if sentence_count >= max_sentences:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse EMILLE format
                    parts = line.split('\t')
                    if len(parts) != 3:
                        continue
                    
                    doc_id, sent_num, text = parts
                    
                    # Clean text
                    text = text.strip().strip("'\"")
                    processed_text = self.preprocess_sentence(text)
                    
                    # Skip very short/long sentences
                    words = processed_text.split()
                    if len(words) < 3 or len(words) > 100:
                        continue
                    
                    f_out.write(processed_text + '\n')
                    sentence_count += 1
                    
                    if sentence_count % 10000 == 0:
                        print(f"  Processed {sentence_count:,} sentences...")
        
        print(f"✓ Prepared {sentence_count:,} sentences")
        return output_path
    
    def calculate_surprisal_for_hutb(self, hutb_csv, output_csv):
        """
        Calculate surprisal for all HUTB sentences
        Input format: CSV with 'Sentence ID' and 'Sentences' columns
        """
        print(f"\nCalculating surprisal for HUTB sentences...")
        print(f"Input: {hutb_csv}")
        
        # Load data
        hutb_df = pd.read_csv(hutb_csv)
        print(f"Total sentences: {len(hutb_df):,}")
        
        # Check columns
        if 'Sentence ID' not in hutb_df.columns or 'Sentences' not in hutb_df.columns:
            raise ValueError("CSV must have 'Sentence ID' and 'Sentences' columns")
        
        surprisals = []
        errors = 0
        
        # Process with progress bar
        for idx, row in tqdm(hutb_df.iterrows(), total=len(hutb_df), 
                            desc="Calculating surprisals"):
            sentence_id = row['Sentence ID']
            sentence = row['Sentences']
            
            try:
                surprisal = self.calculate_sentence_surprisal(sentence)
                surprisals.append({
                    'sentence_id': sentence_id,
                    'sentence': sentence,
                    'trigram_surprisal': surprisal
                })
            except Exception as e:
                if errors < 5:  # Only print first few errors
                    print(f"\nError at {sentence_id}: {e}")
                surprisals.append({
                    'sentence_id': sentence_id,
                    'sentence': sentence,
                    'trigram_surprisal': np.nan
                })
                errors += 1
        
        # Create results dataframe
        results_df = pd.DataFrame(surprisals)
        
        # Save results
        results_df.to_csv(output_csv, index=False, encoding='utf-8')
        
        # Print statistics
        valid_surprisals = results_df['trigram_surprisal'].dropna()
        print(f"\n✓ Results saved to {output_csv}")
        print(f"\nStatistics:")
        print(f"  Total sentences: {len(results_df):,}")
        print(f"  Valid surprisals: {len(valid_surprisals):,}")
        print(f"  Errors: {errors}")
        if len(valid_surprisals) > 0:
            print(f"  Mean surprisal: {valid_surprisals.mean():.2f}")
            print(f"  Std deviation: {valid_surprisals.std():.2f}")
            print(f"  Min surprisal: {valid_surprisals.min():.2f}")
            print(f"  Max surprisal: {valid_surprisals.max():.2f}")
        
        return results_df
    
    def test_model(self, test_sentences=None):
        """Test model with a few sentences"""
        if test_sentences is None:
            test_sentences = [
                "यह एक परीक्षण वाक्य है।",
                "भवानीनन्दन दफ्तर से आते ही कमरे में बिछी चटाई पर पसर गये।",
                "मैं आज बाजार जाऊंगा।"
            ]
        
        print("\nTesting model with sample sentences:")
        for sent in test_sentences:
            surprisal = self.calculate_sentence_surprisal(sent)
            print(f"Sentence: {sent}")
            print(f"Surprisal: {surprisal:.2f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate trigram surprisal using KenLM (fast alternative to SRILM)"
    )
    parser.add_argument("--model", default="hindi_trigram.klm",
                       help="Path to KenLM binary model file (.klm)")
    parser.add_argument("--hutb", default="hutb-sentences.csv",
                       help="Path to HUTB sentences CSV file")
    parser.add_argument("--output", default="hutb_trigram_surprisals_kenlm.csv",
                       help="Output CSV file for surprisal scores")
    parser.add_argument("--test", action="store_true",
                       help="Run test with sample sentences")
    parser.add_argument("--debug", action="store_true",
                       help="Show debug information")
    
    args = parser.parse_args()
    
    # Initialize model
    kenlm_model = KenLMSurprisal(args.model)
    if args.debug:
        kenlm_model.debug = True
    
    # Run test if requested
    if args.test:
        kenlm_model.test_model()
        return
    
    # Calculate surprisals for HUTB
    if os.path.exists(args.hutb):
        results = kenlm_model.calculate_surprisal_for_hutb(args.hutb, args.output)
        
        # Show sample results
        print("\nSample results (first 10):")
        print(results.head(10).to_string())
    else:
        print(f"Error: HUTB file not found: {args.hutb}")
        print("Please provide a valid HUTB CSV file path")


if __name__ == "__main__":
    main()