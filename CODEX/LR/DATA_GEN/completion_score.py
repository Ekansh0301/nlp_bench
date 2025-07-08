#!/usr/bin/env python3
"""
Generate Incremental Dependency Completion Scores for HUTB sentences
A robust implementation that handles all edge cases
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import logging
from collections import defaultdict
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class IncrementalCompletionScorer:
    def __init__(self):
        """Initialize the scorer with configuration"""
        # Weights for different components
        self.weights = {
            'open_deps': 1.0,      # Active dependencies waiting for head
            'completed': 0.5,      # Just-completed dependencies  
            'future_deps': 0.3,    # Future dependencies from current head
            'variance_penalty': 0.2 # Penalty for uneven distribution
        }
        
    def load_dependencies(self, dependency_file):
        """Load pre-computed dependencies from file"""
        logging.info(f"Loading dependencies from {dependency_file}")
        
        dep_df = pd.read_csv(dependency_file)
        logging.info(f"Loaded {len(dep_df)} sentences")
        
        # Parse dependencies
        parsed_deps = {}
        failed_parses = 0
        
        for _, row in tqdm(dep_df.iterrows(), total=len(dep_df), desc="Loading dependencies"):
            sent_id = row['sentence_id']
            
            if pd.notna(row.get('dependencies')):
                try:
                    # Handle string representation of list
                    deps_str = row['dependencies']
                    if isinstance(deps_str, str):
                        deps = ast.literal_eval(deps_str)
                    else:
                        deps = deps_str
                    
                    # Validate it's a list of dicts
                    if isinstance(deps, list) and len(deps) > 0:
                        parsed_deps[sent_id] = deps
                    else:
                        failed_parses += 1
                except Exception as e:
                    failed_parses += 1
                    if failed_parses <= 5:
                        logging.debug(f"Failed to parse {sent_id}: {e}")
            else:
                failed_parses += 1
        
        logging.info(f"Successfully parsed {len(parsed_deps)} dependencies")
        if failed_parses > 0:
            logging.warning(f"Failed to parse {failed_parses} sentences")
            
        return parsed_deps
    
    def calculate_incremental_score(self, dependencies):
        """
        Calculate incremental completion score for a sentence
        Based on how dependencies accumulate and resolve word by word
        """
        if not dependencies:
            return 0.0
        
        try:
            # Get sentence length
            max_position = max(d['id'] for d in dependencies)
            
            # Initialize tracking
            position_complexities = []
            
            # Process each word position
            for current_pos in range(1, max_position + 1):
                complexity = self._calculate_position_complexity(
                    dependencies, current_pos
                )
                position_complexities.append(complexity)
            
            # Calculate final score
            if position_complexities:
                mean_complexity = np.mean(position_complexities)
                std_complexity = np.std(position_complexities)
                
                # Higher mean = generally complex
                # Higher std = uneven complexity (surprising)
                final_score = (
                    mean_complexity + 
                    self.weights['variance_penalty'] * std_complexity
                )
                
                return max(0.0, final_score)  # Ensure non-negative
            else:
                return 0.0
                
        except Exception as e:
            logging.debug(f"Error calculating score: {e}")
            return 0.0
    
    def _calculate_position_complexity(self, dependencies, position):
        """Calculate complexity at a specific position in the sentence"""
        open_deps = 0
        completed_deps = 0
        future_deps = 0
        
        for dep in dependencies:
            dep_id = dep['id']
            head_id = dep.get('head', 0)
            
            # Skip root dependencies
            if head_id == 0:
                continue
            
            # Classify dependency status at current position
            if dep_id <= position < head_id:
                # Dependency is open (word seen, head not yet)
                open_deps += 1
                
            elif dep_id <= position and head_id == position:
                # Dependency just completed at this position
                completed_deps += 1
                
            elif dep_id > position and head_id <= position:
                # Future dependency (head seen, dependent coming)
                future_deps += 1
        
        # Calculate weighted complexity
        complexity = (
            self.weights['open_deps'] * open_deps +
            self.weights['completed'] * completed_deps +
            self.weights['future_deps'] * future_deps
        )
        
        return complexity
    
    def process_sentences(self, sentences_file, dependency_file):
        """Process all sentences and generate scores"""
        # Load sentences
        logging.info(f"Loading sentences from {sentences_file}")
        sentences_df = pd.read_csv(sentences_file)
        
        # Load dependencies
        parsed_deps = self.load_dependencies(dependency_file)
        
        # Calculate scores
        results = []
        no_dep_count = 0
        
        logging.info("Calculating incremental completion scores...")
        for _, row in tqdm(sentences_df.iterrows(), total=len(sentences_df), desc="Processing"):
            sent_id = row['Sentence ID']
            sentence = row['Sentences']
            
            # Get dependencies
            if sent_id in parsed_deps:
                score = self.calculate_incremental_score(parsed_deps[sent_id])
            else:
                score = 0.0
                no_dep_count += 1
            
            results.append({
                'sentence_id': sent_id,
                'sentence': sentence,
                'incremental_completion_score': score
            })
        
        if no_dep_count > 0:
            logging.warning(f"{no_dep_count} sentences had no dependency parse")
            
        return pd.DataFrame(results)
    
    def validate_and_analyze(self, scores_df, dependency_file):
        """Validate scores and provide analysis"""
        logging.info("\nValidating and analyzing scores...")
        
        # Basic statistics
        scores = scores_df['incremental_completion_score']
        print("\n" + "="*60)
        print("INCREMENTAL COMPLETION SCORE STATISTICS")
        print("="*60)
        print(f"Total sentences: {len(scores_df)}")
        print(f"Mean score: {scores.mean():.3f}")
        print(f"Std deviation: {scores.std():.3f}")
        print(f"Min score: {scores.min():.3f}")
        print(f"Max score: {scores.max():.3f}")
        print(f"Sentences with score > 0: {(scores > 0).sum()} ({(scores > 0).sum()/len(scores)*100:.1f}%)")
        
        # Check correlations
        dep_df = pd.read_csv(dependency_file)
        merged = pd.merge(
            scores_df, 
            dep_df[['sentence_id', 'dependency_length', 'num_words']], 
            on='sentence_id',
            how='left'
        )
        
        if len(merged) > 0:
            print("\nCorrelations:")
            corr_deplen = merged['incremental_completion_score'].corr(merged['dependency_length'])
            corr_numwords = merged['incremental_completion_score'].corr(merged['num_words'])
            print(f"  With dependency length: {corr_deplen:.3f}")
            print(f"  With num words: {corr_numwords:.3f}")
        
        # Show score distribution
        print("\nScore distribution:")
        print(f"  0-1:   {((scores >= 0) & (scores < 1)).sum()} sentences")
        print(f"  1-2:   {((scores >= 1) & (scores < 2)).sum()} sentences")
        print(f"  2-3:   {((scores >= 2) & (scores < 3)).sum()} sentences")
        print(f"  3+:    {(scores >= 3).sum()} sentences")
        
        # Show examples
        print("\n" + "="*60)
        print("EXAMPLE SENTENCES")
        print("="*60)
        
        print("\nHighest complexity scores:")
        for _, row in scores_df.nlargest(3, 'incremental_completion_score').iterrows():
            print(f"\nScore: {row['incremental_completion_score']:.3f}")
            print(f"ID: {row['sentence_id']}")
            print(f"Sentence: {row['sentence'][:80]}...")
            
        print("\nLowest complexity scores:")
        for _, row in scores_df.nsmallest(3, 'incremental_completion_score').iterrows():
            if row['incremental_completion_score'] > 0:
                print(f"\nScore: {row['incremental_completion_score']:.3f}")
                print(f"ID: {row['sentence_id']}")
                print(f"Sentence: {row['sentence'][:80]}...")

def main():
    parser = argparse.ArgumentParser(
        description="Generate incremental dependency completion scores for HUTB"
    )
    parser.add_argument("--sentences", default="hutb-sentences.csv",
                       help="Path to sentences CSV file")
    parser.add_argument("--dependencies", default="enhanced_dependency_detailed.csv",
                       help="Path to dependency parse results")
    parser.add_argument("--output", default="hutb_incremental_completion_scores.csv",
                       help="Output file name")
    
    args = parser.parse_args()
    
    # Initialize scorer
    scorer = IncrementalCompletionScorer()
    
    # Process sentences
    scores_df = scorer.process_sentences(args.sentences, args.dependencies)
    
    # Save results
    scores_df.to_csv(args.output, index=False)
    logging.info(f"\nSaved scores to {args.output}")
    
    # Validate and analyze
    scorer.validate_and_analyze(scores_df, args.dependencies)
    
    print(f"\n✓ Successfully generated incremental completion scores!")
    print(f"✓ Output saved to: {args.output}")

if __name__ == "__main__":
    main()