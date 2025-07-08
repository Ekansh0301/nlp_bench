#!/usr/bin/env python3
"""
Generate Information Status (IS) scores for HUTB sentences
Following Ranjan et al. (2022) methodology exactly
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import stanza
import json
import ast
import logging
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ISScoreCalculator:
    def __init__(self, dependency_file=None):
        """Initialize with optional pre-computed dependency parses"""
        logging.info("Initializing IS Score Calculator...")
        
        # Load pre-computed parses if available
        self.precomputed_parses = {}
        if dependency_file:
            logging.info(f"Loading pre-computed parses from {dependency_file}")
            try:
                dep_df = pd.read_csv(dependency_file)
                
                # Debug: Check what we have
                logging.info(f"Dependency file has {len(dep_df)} rows")
                
                # Check first few rows to understand the format
                if len(dep_df) > 0:
                    first_row = dep_df.iloc[0]
                    if 'dependencies' in first_row:
                        logging.info(f"First dependency entry type: {type(first_row['dependencies'])}")
                        logging.info(f"First 200 chars: {str(first_row['dependencies'])[:200]}")
                
                # Load the dependencies
                valid_parses = 0
                for idx, row in dep_df.iterrows():
                    if pd.notna(row.get('dependencies', None)):
                        try:
                            # Try different parsing methods
                            deps_str = row['dependencies']
                            
                            # Method 1: ast.literal_eval (safest for Python literals)
                            try:
                                deps = ast.literal_eval(deps_str)
                                self.precomputed_parses[row['sentence_id']] = deps
                                valid_parses += 1
                                continue
                            except:
                                pass
                            
                            # Method 2: JSON
                            try:
                                deps = json.loads(deps_str)
                                self.precomputed_parses[row['sentence_id']] = deps
                                valid_parses += 1
                                continue
                            except:
                                pass
                            
                            # Method 3: eval (less safe but sometimes necessary)
                            try:
                                deps = eval(deps_str)
                                self.precomputed_parses[row['sentence_id']] = deps
                                valid_parses += 1
                            except:
                                if idx < 5:  # Log first few failures
                                    logging.debug(f"Could not parse dependencies for {row['sentence_id']}")
                        except Exception as e:
                            if idx < 5:
                                logging.debug(f"Error processing row {idx}: {e}")
                
                logging.info(f"Loaded {valid_parses} valid pre-computed parses out of {len(dep_df)} rows")
                
                # Show example of loaded parse
                if valid_parses > 0:
                    example_id = list(self.precomputed_parses.keys())[0]
                    example_parse = self.precomputed_parses[example_id]
                    logging.info(f"Example parse loaded - ID: {example_id}, Type: {type(example_parse)}, Length: {len(example_parse) if isinstance(example_parse, list) else 'N/A'}")
                    
            except Exception as e:
                logging.warning(f"Could not load dependency file: {e}")
        
        # Always initialize Stanza as backup
        logging.info("Initializing Stanza for parsing...")
        try:
            self.nlp = stanza.Pipeline('hi', 
                                      processors='tokenize,pos,lemma,depparse',
                                      verbose=False, 
                                      use_gpu=True)
        except:
            stanza.download('hi')
            self.nlp = stanza.Pipeline('hi', 
                                      processors='tokenize,pos,lemma,depparse',
                                      verbose=False, 
                                      use_gpu=True)
        
        # Comprehensive Hindi pronouns list
        self.hindi_pronouns = {
            # First person
            'मैं', 'हम', 'मुझे', 'हमें', 'मुझको', 'हमको', 'मेरा', 'हमारा', 
            'मेरे', 'हमारे', 'मेरी', 'हमारी', 'मुझसे', 'हमसे', 'मैंने', 'हमने',
            # Second person
            'तू', 'तुम', 'आप', 'तुझे', 'तुम्हें', 'आपको', 'तेरा', 'तुम्हारा',
            'आपका', 'तेरे', 'तुम्हारे', 'आपके', 'तेरी', 'तुम्हारी', 'आपकी',
            'तूने', 'तुमने', 'आपने',
            # Third person
            'वह', 'वे', 'यह', 'ये', 'उसे', 'उन्हें', 'इसे', 'इन्हें',
            'उसको', 'उनको', 'इसको', 'इनको', 'उसका', 'उनका', 'इसका', 'इनका',
            'उसकी', 'उनकी', 'इसकी', 'इनकी', 'उसके', 'उनके', 'इसके', 'इनके',
            'उसने', 'उन्होंने', 'इसने', 'इन्होंने',
            # Reflexive/Respectful
            'अपना', 'अपने', 'अपनी', 'स्वयं', 'खुद'
        }
        
        # Subject and object relations
        self.subject_rels = {'nsubj', 'nsubj:pass', 'csubj', 'csubj:pass'}
        self.object_rels = {'obj', 'iobj', 'obl:arg', 'obl'}
        
        logging.info("Initialization complete!")
    
    def parse_sentence_id(self, sentence_id):
        """Parse HUTB sentence ID format: file-name__sentnum.variant"""
        try:
            parts = sentence_id.split('__')
            if len(parts) != 2:
                return sentence_id, 0, 0
            
            doc_id = parts[0]
            sent_variant = parts[1]
            
            if '.' in sent_variant:
                sent_num_str, variant_str = sent_variant.rsplit('.', 1)
                sent_num = int(sent_num_str)
                variant_num = int(variant_str)
            else:
                sent_num = int(sent_variant)
                variant_num = 0
            
            return doc_id, sent_num, variant_num
        except:
            return sentence_id, 0, 0
    
    def get_constituents(self, sentence_id, sentence_text):
        """Get subject and object constituents from parse"""
        # First check precomputed parses
        if sentence_id in self.precomputed_parses:
            deps = self.precomputed_parses[sentence_id]
            # Ensure it's a list of dicts
            if isinstance(deps, list) and len(deps) > 0 and isinstance(deps[0], dict):
                return self._extract_constituents_from_parse(deps)
        
        # Otherwise parse with Stanza
        if hasattr(self, 'nlp'):
            try:
                doc = self.nlp(sentence_text)
                dependencies = []
                for sent in doc.sentences:
                    for word in sent.words:
                        dependencies.append({
                            'id': word.id,
                            'text': word.text,
                            'lemma': word.lemma,
                            'head': word.head,
                            'deprel': word.deprel,
                            'upos': word.upos
                        })
                return self._extract_constituents_from_parse(dependencies)
            except Exception as e:
                logging.debug(f"Could not parse sentence: {e}")
                return None, None
        
        return None, None
    
    def _extract_constituents_from_parse(self, dependencies):
        """Extract subject and object from dependency parse"""
        subject = None
        object_ = None
        
        for dep in dependencies:
            # Find subject
            if dep.get('deprel') in self.subject_rels:
                subject = {
                    'head_id': dep['id'],
                    'head_word': dep['text'],
                    'head_lemma': dep.get('lemma', dep['text']),
                    'deprel': dep['deprel'],
                    'words': self._get_constituent_words_iterative(dependencies, dep['id'])
                }
            
            # Find object
            elif dep.get('deprel') in self.object_rels:
                if object_ is None or (dep['deprel'] in ['obj', 'iobj'] and object_.get('deprel') == 'obl'):
                    object_ = {
                        'head_id': dep['id'],
                        'head_word': dep['text'],
                        'head_lemma': dep.get('lemma', dep['text']),
                        'deprel': dep['deprel'],
                        'words': self._get_constituent_words_iterative(dependencies, dep['id'])
                    }
        
        return subject, object_
    
    def _get_constituent_words_iterative(self, dependencies, head_id):
        """Get all words in a constituent using iterative approach"""
        words = []
        visited = set()
        queue = deque([head_id])
        
        # Build a map for faster lookup
        dep_map = {d['id']: d for d in dependencies}
        children_map = defaultdict(list)
        for d in dependencies:
            if d.get('head', 0) > 0:
                children_map[d['head']].append(d['id'])
        
        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)
            
            # Add current word
            if current_id in dep_map:
                words.append(dep_map[current_id]['text'])
            
            # Add children to queue
            if current_id in children_map:
                queue.extend(children_map[current_id])
        
        return words
    
    def is_given(self, constituent, previous_sentences):
        """Check if constituent is Given"""
        if constituent is None or not previous_sentences:
            return False
        
        # Check if head is pronoun
        if constituent['head_word'] in self.hindi_pronouns:
            return True
        
        # Check word overlap with previous sentence
        prev_words = set()
        if previous_sentences:
            prev_text = previous_sentences[-1]
            prev_words = set(prev_text.split())
        
        # Check for overlap
        constituent_words = set([constituent['head_word'], constituent['head_lemma']] + constituent.get('words', []))
        
        # Remove punctuation
        constituent_words = {w.strip('।,.!?') for w in constituent_words if w}
        prev_words = {w.strip('।,.!?') for w in prev_words if w}
        
        return bool(constituent_words & prev_words)
    
    def calculate_is_score(self, subject, object_, subject_given, object_given):
        """Calculate IS score following paper's definition"""
        if subject is None or object_ is None:
            return 0
        
        # Determine surface order
        subj_position = subject['head_id']
        obj_position = object_['head_id']
        
        if subj_position < obj_position:
            # Subject-Object order
            if subject_given and not object_given:
                return 1   # Given-New
            elif not subject_given and object_given:
                return -1  # New-Given
        else:
            # Object-Subject order
            if object_given and not subject_given:
                return 1   # Given-New  
            elif not object_given and subject_given:
                return -1  # New-Given
        
        return 0  # Given-Given or New-New
    
    def process_sentences(self, sentences_df):
        """Process sentences maintaining proper discourse context"""
        results = []
        
        # Group by document
        doc_groups = defaultdict(list)
        
        logging.info("Grouping sentences by document...")
        for idx, row in sentences_df.iterrows():
            sent_id = row['Sentence ID']
            doc_id, sent_num, variant_num = self.parse_sentence_id(sent_id)
            
            doc_groups[doc_id].append({
                'sentence_id': sent_id,
                'sentence': row['Sentences'].strip(),
                'sent_num': sent_num,
                'variant_num': variant_num
            })
        
        logging.info(f"Found {len(doc_groups)} documents")
        
        # Process each document
        for doc_id, doc_sentences in tqdm(doc_groups.items(), desc="Processing documents"):
            # Sort by sentence number and variant
            doc_sentences.sort(key=lambda x: (x['sent_num'], x['variant_num']))
            
            # Track previous sentences for context
            previous_sentences = []
            
            for sent_data in doc_sentences:
                # Get constituents
                subject, object_ = self.get_constituents(sent_data['sentence_id'], 
                                                        sent_data['sentence'])
                
                # Initialize
                is_score = 0
                subject_given = False
                object_given = False
                
                if subject and object_:
                    # Check given status
                    subject_given = self.is_given(subject, previous_sentences)
                    object_given = self.is_given(object_, previous_sentences)
                    
                    # Calculate IS score
                    is_score = self.calculate_is_score(subject, object_, 
                                                      subject_given, object_given)
                
                results.append({
                    'sentence_id': sent_data['sentence_id'],
                    'sentence': sent_data['sentence'],
                    'has_subject': subject is not None,
                    'has_object': object_ is not None,
                    'is_score': is_score,
                    'subject_given': subject_given,
                    'object_given': object_given,
                    'subject_head': subject['head_word'] if subject else None,
                    'object_head': object_['head_word'] if object_ else None
                })
                
                # Update context only with reference sentences
                if sent_data['variant_num'] == 0:
                    previous_sentences.append(sent_data['sentence'])
                    if len(previous_sentences) > 3:
                        previous_sentences.pop(0)
        
        return pd.DataFrame(results)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate IS scores for HUTB sentences")
    parser.add_argument("--sentences", default="hutb-sentences.csv", 
                       help="Path to sentences CSV")
    parser.add_argument("--dependency", default="enhanced_dependency_detailed.csv",
                       help="Path to dependency parse results (optional)")
    parser.add_argument("--output", default="hutb_is_scores.csv",
                       help="Output file name")
    
    args = parser.parse_args()
    
    # Load sentences
    logging.info(f"Loading sentences from {args.sentences}")
    sentences_df = pd.read_csv(args.sentences)
    logging.info(f"Loaded {len(sentences_df)} sentences")
    
    # Initialize calculator
    calculator = ISScoreCalculator(dependency_file=args.dependency)
    
    # Process sentences
    results_df = calculator.process_sentences(sentences_df)
    
    # Save results
    results_df.to_csv(args.output, index=False)
    logging.info(f"Saved IS scores to {args.output}")
    
    # Print statistics
    print("\n" + "="*60)
    print("IS SCORE STATISTICS")
    print("="*60)
    
    both_so = results_df[results_df['has_subject'] & results_df['has_object']]
    print(f"Total sentences: {len(results_df)}")
    print(f"Sentences with both S and O: {len(both_so)} ({len(both_so)/len(results_df)*100:.1f}%)")
    
    if len(both_so) > 0:
        print(f"\nIS Score Distribution:")
        print(f"  Given-New (+1): {sum(both_so['is_score'] == 1):>5} ({sum(both_so['is_score'] == 1)/len(both_so)*100:>5.1f}%)")
        print(f"  New-Given (-1): {sum(both_so['is_score'] == -1):>5} ({sum(both_so['is_score'] == -1)/len(both_so)*100:>5.1f}%)")
        print(f"  Same (0):       {sum(both_so['is_score'] == 0):>5} ({sum(both_so['is_score'] == 0)/len(both_so)*100:>5.1f}%)")

if __name__ == "__main__":
    main()