import kenlm
import pandas as pd
import numpy as np
import math
from scipy.stats import linregress

# Load KenLM model
model = kenlm.Model("hindi_trigram.klm")

# Load your sentence CSV
df = pd.read_csv("hutb-sentences.csv")

def get_per_word_surprisal(sentence):
    words = sentence.strip().split()
    surprisals = []
    for i in range(len(words)):
        context = ' '.join(words[max(0, i-2):i])
        target = words[i]
        full_phrase = f"{context} {target}".strip()
        log_prob_full = model.score(full_phrase, bos=(i==0), eos=(i==len(words)-1))
        log_prob_context = model.score(context, bos=(i==0), eos=False) if context else 0.0
        log_prob_word = log_prob_full - log_prob_context
        surprisal = -log_prob_word / math.log(2)
        surprisals.append(surprisal)
    return surprisals

def compute_uid_metrics(surprisals):
    if len(surprisals) < 2:
        return pd.Series({
            'uid_mean': np.mean(surprisals),
            'uid_std': 0,
            'uid_var': 0,
            'uid_slope': 0,
            'uid_combined': 0
        })
    uid_mean = np.mean(surprisals)
    uid_std = np.std(surprisals)
    uid_var = np.var(surprisals)
    uid_slope = linregress(range(len(surprisals)), surprisals).slope
    uid_combined = uid_mean * uid_std * abs(uid_slope)
    return pd.Series({
        'uid_mean': uid_mean,
        'uid_std': uid_std,
        'uid_var': uid_var,
        'uid_slope': uid_slope,
        'uid_combined': uid_combined
    })

# Apply to all rows
df[['uid_mean', 'uid_std', 'uid_var', 'uid_slope', 'uid_combined']] = df['Sentences'].apply(
    lambda x: compute_uid_metrics(get_per_word_surprisal(x))
)

# Save to CSV
df.to_csv("hutb_uid_scores.csv", index=False)
