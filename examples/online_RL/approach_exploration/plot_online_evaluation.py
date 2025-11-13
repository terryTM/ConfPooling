"""
Replicates the Accuracy vs. Generated Tokens plots (similar to Fig. 7 and 9)
from the "Deep Think with Confidence" paper using pre-generated trace data.

This script iterates through a series of increasing budgets (K) and simulates
the online evaluation for each budget point to generate performance curves.
It also prints the question IDs that were answered incorrectly for each method
at each budget step.

Usage:
  python plot_online_evaluation.py --data_dir trace_data
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# --- Configuration ---
NUM_WARMUP_TRACES = 16
# Define the budget points to evaluate, up to our max of 256
BUDGET_STEPS = [32, 64, 128, 256] 
CONSENSUS_THRESHOLD = 0.95

# --- Helper Functions (from previous script) ---

def load_concatenated_json(file_path):
    decoder = json.JSONDecoder()
    data_parts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
    except FileNotFoundError: return None
    idx = 0
    while idx < len(content):
        while idx < len(content) and content[idx].isspace(): idx += 1
        if idx == len(content): break
        try:
            obj, end = decoder.raw_decode(content, idx)
            data_parts.append(obj)
            idx = end
        except json.JSONDecodeError: break
    if not data_parts: return None
    final_data = {k: v for k, v in data_parts[0].items() if k != 'traces'}
    all_traces = [trace for part in data_parts for trace in part.get('traces', [])]
    final_data['traces'] = all_traces
    final_data['num_traces'] = len(all_traces)
    return final_data

def get_consensus_and_vote(answers, weights=None):
    if not answers: return 0.0, None
    if weights is None:
        total_votes = len(answers)
        if total_votes == 0: return 0.0, None
        counts = Counter(answers)
        most_common, top_count = counts.most_common(1)[0]
        return top_count / total_votes, most_common
    else:
        answer_weights = {}
        for answer, weight in zip(answers, weights):
            if answer is not None:
                answer_str = str(answer)
                answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
        total_weight = sum(answer_weights.values())
        if total_weight == 0: return 0.0, None
        winner = max(answer_weights, key=answer_weights.get)
        return answer_weights[winner] / total_weight, winner

def analyze_question_at_budget(file_path, max_budget):
    """
    Analyzes a single question file for a given max_budget K.
    Now also returns the question_id.
    """
    data = load_concatenated_json(file_path)
    if not data or len(data['traces']) < max_budget:
        return None

    all_traces = data['traces'][:max_budget] # Crucially, only use traces up to the current budget
    ground_truth = str(data['ground_truth'])
    qid = data['question_id'] # Get the QID

    # --- Baseline (Cons@K) ---
    baseline_tokens = sum(len(t['tokens']) for t in all_traces)
    baseline_answers = [str(t.get('answer')) for t in all_traces]
    _, baseline_voted_answer = get_consensus_and_vote(baseline_answers)
    baseline_is_correct = (baseline_voted_answer == ground_truth)

    # --- Warmup Phase ---
    warmup_traces = all_traces[:NUM_WARMUP_TRACES]
    online_traces = all_traces[NUM_WARMUP_TRACES:]
    lowest_confs_warmup = [min(t['group_confidence']) for t in warmup_traces if t['group_confidence']]
    if not lowest_confs_warmup: return None

    s_high = np.percentile(lowest_confs_warmup, 10)
    s_low = np.percentile(lowest_confs_warmup, 90)

    # --- Simulate DeepConf Strategies ---
    def simulate_strategy(threshold):
        total_tokens = sum(len(t['tokens']) for t in warmup_traces)
        voting_pool = [{'answer': str(t.get('answer')), 'lgc': min(t['group_confidence']) if t['group_confidence'] else 0} for t in warmup_traces]
        
        consensus, _ = get_consensus_and_vote([t['answer'] for t in voting_pool], [t['lgc'] for t in voting_pool])

        if consensus < CONSENSUS_THRESHOLD:
            for trace in online_traces:
                conf_curve = np.array(trace['group_confidence'])
                stop_indices = np.where(conf_curve < threshold)[0]
                
                current_lgc = 0
                if len(stop_indices) > 0:
                    stop_index = stop_indices[0]
                    num_tokens = stop_index + 1
                    total_tokens += num_tokens
                    current_lgc = conf_curve[stop_index]
                else:
                    total_tokens += len(trace['tokens'])
                    current_lgc = min(conf_curve) if len(conf_curve) > 0 else 0
                
                voting_pool.append({'answer': str(trace.get('answer')), 'lgc': current_lgc})
                
                consensus, _ = get_consensus_and_vote([t['answer'] for t in voting_pool], [t['lgc'] for t in voting_pool])
                if consensus >= CONSENSUS_THRESHOLD:
                    break

        _, final_voted_answer = get_consensus_and_vote([t['answer'] for t in voting_pool], [t['lgc'] for t in voting_pool])
        is_correct = (final_voted_answer == ground_truth) if final_voted_answer is not None else False
        return {'is_correct': is_correct, 'tokens': total_tokens}

    return {
        'qid': qid, # Return the QID
        'baseline': {'is_correct': baseline_is_correct, 'tokens': baseline_tokens},
        'deepconf_high': simulate_strategy(s_high),
        'deepconf_low': simulate_strategy(s_low),
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot Online Evaluation Curves.")
    parser.add_argument('--data_dir', type=str, default='trace_data', help='Directory containing trace data files.')
    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.data_dir, 'aime_2025_*_full.jsonl')))
    if not all_files:
        print(f"Error: No '*_full.jsonl' files found in '{args.data_dir}'")
        return

    results_by_budget = {
        'Majority Voting': [], 'DeepConf-high': [], 'DeepConf-low': []
    }

    # --- Main Loop: Iterate through each budget step ---
    for budget in tqdm(BUDGET_STEPS, desc="Processing Budgets"):
        print(f"\n--- Analyzing for Budget K = {budget} ---")
        budget_results = []
        for file_path in all_files:
            res = analyze_question_at_budget(file_path, budget)
            if res:
                budget_results.append(res)
        
        if not budget_results: continue

        num_q = len(budget_results)
        
        # --- New: Track and Print Incorrect QIDs for this budget ---
        incorrect_qids = {'Majority Voting': [], 'DeepConf-high': [], 'DeepConf-low': []}
        for r in budget_results:
            if not r['baseline']['is_correct']: incorrect_qids['Majority Voting'].append(r['qid'])
            if not r['deepconf_high']['is_correct']: incorrect_qids['DeepConf-high'].append(r['qid'])
            if not r['deepconf_low']['is_correct']: incorrect_qids['DeepConf-low'].append(r['qid'])
        
        print("\nIncorrectly Answered Questions:")
        for method, qids in incorrect_qids.items():
            print(f"  - {method}: {qids if qids else 'None'}")
        
        # Aggregate results for this budget
        total_bl_correct = sum(not r['baseline']['is_correct'] for r in budget_results)
        total_bl_tokens = sum(r['baseline']['tokens'] for r in budget_results)
        results_by_budget['Majority Voting'].append((total_bl_tokens, ((num_q - total_bl_correct) / num_q) * 100))
        
        total_high_correct = sum(not r['deepconf_high']['is_correct'] for r in budget_results)
        total_high_tokens = sum(r['deepconf_high']['tokens'] for r in budget_results)
        results_by_budget['DeepConf-high'].append((total_high_tokens, ((num_q - total_high_correct) / num_q) * 100))

        total_low_correct = sum(not r['deepconf_low']['is_correct'] for r in budget_results)
        total_low_tokens = sum(r['deepconf_low']['tokens'] for r in budget_results)
        results_by_budget['DeepConf-low'].append((total_low_tokens, ((num_q - total_low_correct) / num_q) * 100))

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {'Majority Voting': 'orange', 'DeepConf-high': 'blue', 'DeepConf-low': 'green'}
    markers = {'Majority Voting': 'o', 'DeepConf-high': 's', 'DeepConf-low': '^'}
    
    for method, results in results_by_budget.items():
        if results:
            tokens, accuracies = zip(*results)
            tokens_in_millions = [t / 1e6 for t in tokens]
            ax.plot(tokens_in_millions, accuracies, marker=markers[method], linestyle='--', label=method)

    ax.set_title('Accuracy vs. Generated Tokens (Online Evaluation Simulation)', fontsize=16)
    ax.set_xlabel('Total Tokens Generated (Millions)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    # Ensure the output directory exists
    output_path = 'examples/online_RL/question_exploration/output/online_evaluation_curves.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()
    
    print(f"\nPlot saved as '{output_path}'")

if __name__ == '__main__':
    main()