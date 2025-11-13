"""
Replicates the Online Evaluation results from the "Deep Think with Confidence"
paper, including consensus-based early stopping and correct voting logic.

This script processes each question's data file individually to simulate:
1. An offline warmup phase to set confidence thresholds.
2. An online generation phase with both trace-level and problem-level (consensus)
   early stopping.
3. Confidence-weighted majority voting using Lowest Group Confidence, including
   all generated traces in the voting pool.
4. Aggregation of results and reporting of incorrectly answered questions.

Usage:
  python replicate_online_evaluation.py --data_dir trace_data
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# --- Configuration based on the paper ---
NUM_WARMUP_TRACES = 16
MAX_BUDGET = 256
CONSENSUS_THRESHOLD = 0.95 # As specified in paper section 4.3

# --- Helper Functions ---

def load_concatenated_json(file_path):
    """
    Reads a file containing one or more concatenated JSON objects
    and merges them into a single, valid data structure.
    """
    decoder = json.JSONDecoder()
    data_parts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return None
        
    idx = 0
    while idx < len(content):
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx == len(content): break
        try:
            obj, end = decoder.raw_decode(content, idx)
            data_parts.append(obj)
            idx = end
        except json.JSONDecodeError:
            break
            
    if not data_parts: return None
    
    final_data = {k: v for k, v in data_parts[0].items() if k != 'traces'}
    all_traces = [trace for part in data_parts for trace in part.get('traces', [])]
    final_data['traces'] = all_traces
    final_data['num_traces'] = len(all_traces)
    return final_data

def get_consensus_and_vote(answers, weights=None):
    """
    Calculates consensus and returns the winning answer.
    Supports both weighted and unweighted voting.
    """
    if not answers:
        return 0.0, None

    if weights is None: # Unweighted majority vote
        total_votes = len(answers)
        if total_votes == 0: return 0.0, None
        counts = Counter(answers)
        most_common_answer, top_count = counts.most_common(1)[0]
        consensus = top_count / total_votes
        return consensus, most_common_answer
    else: # Weighted majority vote
        answer_weights = {}
        for answer, weight in zip(answers, weights):
            if answer is not None:
                answer_str = str(answer)
                answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
        
        total_weight = sum(answer_weights.values())
        if total_weight == 0: return 0.0, None
        
        # print out answer_weights
        for ans, wgt in answer_weights.items():
            print(f"Answer: {ans!r}, Weight: {wgt:.4f}")
        winning_answer = max(answer_weights, key=answer_weights.get)
        consensus = answer_weights[winning_answer] / total_weight
        return consensus, winning_answer

def analyze_question(file_path):
    """
    Analyzes a single question file to simulate online evaluation,
    now including consensus stopping and correct voting logic.
    """
    data = load_concatenated_json(file_path)
    if not data or len(data['traces']) < MAX_BUDGET:
        return None

    all_traces = data['traces']
    ground_truth = str(data['ground_truth'])
    qid = data['question_id']

    # --- Baseline (Cons@256) ---
    baseline_tokens = sum(len(t['tokens']) for t in all_traces)
    baseline_answers = [str(t.get('answer')) for t in all_traces]
    _, baseline_voted_answer = get_consensus_and_vote(baseline_answers)
    baseline_is_correct = (baseline_voted_answer == ground_truth)

    # --- Warmup Phase ---
    warmup_traces = all_traces[:NUM_WARMUP_TRACES]
    online_traces = all_traces[NUM_WARMUP_TRACES:MAX_BUDGET]
    lowest_confs_warmup = [min(t['group_confidence']) for t in warmup_traces if t['group_confidence']]
    if not lowest_confs_warmup: return None

    s_high = np.percentile(lowest_confs_warmup, 10) # For DeepConf-high
    s_low = np.percentile(lowest_confs_warmup, 90)  # For DeepConf-low

    # --- Simulate DeepConf Strategies ---
    def simulate_strategy(threshold):
        total_tokens = sum(len(t['tokens']) for t in warmup_traces)
        
        # All traces (stopped or not) are included in the voting pool
        voting_pool = []
        for t in warmup_traces:
            lgc = min(t['group_confidence']) if t['group_confidence'] else 0
            voting_pool.append({'answer': str(t.get('answer')), 'lgc': lgc})

        # Check for consensus after warmup
        answers = [t['answer'] for t in voting_pool]
        weights = [t['lgc'] for t in voting_pool]
        consensus, _ = get_consensus_and_vote(answers, weights)

        if consensus < CONSENSUS_THRESHOLD:
            for trace in online_traces:
                conf_curve = np.array(trace['group_confidence'])
                stop_indices = np.where(conf_curve < threshold)[0]
                
                current_lgc = 0
                if len(stop_indices) > 0: # Trace is stopped early
                    stop_index = stop_indices[0]
                    # Estimate token count for stopped trace
                    # This is an approximation. A more precise method would require token-level timestamps.
                    # Using the number of group confidences as a proxy for generated length
                    num_tokens = stop_index + 1 
                    total_tokens += num_tokens
                    current_lgc = conf_curve[stop_index]
                else: # Trace completes
                    total_tokens += len(trace['tokens'])
                    current_lgc = min(conf_curve) if len(conf_curve) > 0 else 0
                
                # Add the trace to the voting pool regardless of completion
                voting_pool.append({'answer': str(trace.get('answer')), 'lgc': current_lgc})

                # Check consensus after adding each new trace
                answers = [t['answer'] for t in voting_pool]
                weights = [t['lgc'] for t in voting_pool]
                consensus, _ = get_consensus_and_vote(answers, weights)
                if consensus >= CONSENSUS_THRESHOLD:
                    break # Problem-level early stopping

        # Final voting with all collected traces
        final_answers = [t['answer'] for t in voting_pool]
        final_weights = [t['lgc'] for t in voting_pool]
        _, final_voted_answer = get_consensus_and_vote(final_answers, final_weights)
        is_correct = (final_voted_answer == ground_truth) if final_voted_answer is not None else False
        
        return {'is_correct': is_correct, 'tokens': total_tokens}

    result_high = simulate_strategy(s_high)
    result_low = simulate_strategy(s_low)

    return {
        'qid': qid,
        'baseline': {'is_correct': baseline_is_correct, 'tokens': baseline_tokens},
        'deepconf_high': result_high,
        'deepconf_low': result_low,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Online Evaluation Simulation.")
    parser.add_argument('--data_dir', type=str, default='trace_data', help='Directory containing the trace data files.')
    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.data_dir, 'aime_2025_*_full.jsonl')))
    if not all_files:
        print(f"Error: No '*_full.jsonl' files found in '{args.data_dir}'")
        return

    all_results = [analyze_question(fp) for fp in tqdm(all_files, desc="Processing files")]
    all_results = [r for r in all_results if r is not None]

    if not all_results:
        print("No results were generated."); return

    # --- Aggregate results and find incorrect QIDs ---
    incorrect_qids = {'baseline': [], 'deepconf_high': [], 'deepconf_low': []}
    for r in all_results:
        for method in incorrect_qids.keys():
            if not r[method]['is_correct']:
                incorrect_qids[method].append(r['qid'])

    # --- Create summary table ---
    num_q = len(all_results)
    summary_data = []
    for method, name in [('baseline', 'Cons@256'), ('deepconf_high', 'DeepConf-high'), ('deepconf_low', 'DeepConf-low')]:
        total_correct = sum(r[method]['is_correct'] for r in all_results)
        total_tokens = sum(r[method]['tokens'] for r in all_results)
        summary_data.append({
            'Method': name,
            'Accuracy (%)': (total_correct / num_q) * 100,
            'Tokens (Total)': total_tokens
        })
    
    df_summary = pd.DataFrame(summary_data)
    baseline_tokens = df_summary.loc[0, 'Tokens (Total)']
    df_summary['Token Saving (%)'] = df_summary['Tokens (Total)'].apply(
        lambda t: (1 - t / baseline_tokens) * 100
    )
    
    # Formatting
    pd.options.display.float_format = '{:,.1f}'.format
    df_summary['Tokens (Total)'] = df_summary['Tokens (Total)'].map('{:,.0f}'.format)
    df_summary.at[0, 'Token Saving (%)'] = 0.0 # Set baseline saving to 0 explicitly
    
    print("\n--- Online Evaluation Summary ---")
    print(df_summary.to_string(index=False))

    # --- Print incorrect QIDs ---
    print("\n--- Incorrectly Answered Questions ---")
    for method, qids in incorrect_qids.items():
        print(f"\nMethod: {method}")
        if qids:
            print(f"  Count: {len(qids)}")
            print(f"  QIDs: {', '.join(sorted(qids))}")
        else:
            print("  All questions answered correctly.")

if __name__ == '__main__':
    main()