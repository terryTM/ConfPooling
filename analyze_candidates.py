#!/usr/bin/env python3
"""
Analyze candidate reasoning summaries from Phase 1 traces.

Produces the same candidate display text that Phase 2 uses for follow-up prompts.

Usage:
    python analyze_candidates.py <traces.jsonl>
    python analyze_candidates.py traces.jsonl --budget 256 --percentile 80
    python analyze_candidates.py traces.jsonl --top-n 5 --agg-method sumc
"""
import json
import os
import re
import sys
import argparse
import numpy as np
from collections import Counter


def clean_answer(ans: str) -> str:
    """Clean LaTeX formatting from answer for comparison."""
    if not ans:
        return ans
    ans = re.sub(r"\\(?:,|;|:|!|quad|qquad|enspace|,| )", "", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace(r"\dfrac", r"\frac")
    ans = ans.replace(r"\tfrac", r"\frac")
    ans = ans.replace(r"\displaystyle", "")
    # Only remove variable prefix at START of answer (e.g., "x=" or "n=")
    # Don't remove "c=" from "a+b+c=..." which would corrupt the answer
    ans = re.sub(r"^[a-zA-Z]\s*=", "", ans)
    ans = re.sub(r",+", ",", ans)
    ans = ans.strip(",")
    return ans


def get_trace_confidence(trace: dict) -> float:
    """Get confidence value for a trace."""
    # First try: use existing group_confidence (DeepConf method)
    gc = trace.get('group_confidence', [])
    if gc:
        return min(gc)

    # Second try: use old_group_confidence
    old_gc = trace.get('old_group_confidence', [])
    if old_gc:
        return min(old_gc)

    return 0.0


def split_harmony_content(text: str) -> tuple:
    """Split GPT-OSS output by 'assistantfinal' marker."""
    marker = "assistantfinal"
    if marker in text.lower():
        parts = re.split(marker, text, flags=re.IGNORECASE)
        summary = parts[-1].strip()
        reasoning = parts[0].strip()
        return reasoning, summary
    return "", text.strip()


def get_reasoning_summary(text: str, max_chars: int = 500) -> str:
    """Extract a summary of the reasoning from the trace text."""
    if not text:
        return ""

    # For GPT-OSS, use assistantfinal marker
    _, summary = split_harmony_content(text)
    if summary and summary != text.strip():
        return summary[:max_chars]

    # Fallback: Try to get the conclusion/final part
    conclusion_markers = [
        "therefore", "thus", "hence", "so the answer",
        "the answer is", "we get", "we have", "finally"
    ]

    text_lower = text.lower()
    best_pos = len(text)

    for marker in conclusion_markers:
        pos = text_lower.rfind(marker)
        if pos != -1 and pos < best_pos:
            best_pos = pos

    if best_pos < len(text) - 50:
        summary = text[best_pos:best_pos + max_chars]
    else:
        summary = text[-max_chars:]

    return summary.strip()


def analyze_traces(
    traces: list,
    budget: int = 256,
    num_calibration: int = 32,
    percentile: float = 80.0,
    top_n: int = 4,
    agg_method: str = "maxc",
) -> dict:
    """
    Analyze traces and produce candidate information.

    Returns dict with:
        - top_n_data: the candidate data
        - candidate_display_text: formatted text
        - stats: various statistics
    """
    # Limit traces to budget
    traces = traces[:budget]

    # Step 1: Compute min_conf and clean answers
    for t in traces:
        t['min_conf'] = get_trace_confidence(t)
        raw_answer = t.get('answer', '')
        t['clean_answer'] = clean_answer(raw_answer) if raw_answer else None

    # Step 2: Compute screening threshold
    screening_threshold = 0.0
    if len(traces) >= num_calibration:
        calibration_confs = [t['min_conf'] for t in traces[:num_calibration]]
        screening_threshold = np.percentile(calibration_confs, percentile)

    # Step 3: Filter high-confidence traces
    filtered_traces = [
        t for t in traces
        if t['min_conf'] >= screening_threshold and t.get('clean_answer')
    ]

    if not filtered_traces:
        return {
            'top_n_data': {},
            'candidate_display_text': "No traces passed filtering.",
            'stats': {
                'total_traces': len(traces),
                'filtered_traces': 0,
                'screening_threshold': screening_threshold,
            }
        }

    # Step 4: Group by answer and aggregate
    answer_stats = {}
    for t in filtered_traces:
        ans = t['clean_answer']
        conf = t['min_conf']
        if ans not in answer_stats:
            answer_stats[ans] = {'confs': [], 'best_conf': 0, 'best_trace': None}
        answer_stats[ans]['confs'].append(conf)
        if conf > answer_stats[ans]['best_conf']:
            answer_stats[ans]['best_conf'] = conf
            answer_stats[ans]['best_trace'] = t

    # Compute aggregated scores
    label_map = {"maxc": "Max Confidence", "sumc": "Cumulative Confidence", "meanc": "Mean Confidence"}

    for ans, stats in answer_stats.items():
        confs = stats['confs']
        if agg_method == "maxc":
            stats['agg_score'] = max(confs)
        elif agg_method == "sumc":
            stats['agg_score'] = sum(confs)
        elif agg_method == "meanc":
            stats['agg_score'] = sum(confs) / len(confs)
        stats['count'] = len(confs)

    # Step 5: Select top N answers
    top_n_answers = sorted(
        answer_stats.keys(),
        key=lambda x: answer_stats[x]['agg_score'],
        reverse=True
    )[:top_n]

    # Build top_n_data
    top_n_data = {}
    for ans in top_n_answers:
        stats = answer_stats[ans]
        best_trace = stats['best_trace']
        summary = get_reasoning_summary(best_trace.get('text', '')) if best_trace else ''
        top_n_data[ans] = {
            'score': stats['agg_score'],
            'metric_name': label_map[agg_method],
            'summary': summary,
            'count': stats['count'],
            'best_conf': stats['best_conf'],
        }

    # Build candidate display text (same as Phase 2)
    candidate_display_text = ""
    for i, ans in enumerate(top_n_answers, 1):
        info = top_n_data[ans]
        candidate_display_text += f"\n[Candidate {i}]: {ans} ({info['metric_name']}: {info['score']:.4f})\n"
        candidate_display_text += f"Reasoning Summary: {info['summary']}\n"

    return {
        'top_n_data': top_n_data,
        'top_n_answers': top_n_answers,
        'candidate_display_text': candidate_display_text,
        'stats': {
            'total_traces': len(traces),
            'filtered_traces': len(filtered_traces),
            'unique_answers': len(answer_stats),
            'screening_threshold': screening_threshold,
        },
        'all_answer_stats': answer_stats,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze candidate reasoning summaries from Phase 1 traces')
    parser.add_argument('traces_file', help='Path to JSONL file with traces')
    parser.add_argument('--budget', type=int, default=256, help='Max traces to use (default: 256)')
    parser.add_argument('--num-calibration', type=int, default=32, help='Calibration traces (default: 32)')
    parser.add_argument('--percentile', type=float, default=80.0, help='Screening percentile (default: 80)')
    parser.add_argument('--top-n', type=int, default=4, help='Top N candidates (default: 4)')
    parser.add_argument('--agg-method', choices=['maxc', 'sumc', 'meanc'], default='maxc',
                        help='Aggregation method (default: maxc)')
    parser.add_argument('--show-all', action='store_true', help='Show all answers, not just top N')
    parser.add_argument('--full-summary', action='store_true', help='Show full reasoning summaries')

    args = parser.parse_args()

    # Load traces
    if not os.path.exists(args.traces_file):
        print(f"Error: File not found: {args.traces_file}")
        sys.exit(1)

    traces = []
    with open(args.traces_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not traces:
        print("No traces loaded from file")
        sys.exit(1)

    # Get question info
    question = traces[0].get('question', 'N/A')
    ground_truth = traces[0].get('ground_truth', 'N/A')

    print("=" * 80)
    print("CANDIDATE ANALYSIS")
    print("=" * 80)
    print(f"\nFile: {args.traces_file}")
    print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")
    print(f"\nSettings: budget={args.budget}, percentile={args.percentile}, top_n={args.top_n}, agg={args.agg_method}")
    print("=" * 80)

    # Analyze
    result = analyze_traces(
        traces,
        budget=args.budget,
        num_calibration=args.num_calibration,
        percentile=args.percentile,
        top_n=args.top_n,
        agg_method=args.agg_method,
    )

    stats = result['stats']
    print(f"\nStats:")
    print(f"  Total traces: {stats['total_traces']}")
    print(f"  Screening threshold (p{args.percentile}): {stats['screening_threshold']:.4f}")
    print(f"  Filtered traces: {stats['filtered_traces']}")
    print(f"  Unique answers: {stats.get('unique_answers', 0)}")

    # Show candidate display text
    print("\n" + "=" * 80)
    print("CANDIDATE DISPLAY TEXT (as shown to model in Phase 2)")
    print("=" * 80)
    print(result['candidate_display_text'])

    # Check if ground truth is in top N
    clean_gt = clean_answer(ground_truth)
    top_n_answers = result.get('top_n_answers', [])
    gt_in_top_n = clean_gt in top_n_answers

    print("=" * 80)
    print(f"Ground truth '{ground_truth}' in top {args.top_n}: {'YES' if gt_in_top_n else 'NO'}")

    if not gt_in_top_n and result.get('all_answer_stats'):
        # Check if GT appears anywhere
        all_stats = result['all_answer_stats']
        if clean_gt in all_stats:
            gt_stats = all_stats[clean_gt]
            print(f"  (GT appears with count={gt_stats['count']}, agg_score={gt_stats['agg_score']:.4f})")
        else:
            print(f"  (GT does not appear in any filtered trace)")

    # Optionally show all answers
    if args.show_all and result.get('all_answer_stats'):
        print("\n" + "=" * 80)
        print("ALL ANSWERS (sorted by score)")
        print("=" * 80)
        all_stats = result['all_answer_stats']
        sorted_answers = sorted(all_stats.keys(), key=lambda x: all_stats[x]['agg_score'], reverse=True)

        for i, ans in enumerate(sorted_answers, 1):
            stats = all_stats[ans]
            is_gt = "  <-- GT" if clean_answer(ans) == clean_gt else ""
            in_top = " [TOP]" if ans in top_n_answers else ""
            print(f"{i:3d}. {ans[:50]:<50} count={stats['count']:3d}  score={stats['agg_score']:.4f}{in_top}{is_gt}")


if __name__ == "__main__":
    main()
