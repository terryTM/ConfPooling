#!/usr/bin/env python3
"""
Analyze benchmark results from Modal runs

Usage:
    python analyze_results.py results/benchmark_results_20250119_*.jsonl
    python analyze_results.py results/*.jsonl --methods mv256,online,pooling
    python analyze_results.py results/*.jsonl --output summary.txt
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
import sys


def load_results(file_paths: List[str]) -> List[Dict]:
    """Load results from JSONL files"""
    all_results = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}", file=sys.stderr)

    return all_results


def compute_statistics(results: List[Dict], methods: List[str]) -> Dict[str, Dict]:
    """Compute statistics for each method"""
    stats = {}

    for method in methods:
        correct_key = f"{method}_correct"
        tokens_key = f"{method}_tokens"
        time_key = f"{method}_time"

        # Filter valid results
        valid = [r for r in results if r.get(correct_key) is not None]

        if not valid:
            stats[method] = {
                'count': 0,
                'accuracy': 0.0,
                'correct': 0,
                'total': 0,
                'avg_tokens': 0,
                'total_tokens': 0,
                'avg_time': 0.0,
                'total_time': 0.0,
            }
            continue

        correct_count = sum(1 for r in valid if r[correct_key])
        total_tokens = sum(r[tokens_key] for r in valid if r.get(tokens_key))
        total_time = sum(r[time_key] for r in valid if r.get(time_key))

        stats[method] = {
            'count': len(valid),
            'accuracy': correct_count / len(valid),
            'correct': correct_count,
            'total': len(valid),
            'avg_tokens': total_tokens / len(valid) if valid else 0,
            'total_tokens': total_tokens,
            'avg_time': total_time / len(valid) if valid else 0,
            'total_time': total_time,
        }

        # Method-specific stats
        if method == 'online':
            survivors_list = [r.get('online_survivors', 0) for r in valid]
            stats[method]['avg_survivors'] = sum(survivors_list) / len(survivors_list) if survivors_list else 0

        if method == 'pooling':
            candidates_list = [r.get('pooling_candidates_count', 0) for r in valid]
            stats[method]['avg_candidates'] = sum(candidates_list) / len(candidates_list) if candidates_list else 0

    return stats


def print_summary_table(stats: Dict[str, Dict]):
    """Print summary table"""
    print("\n" + "="*100)
    print(" "*35 + "BENCHMARK SUMMARY")
    print("="*100)
    print(f"{'Method':<15} {'Accuracy':<12} {'Avg Tokens':<15} {'Total Tokens':<15} {'Avg Time':<12} {'Total Time':<12}")
    print("-"*100)

    for method, s in stats.items():
        if s['count'] == 0:
            continue

        accuracy_str = f"{s['accuracy']:.1%} ({s['correct']}/{s['total']})"
        avg_tokens_str = f"{s['avg_tokens']:,.0f}"
        total_tokens_str = f"{s['total_tokens']:,.0f}"
        avg_time_str = f"{s['avg_time']:.1f}s"
        total_time_str = f"{s['total_time']:.1f}s"

        print(f"{method.upper():<15} {accuracy_str:<12} {avg_tokens_str:<15} {total_tokens_str:<15} {avg_time_str:<12} {total_time_str:<12}")

    print("="*100)


def print_efficiency_comparison(stats: Dict[str, Dict]):
    """Print efficiency comparison"""
    if 'mv256' not in stats or stats['mv256']['count'] == 0:
        return

    mv_tokens = stats['mv256']['avg_tokens']
    mv_time = stats['mv256']['avg_time']

    print("\n" + "="*100)
    print(" "*30 + "EFFICIENCY COMPARISON (vs MV@256)")
    print("="*100)
    print(f"{'Method':<15} {'Token Savings':<20} {'Time Savings':<20} {'Accuracy Delta':<20}")
    print("-"*100)

    for method, s in stats.items():
        if method == 'mv256' or s['count'] == 0:
            continue

        token_savings = (1 - s['avg_tokens'] / mv_tokens) * 100 if mv_tokens > 0 else 0
        time_savings = (1 - s['avg_time'] / mv_time) * 100 if mv_time > 0 else 0
        accuracy_delta = (s['accuracy'] - stats['mv256']['accuracy']) * 100

        token_str = f"{token_savings:+.1f}%"
        time_str = f"{time_savings:+.1f}%"
        accuracy_str = f"{accuracy_delta:+.1f} pp"

        print(f"{method.upper():<15} {token_str:<20} {time_str:<20} {accuracy_str:<20}")

    print("="*100)
    print("Note: '+' means savings/improvement, '-' means more expensive/worse")


def print_method_specific_stats(stats: Dict[str, Dict]):
    """Print method-specific statistics"""
    print("\n" + "="*100)
    print(" "*30 + "METHOD-SPECIFIC STATISTICS")
    print("="*100)

    if 'online' in stats and stats['online']['count'] > 0:
        print(f"\nONLINE MODE:")
        print(f"  Average Survivors: {stats['online'].get('avg_survivors', 0):.1f} / 256")
        survival_rate = stats['online'].get('avg_survivors', 0) / 256 * 100
        print(f"  Survival Rate: {survival_rate:.1f}%")

    if 'pooling' in stats and stats['pooling']['count'] > 0:
        print(f"\nPOOLING MODE:")
        print(f"  Average Candidates in Info Packet: {stats['pooling'].get('avg_candidates', 0):.1f}")

    print("="*100)


def print_detailed_results(results: List[Dict], methods: List[str], max_rows: int = None):
    """Print detailed per-question results"""
    print("\n" + "="*150)
    print(" "*60 + "DETAILED RESULTS")
    print("="*150)

    header = f"{'QID':<6} {'Question':<40}"
    for method in methods:
        header += f" {method.upper():<30}"
    print(header)
    print("-"*150)

    for i, r in enumerate(results):
        if max_rows and i >= max_rows:
            print(f"... ({len(results) - max_rows} more rows)")
            break

        qid = r['qid']
        question = r['question'][:37] + "..." if len(r['question']) > 40 else r['question']

        row = f"{qid:<6} {question:<40}"

        for method in methods:
            correct_key = f"{method}_correct"
            answer_key = f"{method}_answer"
            tokens_key = f"{method}_tokens"

            if r.get(correct_key) is not None:
                correct = r[correct_key]
                symbol = "✓" if correct else "✗"
                answer = str(r.get(answer_key, "N/A"))[:15]
                tokens = r.get(tokens_key, 0)
                tokens_str = f"{tokens/1000:.0f}k" if tokens > 0 else "N/A"

                cell = f"{symbol} {answer:<10} ({tokens_str})"
            else:
                cell = "ERROR" if method in r.get('errors', {}) else "N/A"

            row += f" {cell:<30}"

        print(row)

    print("="*150)


def print_error_summary(results: List[Dict], methods: List[str]):
    """Print error summary"""
    error_counts = defaultdict(lambda: defaultdict(int))

    for r in results:
        for method in methods:
            if method in r.get('errors', {}):
                error_msg = r['errors'][method]
                # Truncate long error messages
                error_key = error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
                error_counts[method][error_key] += 1

    if not any(error_counts.values()):
        return

    print("\n" + "="*100)
    print(" "*35 + "ERROR SUMMARY")
    print("="*100)

    for method, errors in error_counts.items():
        if errors:
            print(f"\n{method.upper()}:")
            for error_msg, count in sorted(errors.items(), key=lambda x: -x[1]):
                print(f"  [{count}×] {error_msg}")

    print("="*100)


def export_to_csv(results: List[Dict], methods: List[str], output_path: str):
    """Export results to CSV file"""
    import csv

    with open(output_path, 'w', newline='') as f:
        # Build header
        fieldnames = ['qid', 'question', 'ground_truth']

        for method in methods:
            fieldnames.extend([
                f"{method}_answer",
                f"{method}_correct",
                f"{method}_tokens",
                f"{method}_time",
            ])

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {
                'qid': r['qid'],
                'question': r['question'],
                'ground_truth': r['ground_truth'],
            }

            for method in methods:
                row[f"{method}_answer"] = r.get(f"{method}_answer")
                row[f"{method}_correct"] = r.get(f"{method}_correct")
                row[f"{method}_tokens"] = r.get(f"{method}_tokens")
                row[f"{method}_time"] = r.get(f"{method}_time")

            writer.writerow(row)

    print(f"\nResults exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze confidence pooling benchmark results')
    parser.add_argument('files', nargs='+', help='JSONL result files to analyze')
    parser.add_argument('--methods', type=str, default='mv256,online,pooling',
                       help='Comma-separated list of methods to analyze')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed per-question results')
    parser.add_argument('--max-rows', type=int, default=None,
                       help='Maximum rows to show in detailed view')
    parser.add_argument('--export-csv', type=str, default=None,
                       help='Export results to CSV file')
    parser.add_argument('--output', type=str, default=None,
                       help='Save output to file instead of printing')

    args = parser.parse_args()

    # Parse methods
    methods = [m.strip() for m in args.methods.split(',')]

    # Load results
    print(f"Loading results from {len(args.files)} file(s)...")
    results = load_results(args.files)

    if not results:
        print("Error: No results loaded!", file=sys.stderr)
        return 1

    print(f"Loaded {len(results)} question results")

    # Redirect output if requested
    original_stdout = sys.stdout
    if args.output:
        sys.stdout = open(args.output, 'w')

    try:
        # Compute statistics
        stats = compute_statistics(results, methods)

        # Print summary
        print_summary_table(stats)

        # Print efficiency comparison
        print_efficiency_comparison(stats)

        # Print method-specific stats
        print_method_specific_stats(stats)

        # Print error summary
        print_error_summary(results, methods)

        # Print detailed results
        if args.detailed:
            print_detailed_results(results, methods, max_rows=args.max_rows)

        # Export to CSV
        if args.export_csv:
            export_to_csv(results, methods, args.export_csv)

    finally:
        if args.output:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"\nAnalysis saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
