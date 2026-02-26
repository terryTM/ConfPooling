#!/usr/bin/env python3
"""
Analyze majority vote results with LaTeX normalization for fair comparison.

Usage:
    python analyze_majority_vote.py
    python analyze_majority_vote.py --normalize
    python analyze_majority_vote.py --dataset hmmt2025
"""
import json
import os
import re
from collections import Counter
import argparse


def normalize_latex(ans: str) -> str:
    """Normalize LaTeX formatting for fair comparison."""
    if not ans:
        return ans

    # Remove common prefixes like "x=", "AD=", "AB=", "n=", etc.
    ans = re.sub(r'^[A-Za-z]+=', '', ans)

    # Remove displaystyle
    ans = re.sub(r'\\displaystyle\s*', '', ans)

    # Normalize fractions: \dfrac, \tfrac -> \frac
    ans = ans.replace(r'\dfrac', r'\frac')
    ans = ans.replace(r'\tfrac', r'\frac')

    # Normalize spacing commands
    ans = re.sub(r'\\[,;:!]', '', ans)  # \, \; \: \!
    ans = re.sub(r'\\(?:quad|qquad|enspace|thinspace|negthinspace)', '', ans)

    # Normalize \sqrt without braces: \sqrt6 -> \sqrt{6}
    ans = re.sub(r'\\sqrt(\d+)', r'\\sqrt{\1}', ans)
    ans = re.sub(r'\\sqrt([a-z])', r'\\sqrt{\1}', ans)

    # Remove text commands and their content
    ans = re.sub(r'\\text\{[^}]*\}', '', ans)
    ans = re.sub(r'\\textrm\{[^}]*\}', '', ans)
    ans = re.sub(r'\\textit\{[^}]*\}', '', ans)
    ans = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', ans)

    # Remove common suffixes like "feet", "cm", "degrees", etc.
    ans = re.sub(r'\s*(feet|foot|ft|cm|m|degrees|deg|units?)\s*$', '', ans, flags=re.IGNORECASE)

    # Normalize cdot vs times
    ans = ans.replace(r'\times', r'\cdot')

    # Remove all whitespace for comparison
    ans = re.sub(r'\s+', '', ans)

    # Remove leading/trailing special chars
    ans = ans.strip('.,;:')

    return ans


def analyze_dataset(name, dir_path, prefix, normalize=False):
    """Analyze a dataset with optional LaTeX normalization."""
    correct = 0
    total = 0
    details = []

    for qid in range(30):
        filepath = f"{dir_path}/{prefix}_{qid:03d}.jsonl"
        if not os.path.exists(filepath):
            continue

        traces = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not traces:
            continue

        ground_truth = traces[0].get('ground_truth', '')
        answers = [t.get('answer', '') for t in traces if t.get('answer')]

        if not answers:
            continue

        if normalize:
            # Normalize all answers for counting
            normalized_answers = [normalize_latex(a) for a in answers]
            counter = Counter(normalized_answers)
            majority_answer_norm, majority_count = counter.most_common(1)[0]

            # Find original form for display
            majority_answer = majority_answer_norm
            for a in answers:
                if normalize_latex(a) == majority_answer_norm:
                    majority_answer = a
                    break

            gt_norm = normalize_latex(ground_truth)
            is_correct = (majority_answer_norm == gt_norm)
        else:
            counter = Counter(answers)
            majority_answer, majority_count = counter.most_common(1)[0]
            is_correct = (majority_answer == ground_truth)

        if is_correct:
            correct += 1
        total += 1

        status = "✓" if is_correct else "✗"
        maj_display = majority_answer[:20] if len(majority_answer) <= 20 else majority_answer[:17] + "..."
        gt_display = ground_truth[:25] if len(ground_truth) <= 25 else ground_truth[:22] + "..."
        details.append({
            'qid': qid,
            'status': status,
            'majority': maj_display,
            'count': majority_count,
            'total': len(answers),
            'gt': gt_display,
            'is_correct': is_correct
        })

    return correct, total, details


def main():
    parser = argparse.ArgumentParser(description='Analyze majority vote with LaTeX normalization')
    parser.add_argument('--normalize', action='store_true', help='Apply LaTeX normalization')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to analyze (amc2025, amc2024, hmmt2025, brumo2025)')
    parser.add_argument('--base-dir', type=str,
                       default='/Users/terryma/Documents/conf_pooling/question_results',
                       help='Base directory for results')
    args = parser.parse_args()

    base = args.base_dir

    all_datasets = [
        ("AIME 2025", f"{base}/amc2025", "aime_2025"),
        ("AIME 2024", f"{base}/amc2024", "aime_2024"),
        ("HMMT 2025", f"{base}/hmmt2025", "hmmt_2025"),
        ("BRUMO 2025", f"{base}/brumo2025", "brumo_2025"),
    ]

    # Filter by dataset if specified
    if args.dataset:
        datasets = [(n, p, pf) for n, p, pf in all_datasets if args.dataset.lower() in p.lower()]
    else:
        datasets = all_datasets

    print("=" * 75)
    print("MAJORITY VOTE ANALYSIS" + (" (with LaTeX normalization)" if args.normalize else " (raw)"))
    print("=" * 75)

    summary = []

    for name, path, prefix in datasets:
        correct, total, details = analyze_dataset(name, path, prefix, normalize=args.normalize)

        if total == 0:
            print(f"\n{name}: No data found at {path}")
            continue

        print(f"\n{'='*75}")
        print(f"{name}")
        print(f"{'='*75}")
        print(f"{'Q':>3} {'Status':^6} {'Majority':<22} {'Votes':>10} {'Ground Truth':<25}")
        print("-" * 75)

        for d in details:
            print(f"{d['qid']:3d}   {d['status']}    {d['majority']:<22} ({d['count']:3d}/{d['total']:3d})  {d['gt']:<25}")

        print("-" * 75)
        print(f"Correct: {correct}/{total} ({100*correct/total:.1f}%)")

        summary.append((name, correct, total))

    # Print comparison if not normalizing
    if not args.normalize and not args.dataset:
        print("\n" + "=" * 75)
        print("COMPARISON: RAW vs NORMALIZED")
        print("=" * 75)
        print(f"{'Dataset':<15} {'Raw':>12} {'Normalized':>12} {'Fixed':>10}")
        print("-" * 75)

        for name, path, prefix in datasets:
            raw_correct, raw_total, _ = analyze_dataset(name, path, prefix, normalize=False)
            norm_correct, norm_total, _ = analyze_dataset(name, path, prefix, normalize=True)

            if raw_total == 0:
                continue

            fixed = norm_correct - raw_correct
            fixed_str = f"+{fixed}" if fixed > 0 else str(fixed)

            print(f"{name:<15} {raw_correct:>3}/{raw_total:<3} ({100*raw_correct/raw_total:4.1f}%)  "
                  f"{norm_correct:>3}/{norm_total:<3} ({100*norm_correct/norm_total:4.1f}%)  {fixed_str:>10}")


if __name__ == "__main__":
    main()
