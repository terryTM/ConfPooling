#!/usr/bin/env python3
"""
Compare Phase 1 vs Phase 2 results for a dataset.

Usage:
    python compare_phases.py --phase1-dir run2/hmmt2025 --phase2-dir phase2_results --prefix hmmt_2025
    python compare_phases.py --phase1-dir run2/hmmt2025 --phase2-dir . --prefix hmmt_2025 --suffix _phase2_b256_p80_maxc
"""
import json
import os
import re
import sys
import argparse
import glob
from collections import Counter


def normalize_latex(s):
    """Normalize LaTeX for fair comparison."""
    if not s:
        return s
    # Remove variable prefixes like "XP=", "MD=", "AB=", "x=", "n=" at start
    s = re.sub(r'^[A-Za-z]+\s*=\s*', '', s)
    s = re.sub(r'\\displaystyle\s*', '', s)
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    # Remove spacing commands including \, \; \: \! and also '\ ' (backslash-space)
    s = re.sub(r'\\[,;:! ]', '', s)
    s = re.sub(r'\\sqrt(\d)', r'\\sqrt{\1}', s)
    s = re.sub(r'\\sqrt([a-z])', r'\\sqrt{\1}', s)
    s = re.sub(r'\\text\{[^}]*\}', '', s)
    s = re.sub(r'\s+', '', s)
    # Normalize commas in lists (for multi-value answers like Q9)
    s = re.sub(r',+', ',', s)
    return s.strip('.,;:')


def try_evaluate_numeric(s):
    """Try to evaluate a LaTeX expression to a numeric value."""
    import math
    if not s:
        return None
    try:
        # Clean up for eval
        expr = s
        expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))', expr)
        expr = re.sub(r'\\sqrt\{([^}]+)\}', r'math.sqrt(\1)', expr)
        expr = re.sub(r'\\cdot', '*', expr)
        expr = expr.replace('^', '**')
        # Handle factorial
        if '!' in expr:
            expr = re.sub(r'(\d+)!', r'math.factorial(\1)', expr)
        result = eval(expr)
        return float(result)
    except:
        return None


def check_math_equivalent(a, b):
    """Check if two expressions are mathematically equivalent."""
    if not a or not b:
        return False

    # First try direct string match after normalization
    na, nb = normalize_latex(a), normalize_latex(b)
    if na == nb:
        return True

    # For multi-value answers (comma-separated), sort and compare
    if ',' in na or ',' in nb:
        parts_a = sorted([p.strip() for p in na.split(',')])
        parts_b = sorted([p.strip() for p in nb.split(',')])
        if parts_a == parts_b:
            return True

    # Try numeric evaluation
    va, vb = try_evaluate_numeric(na), try_evaluate_numeric(nb)
    if va is not None and vb is not None:
        if abs(va - vb) < 1e-6 or (vb != 0 and abs((va - vb) / vb) < 1e-6):
            return True

    # Check multiplication reordering (e.g., "26!*2^{25}" vs "2^{25}*26!")
    # Sort multiplication terms
    def sort_mult_terms(expr):
        # First, insert * for implicit multiplication:
        # After ! followed by digit or backslash: 26!2^{25} -> 26!*2^{25}
        # After } followed by digit or backslash: 2^{25}26! -> 2^{25}*26!
        expr = re.sub(r'(\!)(\d|\\)', r'\1*\2', expr)
        expr = re.sub(r'(\})(\d|\\)', r'\1*\2', expr)
        expr = re.sub(r'(\d)(\\)', r'\1*\2', expr)
        # Split by \cdot or * and sort
        terms = re.split(r'\\cdot|\*', expr)
        return '*'.join(sorted([t.strip() for t in terms if t.strip()]))

    if sort_mult_terms(na) == sort_mult_terms(nb):
        return True

    # Check rationalized forms: a/sqrt(b) = a*sqrt(b)/b
    # Pattern: \frac{X}{\sqrt{Y}} should equal \frac{X\sqrt{Y}}{Y}
    def try_rationalize(expr):
        match = re.match(r'\\frac\{([^}]+)\}\{\\sqrt\{([^}]+)\}\}', expr)
        if match:
            num, denom_sqrt = match.groups()
            # Rationalized form: \frac{num*sqrt(denom_sqrt)}{denom_sqrt}
            return f'\\frac{{{num}\\sqrt{{{denom_sqrt}}}}}{{{denom_sqrt}}}'
        return None

    rationalized_a = try_rationalize(na)
    rationalized_b = try_rationalize(nb)

    if rationalized_a and normalize_latex(rationalized_a) == nb:
        return True
    if rationalized_b and normalize_latex(rationalized_b) == na:
        return True

    # Check sqrt form equivalence: sqrt(a/b) = sqrt(a*b)/b
    # \sqrt{\frac{95}{24}} = \frac{\sqrt{570}}{12} since sqrt(95/24) = sqrt(95*24)/24 = sqrt(2280)/24
    # Actually: sqrt(95/24) = sqrt(95)/sqrt(24) = sqrt(95)/(2*sqrt(6))
    # Rationalized: sqrt(95)*sqrt(24)/24 = sqrt(2280)/24 = sqrt(570)/12 (since 2280/4 = 570)
    import math

    # Try to evaluate both numerically for sqrt expressions
    def eval_sqrt_expr(expr):
        try:
            # Handle \sqrt{\frac{a}{b}}
            match = re.match(r'\\sqrt\{\\frac\{(\d+)\}\{(\d+)\}\}', expr)
            if match:
                num, denom = int(match.group(1)), int(match.group(2))
                return math.sqrt(num / denom)

            # Handle \frac{\sqrt{a}}{b}
            match = re.match(r'\\frac\{\\sqrt\{(\d+)\}\}\{(\d+)\}', expr)
            if match:
                sqrt_val, denom = int(match.group(1)), int(match.group(2))
                return math.sqrt(sqrt_val) / denom

            return None
        except:
            return None

    sqrt_va, sqrt_vb = eval_sqrt_expr(na), eval_sqrt_expr(nb)
    if sqrt_va is not None and sqrt_vb is not None:
        if abs(sqrt_va - sqrt_vb) < 1e-6:
            return True

    return False


def extract_answer(text):
    """Extract boxed answer from text."""
    if not isinstance(text, str) or 'boxed' not in text:
        return None
    ans = text.split('boxed')[-1]
    if not ans:
        return None
    if ans[0] == '{':
        stack = 1
        a = ''
        for c in ans[1:]:
            if c == '{':
                stack += 1
                a += c
            elif c == '}':
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
        return a.strip()
    return ans.split('$')[0].strip()


def get_majority_vote(answers):
    """Get majority vote from list of answers."""
    if not answers:
        return None, 0, 0
    norm_answers = [normalize_latex(a) for a in answers if a]
    if not norm_answers:
        return None, 0, 0
    counter = Counter(norm_answers)
    majority_norm, count = counter.most_common(1)[0]
    # Find original form
    majority = next((a for a in answers if normalize_latex(a) == majority_norm), majority_norm)
    return majority, count, len(norm_answers)


def check_correct(answer, ground_truth):
    """Check if answer matches ground truth (with normalization and math equivalence)."""
    if not answer or not ground_truth:
        return False
    return check_math_equivalent(answer, ground_truth)


def load_phase1_file(filepath):
    """Load Phase 1 traces and return answers + ground truth."""
    if not os.path.exists(filepath):
        return None, None, []

    traces = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not traces:
        return None, None, []

    gt = traces[0].get('ground_truth', '')
    answers = [t.get('answer', '') for t in traces if t.get('answer')]
    return gt, traces[0].get('question', ''), answers


def load_phase2_file(filepath):
    """Load Phase 2 traces and extract revised answers."""
    if not os.path.exists(filepath):
        return []

    answers = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    d = json.loads(line)
                    ans = extract_answer(d.get('trace_2', ''))
                    if ans:
                        answers.append(ans)
                except json.JSONDecodeError:
                    pass

    return answers


def main():
    parser = argparse.ArgumentParser(description='Compare Phase 1 vs Phase 2 results')
    parser.add_argument('--phase1-dir', required=True, help='Directory with Phase 1 traces')
    parser.add_argument('--phase2-dir', required=True, help='Directory with Phase 2 traces')
    parser.add_argument('--prefix', required=True, help='Dataset prefix (e.g., hmmt_2025)')
    parser.add_argument('--suffix', default='_phase2_b256_p80_maxc', help='Phase 2 file suffix')
    parser.add_argument('--num-questions', type=int, default=30, help='Number of questions')

    args = parser.parse_args()

    print("=" * 100)
    print("PHASE 1 vs PHASE 2 COMPARISON")
    print("=" * 100)
    print(f"Phase 1 dir: {args.phase1_dir}")
    print(f"Phase 2 dir: {args.phase2_dir}")
    print(f"Prefix: {args.prefix}, Suffix: {args.suffix}")
    print("=" * 100)
    print()

    results = []

    for qid in range(args.num_questions):
        # Load Phase 1
        p1_file = f"{args.phase1_dir}/{args.prefix}_{qid:03d}.jsonl"
        gt, question, p1_answers = load_phase1_file(p1_file)

        if gt is None:
            continue

        # Load Phase 2
        p2_file = f"{args.phase2_dir}/{args.prefix}_{qid:03d}{args.suffix}.jsonl"
        p2_answers = load_phase2_file(p2_file)

        # Get majority votes
        p1_majority, p1_count, p1_total = get_majority_vote(p1_answers)
        p2_majority, p2_count, p2_total = get_majority_vote(p2_answers)

        # Check correctness
        p1_correct = check_correct(p1_majority, gt)
        p2_correct = check_correct(p2_majority, gt) if p2_majority else None

        # Determine improvement
        if p2_correct is None:
            improved = "N/A"
        elif p1_correct and p2_correct:
            improved = "Both ✓"
        elif not p1_correct and p2_correct:
            improved = "Yes!"
        elif p1_correct and not p2_correct:
            improved = "Regressed"
        else:
            improved = "No"

        results.append({
            'qid': qid,
            'gt': gt,
            'p1_majority': p1_majority,
            'p1_votes': f"{p1_count}/{p1_total}",
            'p1_correct': p1_correct,
            'p2_majority': p2_majority,
            'p2_votes': f"{p2_count}/{p2_total}" if p2_total > 0 else "N/A",
            'p2_correct': p2_correct,
            'improved': improved,
        })

    # Print results table
    print(f"{'Q':>3} {'P1':^3} {'P2':^3} {'Improved':<10} {'P1 Votes':>10} {'P2 Votes':>10}  {'P1 Majority':<50} {'P2 Majority':<50} {'GT':<50}")
    print("-" * 180)

    p1_total_correct = 0
    p2_total_correct = 0
    improved_count = 0
    regressed_count = 0

    for r in results:
        p1_status = "✓" if r['p1_correct'] else "✗"
        p2_status = "✓" if r['p2_correct'] else ("✗" if r['p2_correct'] is not None else "-")

        p1_maj = r['p1_majority'] or ''
        p2_maj = r['p2_majority'] or ''
        gt = r['gt'] or ''

        print(f"{r['qid']:3d} {p1_status:^3} {p2_status:^3} {r['improved']:<10} {r['p1_votes']:>10} {r['p2_votes']:>10}  {p1_maj:<50} {p2_maj:<50} {gt:<50}")

        if r['p1_correct']:
            p1_total_correct += 1
        if r['p2_correct']:
            p2_total_correct += 1
        if r['improved'] == "Yes!":
            improved_count += 1
        if r['improved'] == "Regressed":
            regressed_count += 1

    print("-" * 130)
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    total = len(results)
    print(f"Phase 1 Accuracy: {p1_total_correct}/{total} ({100*p1_total_correct/total:.1f}%)")
    print(f"Phase 2 Accuracy: {p2_total_correct}/{total} ({100*p2_total_correct/total:.1f}%)")
    print(f"Improved (P1 wrong -> P2 correct): {improved_count}")
    print(f"Regressed (P1 correct -> P2 wrong): {regressed_count}")
    print(f"Net change: {'+' if improved_count >= regressed_count else ''}{improved_count - regressed_count}")
    print()

    # List improvements and regressions
    if improved_count > 0:
        print("Improvements:")
        for r in results:
            if r['improved'] == "Yes!":
                print(f"  Q{r['qid']}: {r['p1_majority'][:40]} -> {r['p2_majority'][:40]} (GT: {r['gt'][:30]})")

    if regressed_count > 0:
        print("\nRegressions:")
        for r in results:
            if r['improved'] == "Regressed":
                print(f"  Q{r['qid']}: {r['p1_majority'][:40]} -> {r['p2_majority'][:40]} (GT: {r['gt'][:30]})")


if __name__ == "__main__":
    main()
