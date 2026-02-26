#!/usr/bin/env python3
"""
Check majority vote accuracy with LaTeX normalization.

Usage:
    python check_results.py <path>                    # Check file or directory
    python check_results.py traces_jsonl/             # Check all jsonl in directory
    python check_results.py results/aime_2024_000.jsonl  # Check single file
    python check_results.py *.jsonl                   # Check multiple files (shell glob)
"""
import json
import os
import re
import sys
import glob
from collections import Counter


def normalize_latex(s):
    """Normalize LaTeX for fair comparison."""
    if not s:
        return s
    # Remove spacing commands first so prefixes like \,x = ... become x = ...
    s = re.sub(r'\\[,;:! ]', '', s)
    # Remove \in prefix (e.g., "a\in\{...")
    s = re.sub(r'^[A-Za-z]\s*\\in\s*', '', s)
    # Remove variable prefixes: "f(3)=", "XP=", "n=", "30 \times 8 = 240"
    # Take content after last "=" if the prefix before first "=" is simple (no LaTeX commands except \times)
    if '=' in s:
        before_first_eq = s.split('=')[0]
        # Allow \times, \cdot in the prefix (arithmetic expressions)
        cleaned_prefix = re.sub(r'\\(times|cdot)', '', before_first_eq)
        if '\\' not in cleaned_prefix:
            s = s.split('=')[-1].strip()
    # Remove \displaystyle
    s = re.sub(r'\\displaystyle\s*', '', s)
    # Normalize fractions
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    # Remove spacing commands including \, \; \: \! and also '\ ' (backslash-space)
    s = re.sub(r'\\[,;:! ]', '', s)
    # Remove degree symbols: ^{\circ}, ^\circ, °
    s = re.sub(r'\^\{?\\circ\}?', '', s)
    s = s.replace('°', '')
    # Normalize \sqrt without braces: \sqrt6 -> \sqrt{6}
    s = re.sub(r'\\sqrt(\d)', r'\\sqrt{\1}', s)
    s = re.sub(r'\\sqrt([a-z])', r'\\sqrt{\1}', s)
    # Remove \text{...}
    s = re.sub(r'\\text\{[^}]*\}', '', s)
    # Remove \mathbb{E}[...] and similar decorators
    s = re.sub(r'\\mathbb\{[^}]*\}\[[^\]]*\]=?', '', s)
    # Remove \approx and everything after it
    s = re.sub(r'\\approx.*$', '', s)
    # Remove set delimiters: \Bigl\{, \Bigr\}, \left\{, \right\}, \{, \}
    s = re.sub(r'\\(?:Bigl?|Bigr?|left|right)\\[{}]', '', s)
    s = re.sub(r'\\[{}]', '', s)
    # Normalize \frac without braces: \frac19 -> \frac{1}{9}
    s = re.sub(r'\\frac(\d)(\d)', r'\\frac{\1}{\2}', s)
    # Remove whitespace
    s = re.sub(r'\s+', '', s)
    # Normalize commas in lists (for multi-value answers)
    s = re.sub(r',+', ',', s)
    return s.strip('.,;:')


def try_evaluate_numeric(s):
    """Try to evaluate a LaTeX expression to a numeric value."""
    import math
    if not s:
        return None
    try:
        expr = s
        expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))', expr)
        expr = re.sub(r'\\sqrt\{([^}]+)\}', r'math.sqrt(\1)', expr)
        expr = re.sub(r'\\cdot', '*', expr)
        expr = re.sub(r'\\times', '*', expr)
        expr = expr.replace('^', '**')
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
    def sort_mult_terms(expr):
        # Insert * for implicit multiplication
        expr = re.sub(r'(\!)(\d|\\)', r'\1*\2', expr)
        expr = re.sub(r'(\})(\d|\\)', r'\1*\2', expr)
        expr = re.sub(r'(\d)(\\)', r'\1*\2', expr)
        terms = re.split(r'\\cdot|\*', expr)
        return '*'.join(sorted([t.strip() for t in terms if t.strip()]))

    if sort_mult_terms(na) == sort_mult_terms(nb):
        return True

    # Check rationalized forms: a/sqrt(b) = a*sqrt(b)/b
    def try_rationalize(expr):
        match = re.match(r'\\frac\{([^}]+)\}\{\\sqrt\{([^}]+)\}\}', expr)
        if match:
            num, denom_sqrt = match.groups()
            return f'\\frac{{{num}\\sqrt{{{denom_sqrt}}}}}{{{denom_sqrt}}}'
        return None

    rationalized_a = try_rationalize(na)
    rationalized_b = try_rationalize(nb)
    if rationalized_a and normalize_latex(rationalized_a) == nb:
        return True
    if rationalized_b and normalize_latex(rationalized_b) == na:
        return True

    # Check sqrt form equivalence via numeric evaluation
    import math

    def eval_sqrt_expr(expr):
        try:
            match = re.match(r'\\sqrt\{\\frac\{(\d+)\}\{(\d+)\}\}', expr)
            if match:
                num, denom = int(match.group(1)), int(match.group(2))
                return math.sqrt(num / denom)
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
    if not isinstance(text, str):
        return None
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
            return a.strip()
        else:
            return ans.split("$")[0].strip()
    return None


def check_file(filepath, normalize=True):
    """Check accuracy for a single JSONL file."""
    if not os.path.exists(filepath):
        return None

    traces = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not traces:
        return None

    # Get ground truth from first trace
    gt = traces[0].get('ground_truth', '')

    # Try to get answers - check multiple possible field names
    answers = []
    for t in traces:
        # Direct answer field
        ans = t.get('answer', '')
        if ans:
            answers.append(ans)
            continue
        # Phase 2 format: extract from trace_2
        trace_2 = t.get('trace_2', '')
        if trace_2:
            extracted = extract_answer(trace_2)
            if extracted:
                answers.append(extracted)

    if not answers:
        return None

    if normalize:
        norm_answers = [normalize_latex(a) for a in answers]
        counter = Counter(norm_answers)
        majority_norm, count = counter.most_common(1)[0]
        # Find original form
        majority = next((a for a in answers if normalize_latex(a) == majority_norm), majority_norm)
        is_correct = check_math_equivalent(majority, gt)
    else:
        counter = Counter(answers)
        majority, count = counter.most_common(1)[0]
        is_correct = (majority == gt)

    return {
        'filepath': filepath,
        'correct': is_correct,
        'majority': majority,
        'gt': gt,
        'votes': f"{count}/{len(answers)}",
        'total_traces': len(answers),
        'majority_count': count,
    }


def find_jsonl_files(path):
    """Find all JSONL files from a path (file, directory, or glob pattern)."""
    files = []

    if os.path.isfile(path):
        files.append(path)
    elif os.path.isdir(path):
        # Recursively find all .jsonl files
        for root, _, filenames in os.walk(path):
            for f in filenames:
                if f.endswith('.jsonl'):
                    files.append(os.path.join(root, f))
    else:
        # Try as glob pattern
        files = glob.glob(path)

    return sorted(files)


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_results.py <path> [path2] ...")
        print("  <path> can be a file, directory, or glob pattern")
        print("\nExamples:")
        print("  python check_results.py traces_jsonl/")
        print("  python check_results.py results/aime_2024_000.jsonl")
        print("  python check_results.py 'traces_jsonl/**/*.jsonl'")
        sys.exit(1)

    # Collect all files from all arguments
    all_files = []
    for arg in sys.argv[1:]:
        all_files.extend(find_jsonl_files(arg))

    if not all_files:
        print(f"No JSONL files found")
        sys.exit(1)

    print("=" * 70)
    print("MAJORITY VOTE ACCURACY (with LaTeX normalization)")
    print("=" * 70)
    print(f"\nChecking {len(all_files)} file(s)...\n")

    results = []
    for filepath in all_files:
        result = check_file(filepath, normalize=True)
        if result:
            results.append(result)

    if not results:
        print("No valid results found in any files.")
        sys.exit(1)

    # Print per-file results
    print(f"{'File':<50} {'Status':^6} {'Majority':<30} {'GT':<30} {'Votes':>8}")
    print("-" * 130)

    correct = 0
    total = len(results)
    for r in results:
        filename = os.path.basename(r['filepath'])
        status = "✓" if r['correct'] else "✗"
        majority_short = r['majority'][:28] + ".." if len(r['majority']) > 30 else r['majority']
        gt_short = r['gt'][:28] + ".." if len(r['gt']) > 30 else r['gt']
        print(f"{filename:<50} {status:^6} {majority_short:<30} {gt_short:<30} {r['votes']:>8}")
        if r['correct']:
            correct += 1

    print("-" * 70)
    print(f"\nSUMMARY: {correct}/{total} correct ({100*correct/total:.1f}%)")

    # Show incorrect ones
    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        print(f"\nIncorrect ({len(incorrect)}):")
        for r in incorrect:
            filename = os.path.basename(r['filepath'])
            print(f"  {filename}: got '{r['majority']}', expected '{r['gt']}'")


if __name__ == "__main__":
    main()
