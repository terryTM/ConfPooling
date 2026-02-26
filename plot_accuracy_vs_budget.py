#!/usr/bin/env python3
"""
Plot accuracy vs voting budget with confidence filtering.

Generates a multi-panel figure showing how accuracy scales with the number
of traces used for majority voting, under different filtering strategies.

Usage:
    python plot_accuracy_vs_budget.py
    python plot_accuracy_vs_budget.py --n-trials 500
    python plot_accuracy_vs_budget.py --output figure5.pdf
"""
import json
import os
import glob
import argparse
import numpy as np
from collections import Counter

# Import from existing modules
from check_results import normalize_latex, check_math_equivalent, extract_answer


# ── Data loading ──────────────────────────────────────────────────────────

def get_trace_confidence(trace):
    """Get confidence value for a trace (min of group_confidence array)."""
    gc = trace.get('group_confidence', [])
    if gc:
        return min(gc)
    old_gc = trace.get('old_group_confidence', [])
    if old_gc:
        return min(old_gc)
    return 0.0


def load_dataset(directory, prefix, n_questions=30):
    """Load Phase 1 traces for a dataset.

    Returns list of dicts, one per question:
        {'gt': str, 'answers': [str], 'confidences': [float]}
    """
    questions = []
    for qid in range(n_questions):
        filepath = os.path.join(directory, f"{prefix}_{qid:03d}.jsonl")
        if not os.path.exists(filepath):
            continue
        traces = []
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if not traces:
            continue
        gt = traces[0].get('ground_truth', '')
        answers = []
        confidences = []
        for t in traces:
            ans = t.get('answer', '')
            # Include failed/empty answers as a distinct placeholder so budgets align with trace counts.
            if not ans:
                ans = "__MISSING__"
            answers.append(ans)
            confidences.append(get_trace_confidence(t))
        if answers:
            questions.append({
                'qid': qid,
                'gt': gt,
                'answers': answers,
                'confidences': np.array(confidences),
            })
    return questions


def load_phase2_dataset(directory, prefix, suffix="_phase2_b256_p80_maxc", n_questions=30):
    """Load Phase 2 revised answers for a dataset.

    Returns list of dicts:
        {'qid': int, 'gt': str, 'answers': [str]}
    """
    questions = []
    for qid in range(n_questions):
        filepath = os.path.join(directory, f"{prefix}_{qid:03d}{suffix}.jsonl")
        if not os.path.exists(filepath):
            continue
        answers = []
        gt = None
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        ans = extract_answer(d.get('trace_2', ''))
                        if ans:
                            answers.append(ans)
                        # Try to get GT from a Phase 1 file if needed
                    except json.JSONDecodeError:
                        pass
        if answers:
            questions.append({
                'qid': qid,
                'answers': answers,
            })
    return questions


# ── Accuracy computation ──────────────────────────────────────────────────

def precompute_equivalence_classes(questions):
    """Pre-compute equivalence class IDs for each answer in each question.

    This avoids calling normalize_latex() and check_math_equivalent() inside
    the Monte Carlo loop. Each unique normalized answer gets an integer ID,
    and we record which ID matches the ground truth.

    Adds to each question dict:
        'class_ids': np.array of int, one per answer
        'gt_class': int (the class ID matching GT, or -1 if none)
        'class_confs': np.array of float, confidence per answer (unchanged)
    """
    for q in questions:
        gt = q['gt']
        answers = q['answers']

        # Map normalized forms to class IDs
        norm_to_id = {}
        next_id = 0
        class_ids = []
        originals = {}  # id -> first original answer

        for ans in answers:
            n = normalize_latex(ans)
            if n not in norm_to_id:
                norm_to_id[n] = next_id
                originals[next_id] = ans
                next_id += 1
            class_ids.append(norm_to_id[n])

        q['class_ids'] = np.array(class_ids, dtype=np.int32)

        # Find which class ID matches GT
        gt_class = -1
        for cid, orig_ans in originals.items():
            if check_math_equivalent(orig_ans, gt):
                gt_class = cid
                break
        q['gt_class'] = gt_class


def compute_accuracy_curve(questions, budgets, filter_pct=None, weighted=False,
                           n_trials=200, rng=None):
    """Compute accuracy for each budget via Monte Carlo sampling.

    Requires precompute_equivalence_classes() to have been called on questions.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Pre-compute filtered pools per question (invariant across budgets)
    pools = []
    for q in questions:
        cids = q['class_ids']
        confs = q['confidences']
        gt_class = q['gt_class']

        if filter_pct is not None:
            n_keep = max(1, int(len(cids) * filter_pct / 100))
            top_idx = np.argsort(confs)[::-1][:n_keep]
            pool_cids = cids[top_idx]
            pool_confs = confs[top_idx]
        else:
            pool_cids = cids
            pool_confs = confs

        pools.append((pool_cids, pool_confs, gt_class, len(pool_cids)))

    # Effective max pool size: minimum across all questions
    # (beyond this budget, every question is just using its full pool)
    max_pool = min(ps for _, _, _, ps in pools) if pools else 0

    accuracies = []
    for k in budgets:
        correct_total = 0
        count_total = 0
        for pool_cids, pool_confs, gt_class, pool_size in pools:
            sample_k = min(k, pool_size)

            for _ in range(n_trials):
                idx = rng.choice(pool_size, size=sample_k, replace=False)
                sampled_cids = pool_cids[idx]

                if weighted:
                    sampled_w = pool_confs[idx]
                    # Weighted majority: sum weights per class
                    weight_sums = {}
                    for cid, w in zip(sampled_cids, sampled_w):
                        weight_sums[cid] = weight_sums.get(cid, 0.0) + w
                    best_cid = max(weight_sums, key=weight_sums.get)
                else:
                    # Unweighted majority: count per class
                    counts = np.bincount(sampled_cids)
                    best_cid = int(np.argmax(counts))

                if best_cid == gt_class:
                    correct_total += 1
                count_total += 1

        accuracies.append(100.0 * correct_total / count_total if count_total > 0 else 0)
    return accuracies, max_pool


def compute_online_accuracy_curve(questions, budgets, percentile=80, num_calibration=32,
                                  weighted=False, n_trials=200, rng=None):
    """Compute DeepConf-Online style curve using a percentile cutoff from calibration."""
    if rng is None:
        rng = np.random.default_rng(42)

    pools = []
    for q in questions:
        cids = q['class_ids']
        confs = q['confidences']
        gt_class = q['gt_class']

        # Calibration threshold from first num_calibration traces
        if len(confs) >= num_calibration:
            calib = confs[:num_calibration]
            threshold = np.percentile(calib, percentile)
        else:
            threshold = 0.0

        keep_idx = np.where(confs >= threshold)[0]
        if keep_idx.size == 0:
            # Ensure at least one trace survives
            keep_idx = np.array([int(np.argmax(confs))])

        pool_cids = cids[keep_idx]
        pool_confs = confs[keep_idx]
        pools.append((pool_cids, pool_confs, gt_class, len(pool_cids)))

    max_pool = min(ps for _, _, _, ps in pools) if pools else 0

    accuracies = []
    for k in budgets:
        correct_total = 0
        count_total = 0
        for pool_cids, pool_confs, gt_class, pool_size in pools:
            sample_k = min(k, pool_size)

            for _ in range(n_trials):
                idx = rng.choice(pool_size, size=sample_k, replace=False)
                sampled_cids = pool_cids[idx]

                if weighted:
                    sampled_w = pool_confs[idx]
                    weight_sums = {}
                    for cid, w in zip(sampled_cids, sampled_w):
                        weight_sums[cid] = weight_sums.get(cid, 0.0) + w
                    best_cid = max(weight_sums, key=weight_sums.get)
                else:
                    counts = np.bincount(sampled_cids)
                    best_cid = int(np.argmax(counts))

                if best_cid == gt_class:
                    correct_total += 1
                count_total += 1

        accuracies.append(100.0 * correct_total / count_total if count_total > 0 else 0)

    return accuracies, max_pool


def compute_phase2_accuracy_curve(p1_questions, p2_questions, budgets, n_trials=200, rng=None):
    """Compute Phase 2 accuracy curve.

    Uses Phase 1 ground truth matched by qid.
    Requires precompute_equivalence_classes() on p2_questions (with GT filled in).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Build GT lookup from Phase 1
    gt_lookup = {q['qid']: q['gt'] for q in p1_questions}

    # Fill GT into Phase 2 questions and precompute
    valid_p2 = []
    for q in p2_questions:
        if q['qid'] in gt_lookup:
            q['gt'] = gt_lookup[q['qid']]
            valid_p2.append(q)

    if not valid_p2:
        return None

    # Precompute equivalence classes for Phase 2
    precompute_equivalence_classes(valid_p2)
    # Add dummy confidences (uniform)
    for q in valid_p2:
        q['confidences'] = np.ones(len(q['answers']))

    accs, max_pool = compute_accuracy_curve(valid_p2, budgets, filter_pct=None,
                                            weighted=False, n_trials=n_trials, rng=rng)
    return accs, max_pool


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_figure(all_results, budgets, output_path="figure_accuracy_vs_budget.pdf"):
    """Create the multi-panel figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_panels = len(all_results)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.8), sharey=False)
    if n_panels == 1:
        axes = [axes]

    def truncate(accs_and_pool, budgets):
        """Truncate curve at max pool size. Returns (x_vals, y_vals)."""
        accs, max_pool = accs_and_pool
        x, y = [], []
        for b, a in zip(budgets, accs):
            if b <= max_pool:
                x.append(b)
                y.append(a)
        return x, y

    for ax, (title, curves) in zip(axes, all_results):
        # No Voting baseline
        if 'no_voting' in curves:
            ax.axhline(y=curves['no_voting'], color='black', linestyle='--',
                       linewidth=1.5, label='No Voting', zorder=1)

        # Majority Voting
        if 'majority' in curves:
            bx, by = truncate(curves['majority'], budgets)
            ax.plot(bx, by, color='#DAA520', linewidth=2,
                    marker='o', markersize=3, label='Majority Voting', zorder=2)

        # DeepConf-Online (percentile threshold)
        if 'online' in curves:
            bx, by = truncate(curves['online'], budgets)
            ax.plot(bx, by, color='#9467BD', linewidth=2,
                    marker='D', markersize=3.5, label='DeepConf-Online', zorder=4)

        # Phase 2 Consensus
        if 'phase2_maxc' in curves and curves['phase2_maxc'] is not None:
            bx, by = truncate(curves['phase2_maxc'], budgets)
            ax.plot(bx, by, color='#D62728', linewidth=2,
                    marker='^', markersize=4, label='PACER', zorder=6)

        ax.set_xscale('log')
        ax.set_xlabel('Traces', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel('Accuracy (%)', fontsize=11)

    # Single legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    # Merge handles from all axes (in case some curves only appear in certain panels)
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)

    fig.legend(handles, labels, loc='upper center', ncol=min(6, len(labels)),
               fontsize=8.5, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure to {output_path}")

    # Also save PNG
    png_path = output_path.replace('.pdf', '.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=200, bbox_inches='tight')
        print(f"Saved PNG to {png_path}")


# ── Main ──────────────────────────────────────────────────────────────────

DATASETS = [
    {
        'name': 'AIME25',
        'p1_dir': 'run2/aime2025/traces_jsonl_aime_2025_run2',
        'prefix': 'aime_2025',
        'p2_dir': 'phase2_aime2025_maxc',
    },
    {
        'name': 'BRUMO25',
        'p1_dir': 'run2/brumo2025/traces_jsonl_brumo_2025_run2',
        'prefix': 'brumo_2025',
        'p2_dir': 'phase2_brumo_maxc',
    },
    {
        'name': 'HMMT_FEB25',
        'p1_dir': 'run2/hmmt2025',
        'prefix': 'hmmt_2025',
        'p2_dir': 'phase2_hmmt_maxc',
    },
]

BUDGETS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def average_curve_runs(run_results):
    """Average (accuracies, max_pool) tuples across runs.

    Each element of run_results is (accs_list, max_pool).
    Returns (averaged_accs_list, max_pool) — max_pool is the same across runs.
    """
    all_accs = np.array([r[0] for r in run_results])
    avg_accs = all_accs.mean(axis=0).tolist()
    max_pool = run_results[0][1]  # same for all runs
    return avg_accs, max_pool


def main():
    parser = argparse.ArgumentParser(description='Plot accuracy vs voting budget')
    parser.add_argument('--n-trials', type=int, default=200, help='Monte Carlo trials per budget per run')
    parser.add_argument('--n-runs', type=int, default=10, help='Number of independent runs to average')
    parser.add_argument('--output', default='figure_accuracy_vs_budget.pdf', help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--online-percentile', type=float, default=80.0, help='DeepConf-Online percentile cutoff')
    parser.add_argument('--online-calibration', type=int, default=32, help='DeepConf-Online calibration trace count')
    args = parser.parse_args()

    all_results = []

    for ds in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing {ds['name']}...")
        print(f"{'='*60}")

        # Load Phase 1
        questions = load_dataset(ds['p1_dir'], ds['prefix'])
        print(f"  Loaded {len(questions)} questions from Phase 1")

        # Pre-compute equivalence classes (expensive, but only once)
        print(f"  Pre-computing equivalence classes...")
        precompute_equivalence_classes(questions)

        # Load Phase 2 if available (also only once)
        p2_questions = None
        if ds['p2_dir'] and os.path.isdir(ds['p2_dir']):
            p2_questions = load_phase2_dataset(ds['p2_dir'], ds['prefix'])
            n_p2 = len(p2_questions) if p2_questions else 0
            n_p1 = len(questions)
            if not p2_questions or n_p2 < n_p1 * 0.8:
                print(f"  Phase 2: only {n_p2}/{n_p1} questions available, skipping curve")
                p2_questions = None
            else:
                print(f"  Loaded {n_p2} Phase 2 questions")

        # Collect results across runs
        run_no_voting = []
        run_majority = []
        run_online = []
        run_phase2 = []

        for run_i in range(args.n_runs):
            rng = np.random.default_rng(args.seed + run_i)
            print(f"  Run {run_i+1}/{args.n_runs} (seed={args.seed + run_i})...")

            nv_accs, nv_pool = compute_accuracy_curve(questions, [1], n_trials=args.n_trials, rng=rng)
            run_no_voting.append(nv_accs[0])

            run_majority.append(compute_accuracy_curve(
                questions, BUDGETS, filter_pct=None, n_trials=args.n_trials, rng=rng))

            run_online.append(compute_online_accuracy_curve(
                questions, BUDGETS, percentile=args.online_percentile,
                num_calibration=args.online_calibration, n_trials=args.n_trials, rng=rng))

            if p2_questions is not None:
                p2_result = compute_phase2_accuracy_curve(
                    questions, p2_questions, BUDGETS, n_trials=args.n_trials, rng=rng)
                if p2_result:
                    run_phase2.append(p2_result)

        # Average across runs
        curves = {}
        curves['no_voting'] = float(np.mean(run_no_voting))
        curves['majority'] = average_curve_runs(run_majority)
        curves['online'] = average_curve_runs(run_online)

        print(f"  No Voting: {curves['no_voting']:.1f}%")
        print(f"  Majority Voting @256: {curves['majority'][0][-1]:.1f}%")
        print(f"  DeepConf-Online (p{int(args.online_percentile)}): {curves['online'][0][-1]:.1f}%")
        if run_phase2:
            curves['phase2_maxc'] = average_curve_runs(run_phase2)
            print(f"  Phase 2 @max: {curves['phase2_maxc'][0][-1]:.1f}%")

        all_results.append((ds['name'], curves))

    # Plot
    plot_figure(all_results, BUDGETS, output_path=args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
