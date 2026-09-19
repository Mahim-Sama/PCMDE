#!/usr/bin/env python3
"""
Aircraft Ablation Analysis for PCMDE Framework
Produces statistical tables, comparisons, and diagnostic outputs for ICML rebuttal.

Usage:
    python aircraft_ablation_analysis.py --data_dir /path/to/json/files

Expected files in data_dir:
    - aircraft_ablation_full_pipeline.json
    - aircraft_ablation_rules_only.json
    - aircraft_ablation_llm_only.json
    - aircraft_ablation_vlm_only.json
    - ensembled_result_aircraft_yolo_only.json  (detection-level, optional)
    - ensembled_result_aircraft_vlm_only.json    (detection-level, optional)

If you also ran the YOLO-only variant through the full evaluation pipeline
(not just detection), include: aircraft_ablation_yolo_only.json
"""

import json
import os
import sys
import argparse
import numpy as np
from collections import defaultdict

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed. Skipping statistical tests. Install with: pip install scipy")


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_stats(scores):
    """Compute summary statistics matching paper's Table 1/2 format."""
    scores = np.array(scores, dtype=float)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)  # sample std
    mn = np.min(scores)
    mx = np.max(scores)
    rng = mx - mn
    cv = (std / mean * 100) if mean != 0 else 0
    median = np.median(scores)
    return {
        'Mean': round(mean, 2),
        'Std': round(std, 2),
        'Min': round(mn, 2),
        'Max': round(mx, 2),
        'Range': round(rng, 2),
        'CV(%)': round(cv, 2),
        'Median': round(median, 2),
        'N': len(scores)
    }


def extract_overall_scores(data):
    """Extract list of overall_score from ablation JSON results."""
    return [r['overall_score'] for r in data['results']]


def extract_subscores(data):
    """Extract sub-component scores for deeper analysis."""
    results = data['results']
    out = {
        'presence': [],
        'spatial': [],
        'relational': [],
        'caption': [],
        'llm': [],
        'rules': [],
        'overall': []
    }
    for r in results:
        out['presence'].append(r.get('presence_count_score', None))
        out['spatial'].append(r.get('spatial_rules_score', None))
        out['relational'].append(r.get('relational_score', None))
        out['caption'].append(r.get('caption_alignment_score', None))
        out['llm'].append(r.get('llm_knowledge_score', None))
        out['rules'].append(r.get('physical_rules_score', None))
        out['overall'].append(r.get('overall_score', None))
    return out


def extract_pass_fail(data):
    """Extract pass/fail labels."""
    return [r['pass_fail'] for r in data['results']]


def extract_image_ids(data):
    """Extract image IDs in order."""
    return [r['image_id'] for r in data['results']]


def print_separator(char='=', length=80):
    print(char * length)


def print_section(title):
    print()
    print_separator()
    print(f"  {title}")
    print_separator()


def format_table_row(label, stats_dict, cols):
    """Format a single row for display."""
    vals = [str(stats_dict.get(c, '')) for c in cols]
    return f"  {label:<25s} " + " ".join(f"{v:>10s}" for v in vals)


# ========== ANALYSIS FUNCTIONS ==========

def analysis_1_summary_table(ablation_data):
    """Table: Mean, Std, Min, Max, Range, CV for each ablation."""
    print_section("1. ABLATION SUMMARY STATISTICS (Overall Scores)")
    
    cols = ['N', 'Mean', 'Std', 'Min', 'Max', 'Range', 'CV(%)', 'Median']
    header = f"  {'Variant':<25s} " + " ".join(f"{c:>10s}" for c in cols)
    print(header)
    print("  " + "-" * (25 + 11 * len(cols)))
    
    results = {}
    for name, data in ablation_data.items():
        scores = extract_overall_scores(data)
        st = compute_stats(scores)
        results[name] = st
        print(format_table_row(name, st, cols))
    
    print()
    return results


def analysis_2_pass_fail(ablation_data):
    """Pass/Fail rates and agreement with full pipeline."""
    print_section("2. PASS/FAIL ANALYSIS")
    
    # Get full pipeline as reference
    full_pf = None
    all_pf = {}
    for name, data in ablation_data.items():
        pf = extract_pass_fail(data)
        all_pf[name] = pf
        if 'full' in name.lower():
            full_pf = pf
    
    # Pass rates
    print("\n  Pass Rates:")
    print(f"  {'Variant':<25s} {'Passed':>8s} {'Failed':>8s} {'Rate':>8s}")
    print("  " + "-" * 52)
    for name, pf in all_pf.items():
        passed = sum(1 for x in pf if x == 'PASS')
        failed = len(pf) - passed
        rate = passed / len(pf) * 100
        print(f"  {name:<25s} {passed:>8d} {failed:>8d} {rate:>7.1f}%")
    
    # Agreement with full pipeline
    if full_pf is not None:
        print(f"\n  Agreement with Full Pipeline:")
        print(f"  {'Variant':<25s} {'Agreement':>10s} {'Disagree':>10s} {'Rate':>8s}")
        print("  " + "-" * 56)
        for name, pf in all_pf.items():
            if 'full' in name.lower():
                continue
            agree = sum(1 for a, b in zip(full_pf, pf) if a == b)
            disagree = len(pf) - agree
            rate = agree / len(pf) * 100
            print(f"  {name:<25s} {agree:>10d} {disagree:>10d} {rate:>7.1f}%")
    
    # Identify specific disagreement cases
    if full_pf is not None:
        print(f"\n  Disagreement Details (images where ablation differs from full pipeline):")
        ids = extract_image_ids(list(ablation_data.values())[0])
        for name, pf in all_pf.items():
            if 'full' in name.lower():
                continue
            disagreements = []
            for i, (a, b) in enumerate(zip(full_pf, pf)):
                if a != b:
                    full_score = ablation_data[[k for k in ablation_data if 'full' in k.lower()][0]]['results'][i]['overall_score']
                    abl_score = ablation_data[name]['results'][i]['overall_score']
                    disagreements.append((ids[i], a, b, full_score, abl_score))
            if disagreements:
                print(f"\n    {name} ({len(disagreements)} disagreements):")
                for img_id, fp, ap, fs, ascore in disagreements[:10]:  # show first 10
                    print(f"      {img_id}: Full={fp}({fs:.1f}) vs Ablation={ap}({ascore:.1f})")
                if len(disagreements) > 10:
                    print(f"      ... and {len(disagreements)-10} more")


def analysis_3_score_differences(ablation_data):
    """Paired score differences: how much each ablation deviates from full pipeline."""
    print_section("3. SCORE DIFFERENCES FROM FULL PIPELINE")
    
    full_key = [k for k in ablation_data if 'full' in k.lower()]
    if not full_key:
        print("  No full pipeline found. Skipping.")
        return
    
    full_scores = np.array(extract_overall_scores(ablation_data[full_key[0]]))
    
    print(f"\n  {'Variant':<25s} {'Mean Diff':>10s} {'Std Diff':>10s} {'Max Drop':>10s} {'Max Gain':>10s} {'RMSE':>10s}")
    print("  " + "-" * 78)
    
    for name, data in ablation_data.items():
        if 'full' in name.lower():
            continue
        abl_scores = np.array(extract_overall_scores(data))
        diff = abl_scores - full_scores  # positive = ablation scores higher
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        max_drop = np.min(diff)
        max_gain = np.max(diff)
        rmse = np.sqrt(np.mean(diff**2))
        print(f"  {name:<25s} {mean_diff:>+10.2f} {std_diff:>10.2f} {max_drop:>+10.2f} {max_gain:>+10.2f} {rmse:>10.2f}")


def analysis_4_statistical_tests(ablation_data):
    """Wilcoxon signed-rank tests between full pipeline and each ablation."""
    print_section("4. STATISTICAL SIGNIFICANCE TESTS")
    
    if not HAS_SCIPY:
        print("  scipy not available. Install scipy to run statistical tests.")
        return
    
    full_key = [k for k in ablation_data if 'full' in k.lower()]
    if not full_key:
        print("  No full pipeline found. Skipping.")
        return
    
    full_scores = np.array(extract_overall_scores(ablation_data[full_key[0]]))
    
    print(f"\n  Wilcoxon Signed-Rank Test (H0: no difference from full pipeline):")
    print(f"  {'Variant':<25s} {'Statistic':>12s} {'p-value':>12s} {'Significant':>12s}")
    print("  " + "-" * 64)
    
    for name, data in ablation_data.items():
        if 'full' in name.lower():
            continue
        abl_scores = np.array(extract_overall_scores(data))
        diff = abl_scores - full_scores
        # Only run if there are non-zero differences
        if np.all(diff == 0):
            print(f"  {name:<25s} {'N/A':>12s} {'N/A':>12s} {'identical':>12s}")
            continue
        try:
            stat, p = scipy_stats.wilcoxon(full_scores, abl_scores)
            sig = "YES (p<0.05)" if p < 0.05 else "No"
            print(f"  {name:<25s} {stat:>12.1f} {p:>12.6f} {sig:>12s}")
        except Exception as e:
            print(f"  {name:<25s} {'Error':>12s} {str(e)[:12]:>12s}")
    
    # Also Pearson correlation
    print(f"\n  Pearson Correlation with Full Pipeline:")
    print(f"  {'Variant':<25s} {'r':>10s} {'p-value':>12s}")
    print("  " + "-" * 50)
    for name, data in ablation_data.items():
        if 'full' in name.lower():
            continue
        abl_scores = np.array(extract_overall_scores(data))
        r, p = scipy_stats.pearsonr(full_scores, abl_scores)
        print(f"  {name:<25s} {r:>10.4f} {p:>12.6f}")


def analysis_5_subscores(ablation_data):
    """Break down sub-component scores across ablations."""
    print_section("5. SUB-COMPONENT SCORE BREAKDOWN")
    
    components = ['presence', 'spatial', 'relational', 'caption', 'llm', 'rules']
    
    print(f"\n  Mean Sub-Scores by Variant:")
    header = f"  {'Variant':<25s} " + " ".join(f"{c:>12s}" for c in components)
    print(header)
    print("  " + "-" * (25 + 13 * len(components)))
    
    for name, data in ablation_data.items():
        subs = extract_subscores(data)
        vals = []
        for c in components:
            arr = [x for x in subs[c] if x is not None]
            if arr:
                vals.append(f"{np.mean(arr):>12.2f}")
            else:
                vals.append(f"{'N/A':>12s}")
        print(f"  {name:<25s} " + " ".join(vals))


def analysis_6_discrimination_power(ablation_data):
    """
    Key analysis: Can each variant discriminate between good and bad images?
    Uses the full pipeline's PASS/FAIL as ground truth labels.
    """
    print_section("6. DISCRIMINATIVE POWER ANALYSIS")
    
    full_key = [k for k in ablation_data if 'full' in k.lower()]
    if not full_key:
        print("  No full pipeline found. Skipping.")
        return
    
    full_data = ablation_data[full_key[0]]
    full_pf = extract_pass_fail(full_data)
    pass_idx = [i for i, x in enumerate(full_pf) if x == 'PASS']
    fail_idx = [i for i, x in enumerate(full_pf) if x == 'FAIL']
    
    print(f"\n  Using Full Pipeline labels: {len(pass_idx)} PASS, {len(fail_idx)} FAIL")
    print(f"\n  {'Variant':<25s} {'PASS Mean':>10s} {'FAIL Mean':>10s} {'Gap':>8s} {'Effect Size':>12s}")
    print("  " + "-" * 68)
    
    for name, data in ablation_data.items():
        scores = extract_overall_scores(data)
        pass_scores = [scores[i] for i in pass_idx]
        fail_scores = [scores[i] for i in fail_idx]
        
        pass_mean = np.mean(pass_scores)
        fail_mean = np.mean(fail_scores)
        gap = pass_mean - fail_mean
        
        # Cohen's d
        pooled_std = np.sqrt((np.var(pass_scores, ddof=1) * (len(pass_scores)-1) + 
                              np.var(fail_scores, ddof=1) * (len(fail_scores)-1)) / 
                             (len(pass_scores) + len(fail_scores) - 2))
        cohens_d = gap / pooled_std if pooled_std > 0 else 0
        
        print(f"  {name:<25s} {pass_mean:>10.2f} {fail_mean:>10.2f} {gap:>+8.2f} {cohens_d:>12.2f}")


def analysis_7_detection_comparison(data_dir):
    """Compare YOLO-only vs VLM-only detection counts."""
    print_section("7. DETECTION-LEVEL COMPARISON (YOLO-only vs VLM-only)")
    
    yolo_file = os.path.join(data_dir, 'ensembled_result_aircraft_yolo_only.json')
    vlm_file = os.path.join(data_dir, 'ensembled_result_aircraft_vlm_only.json')
    
    if not os.path.exists(yolo_file) or not os.path.exists(vlm_file):
        print("  Detection ensemble files not found. Skipping.")
        return
    
    yolo_data = load_json(yolo_file)
    vlm_data = load_json(vlm_file)
    
    components = ['head', 'tail', 'swept_wing', 'engine', 'tail_wing']
    
    # Compare fused counts
    yolo_counts = defaultdict(list)
    vlm_counts = defaultdict(list)
    
    for r in yolo_data['results']:
        for c in components:
            yolo_counts[c].append(r['ensemble_output']['fused_counts'].get(c, 0))
    
    for r in vlm_data['results']:
        for c in components:
            vlm_counts[c].append(r['ensemble_output']['fused_counts'].get(c, 0))
    
    print(f"\n  Mean Fused Component Counts:")
    print(f"  {'Component':<15s} {'YOLO-only':>12s} {'VLM-only':>12s} {'Difference':>12s}")
    print("  " + "-" * 54)
    for c in components:
        ym = np.mean(yolo_counts[c])
        vm = np.mean(vlm_counts[c])
        print(f"  {c:<15s} {ym:>12.2f} {vm:>12.2f} {ym-vm:>+12.2f}")
    
    # Count disagreements per image
    disagree_count = 0
    total_comparisons = 0
    for i in range(len(yolo_data['results'])):
        for c in components:
            yc = yolo_data['results'][i]['ensemble_output']['fused_counts'].get(c, 0)
            vc = vlm_data['results'][i]['ensemble_output']['fused_counts'].get(c, 0)
            total_comparisons += 1
            if yc != vc:
                disagree_count += 1
    
    print(f"\n  Component-level disagreement rate: {disagree_count}/{total_comparisons} "
          f"({disagree_count/total_comparisons*100:.1f}%)")


def analysis_8_latex_table(stats_results):
    """Generate LaTeX-ready ablation table."""
    print_section("8. LATEX TABLE (copy-paste ready)")
    
    print(r"""
\begin{table}[h]
\centering
\caption{Ablation study on 70 synthetic aircraft images. Each row removes one component from the full PCMDE pipeline.}
\label{tab:ablation_aircraft}
\begin{tabular}{lcccccc}
\toprule
Variant & Mean & Std & Min & Max & Range & CV(\%) \\
\midrule""")
    
    for name, st in stats_results.items():
        clean_name = name.replace('_', ' ').title()
        print(f"{clean_name} & {st['Mean']} & {st['Std']} & {st['Min']} & {st['Max']} & {st['Range']} & {st['CV(%)']} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def analysis_9_per_image_csv(ablation_data, output_path):
    """Export per-image scores to CSV for further analysis."""
    print_section("9. PER-IMAGE CSV EXPORT")
    
    ids = extract_image_ids(list(ablation_data.values())[0])
    
    with open(output_path, 'w') as f:
        header = ['image_id'] + list(ablation_data.keys())
        f.write(','.join(header) + '\n')
        
        all_scores = {}
        for name, data in ablation_data.items():
            all_scores[name] = extract_overall_scores(data)
        
        for i, img_id in enumerate(ids):
            row = [img_id] + [str(all_scores[name][i]) for name in ablation_data.keys()]
            f.write(','.join(row) + '\n')
    
    print(f"  Saved to: {output_path}")


# ========== MAIN ==========

def main():
    parser = argparse.ArgumentParser(description='Aircraft Ablation Analysis for PCMDE')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing ablation JSON files')
    parser.add_argument('--output_csv', type=str, default='aircraft_ablation_per_image.csv',
                        help='Output CSV path for per-image scores')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    # Define expected files and their display names
    file_map = {
        'Full Pipeline': 'aircraft_ablation_full_pipeline.json',
        'Rules Only': 'aircraft_ablation_rules_only.json',
        'LLM Only': 'aircraft_ablation_llm_only.json',
        'VLM-only Detection': 'aircraft_ablation_vlm_only.json',
        # Add YOLO-only if you have the evaluation result (not just detection)
        'YOLO-only Detection': 'aircraft_ablation_yolo_only.json',
    }
    
    # Load available files
    ablation_data = {}
    for name, filename in file_map.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            ablation_data[name] = load_json(filepath)
            print(f"  Loaded: {name} ({filename})")
        else:
            print(f"  MISSING: {name} ({filename}) - skipping")
    
    if not ablation_data:
        print("ERROR: No ablation files found. Check --data_dir path.")
        sys.exit(1)
    
    print(f"\n  Loaded {len(ablation_data)} ablation variants.")
    
    # Run all analyses
    stats_results = analysis_1_summary_table(ablation_data)
    analysis_2_pass_fail(ablation_data)
    analysis_3_score_differences(ablation_data)
    analysis_4_statistical_tests(ablation_data)
    analysis_5_subscores(ablation_data)
    analysis_6_discrimination_power(ablation_data)
    analysis_7_detection_comparison(data_dir)
    analysis_8_latex_table(stats_results)
    analysis_9_per_image_csv(ablation_data, args.output_csv)
    
    print_section("ANALYSIS COMPLETE")
    print(f"  Total variants analyzed: {len(ablation_data)}")
    print(f"  Images per variant: {len(list(ablation_data.values())[0]['results'])}")
    print(f"  CSV export: {args.output_csv}")


if __name__ == '__main__':
    main()