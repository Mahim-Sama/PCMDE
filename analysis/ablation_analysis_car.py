#!/usr/bin/env python3
"""
Car Dataset Ablation Analysis for PCMDE Framework - ICML Rebuttal
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ========== DATA LOADING ==========

DATA_DIR = r'Q:\SOL\SyntheticDataGen\Ablation\car'

FILES = {
    'Full Pipeline':      os.path.join(DATA_DIR, 'car_ablation_full_pipeline.json'),
    'Rules Only':         os.path.join(DATA_DIR, 'car_ablation_rules_only.json'),
    'LLM Only':           os.path.join(DATA_DIR, 'car_ablation_llm_only.json'),
    'VLM-only Detection': os.path.join(DATA_DIR, 'car_ablation_vlm_only.json'),
    'YOLO-only Detection':os.path.join(DATA_DIR, 'car_ablation_yolo_only.json'),
}

ENSEMBLE_FILES = {
    'YOLO-only': os.path.join(DATA_DIR, 'ensembled_result_car_yolo_only.json'),
    'VLM-only':  os.path.join(DATA_DIR, 'ensembled_result_car_vlm_only.json'),
}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_all():
    ablation = {}
    for name, path in FILES.items():
        if os.path.exists(path):
            ablation[name] = load_json(path)
            print(f"  Loaded: {name}")
        else:
            print(f"  MISSING: {name}")
    return ablation

# ========== EXTRACTION HELPERS ==========

def get_scores(data):
    return [r['final_score'] for r in data['results']]

def get_verdicts(data):
    return [r['verdict'] for r in data['results']]

def get_image_ids(data):
    return [r['image_id'] for r in data['results']]

def get_rules_scores(data):
    return [r['score_components']['rules_score'] for r in data['results']]

def get_llm_scores(data):
    return [r['score_components']['llm_score'] for r in data['results']]

def get_sub_breakdown(data):
    """Extract presence, spatial, relational from rules_validation.breakdown."""
    out = {'presence': [], 'spatial': [], 'relational': []}
    for r in data['results']:
        bd = r['rules_validation']['breakdown']
        out['presence'].append(bd.get('presence_and_count', None))
        out['spatial'].append(bd.get('spatial_rules', None))
        out['relational'].append(bd.get('relational_constraints', None))
    return out

def compute_stats(scores):
    a = np.array(scores, dtype=float)
    mean = np.mean(a)
    std = np.std(a, ddof=1)
    mn, mx = np.min(a), np.max(a)
    return {
        'N': len(a),
        'Mean': round(mean, 2),
        'Std': round(std, 2),
        'Min': round(mn, 2),
        'Max': round(mx, 2),
        'Range': round(mx - mn, 2),
        'CV(%)': round(std / mean * 100, 2) if mean else 0,
        'Median': round(np.median(a), 2),
    }

def sep(c='=', n=90):
    print(c * n)

def section(title):
    print(); sep(); print(f"  {title}"); sep()

# ========== ANALYSES ==========

def analysis_1_summary(ablation):
    section("1. ABLATION SUMMARY STATISTICS — Car Dataset (N=70)")
    cols = ['N', 'Mean', 'Std', 'Min', 'Max', 'Range', 'CV(%)', 'Median']
    header = f"  {'Variant':<25s}" + "".join(f"{c:>10s}" for c in cols)
    print(header)
    print("  " + "-" * (25 + 10 * len(cols)))
    results = {}
    for name, data in ablation.items():
        st = compute_stats(get_scores(data))
        results[name] = st
        row = f"  {name:<25s}" + "".join(f"{st[c]:>10}" for c in cols)
        print(row)
    # Also show rules_score and llm_score distributions
    print("\n  Rules-Score Component Distribution:")
    print(f"  {'Variant':<25s}{'Mean':>10s}{'Std':>10s}{'Min':>10s}{'Max':>10s}")
    print("  " + "-" * 65)
    for name, data in ablation.items():
        rs = np.array(get_rules_scores(data))
        print(f"  {name:<25s}{np.mean(rs):>10.2f}{np.std(rs,ddof=1):>10.2f}{np.min(rs):>10.2f}{np.max(rs):>10.2f}")
    print("\n  LLM-Score Component Distribution:")
    print(f"  {'Variant':<25s}{'Mean':>10s}{'Std':>10s}{'Min':>10s}{'Max':>10s}")
    print("  " + "-" * 65)
    for name, data in ablation.items():
        ls = np.array(get_llm_scores(data))
        print(f"  {name:<25s}{np.mean(ls):>10.2f}{np.std(ls,ddof=1):>10.2f}{np.min(ls):>10.2f}{np.max(ls):>10.2f}")
    return results

def analysis_2_passfail(ablation):
    section("2. PASS/FAIL ANALYSIS")
    full_v = get_verdicts(ablation['Full Pipeline'])
    
    print(f"  {'Variant':<25s}{'Passed':>8s}{'Failed':>8s}{'Rate':>8s}")
    print("  " + "-" * 52)
    for name, data in ablation.items():
        v = get_verdicts(data)
        p = sum(1 for x in v if x == 'PASS')
        print(f"  {name:<25s}{p:>8d}{len(v)-p:>8d}{p/len(v)*100:>7.1f}%")
    
    print(f"\n  Agreement with Full Pipeline:")
    print(f"  {'Variant':<25s}{'Agree':>8s}{'Disagree':>10s}{'Rate':>8s}")
    print("  " + "-" * 54)
    ids = get_image_ids(ablation['Full Pipeline'])
    for name, data in ablation.items():
        if name == 'Full Pipeline':
            continue
        v = get_verdicts(data)
        agree = sum(1 for a, b in zip(full_v, v) if a == b)
        print(f"  {name:<25s}{agree:>8d}{len(v)-agree:>10d}{agree/len(v)*100:>7.1f}%")
    
    # Show disagreement details
    print(f"\n  Disagreement Details:")
    for name, data in ablation.items():
        if name == 'Full Pipeline':
            continue
        v = get_verdicts(data)
        scores_full = get_scores(ablation['Full Pipeline'])
        scores_abl = get_scores(data)
        disagrees = []
        for i in range(len(v)):
            if full_v[i] != v[i]:
                disagrees.append((ids[i], full_v[i], v[i], scores_full[i], scores_abl[i]))
        if disagrees:
            print(f"\n    {name} ({len(disagrees)} disagreements):")
            for img, fv, av, fs, asc in disagrees[:8]:
                print(f"      {img}: Full={fv}({fs:.1f}) -> Ablation={av}({asc:.1f})")
            if len(disagrees) > 8:
                print(f"      ... and {len(disagrees)-8} more")

def analysis_3_diffs(ablation):
    section("3. SCORE DIFFERENCES FROM FULL PIPELINE")
    full_s = np.array(get_scores(ablation['Full Pipeline']))
    
    print(f"  {'Variant':<25s}{'MeanDiff':>10s}{'StdDiff':>10s}{'MaxDrop':>10s}{'MaxGain':>10s}{'RMSE':>10s}")
    print("  " + "-" * 78)
    for name, data in ablation.items():
        if name == 'Full Pipeline':
            continue
        abl_s = np.array(get_scores(data))
        diff = abl_s - full_s
        md = np.mean(diff)
        sd = np.std(diff, ddof=1)
        rmse = np.sqrt(np.mean(diff**2))
        print(f"  {name:<25s}{md:>+10.2f}{sd:>10.2f}{np.min(diff):>+10.2f}{np.max(diff):>+10.2f}{rmse:>10.2f}")

def analysis_4_stats_tests(ablation):
    section("4. STATISTICAL SIGNIFICANCE TESTS")
    if not HAS_SCIPY:
        print("  scipy not installed — skipping"); return
    
    full_s = np.array(get_scores(ablation['Full Pipeline']))
    
    print(f"\n  Wilcoxon Signed-Rank (vs Full Pipeline):")
    print(f"  {'Variant':<25s}{'Statistic':>12s}{'p-value':>12s}{'Significant':>14s}")
    print("  " + "-" * 66)
    for name, data in ablation.items():
        if name == 'Full Pipeline':
            continue
        abl_s = np.array(get_scores(data))
        diff = abl_s - full_s
        if np.all(diff == 0):
            print(f"  {name:<25s}{'N/A':>12s}{'N/A':>12s}{'identical':>14s}")
            continue
        try:
            stat, p = scipy_stats.wilcoxon(full_s, abl_s)
            sig = "YES (p<0.05)" if p < 0.05 else "No"
            print(f"  {name:<25s}{stat:>12.1f}{p:>12.6f}{sig:>14s}")
        except Exception as e:
            print(f"  {name:<25s}  Error: {e}")
    
    print(f"\n  Pearson Correlation with Full Pipeline:")
    print(f"  {'Variant':<25s}{'r':>10s}{'p-value':>12s}")
    print("  " + "-" * 50)
    for name, data in ablation.items():
        if name == 'Full Pipeline':
            continue
        abl_s = np.array(get_scores(data))
        r, p = scipy_stats.pearsonr(full_s, abl_s)
        print(f"  {name:<25s}{r:>10.4f}{p:>12.6f}")

def analysis_5_subscores(ablation):
    section("5. SUB-COMPONENT BREAKDOWN (Rules Validation)")
    components = ['presence', 'spatial', 'relational']
    
    print(f"  {'Variant':<25s}" + "".join(f"{c:>14s}" for c in ['presence_mean', 'spatial_mean', 'relational_mean']))
    print("  " + "-" * 67)
    for name, data in ablation.items():
        bd = get_sub_breakdown(data)
        vals = []
        for c in components:
            arr = [x for x in bd[c] if x is not None]
            vals.append(f"{np.mean(arr):>14.2f}" if arr else f"{'N/A':>14s}")
        print(f"  {name:<25s}" + "".join(vals))

def analysis_6_discrimination(ablation):
    section("6. DISCRIMINATIVE POWER (using Full Pipeline labels as ground truth)")
    full_v = get_verdicts(ablation['Full Pipeline'])
    pass_idx = [i for i, x in enumerate(full_v) if x == 'PASS']
    fail_idx = [i for i, x in enumerate(full_v) if x == 'FAIL']
    
    print(f"  Reference: {len(pass_idx)} PASS, {len(fail_idx)} FAIL\n")
    print(f"  {'Variant':<25s}{'PASS Mean':>10s}{'FAIL Mean':>10s}{'Gap':>8s}{'Cohen d':>10s}")
    print("  " + "-" * 66)
    
    for name, data in ablation.items():
        scores = get_scores(data)
        ps = [scores[i] for i in pass_idx]
        fs = [scores[i] for i in fail_idx]
        pm, fm = np.mean(ps), np.mean(fs)
        gap = pm - fm
        pooled = np.sqrt((np.var(ps, ddof=1)*(len(ps)-1) + np.var(fs, ddof=1)*(len(fs)-1)) / (len(ps)+len(fs)-2))
        d = gap / pooled if pooled > 0 else 0
        print(f"  {name:<25s}{pm:>10.2f}{fm:>10.2f}{gap:>+8.2f}{d:>10.2f}")

def analysis_7_detection(ablation):
    section("7. DETECTION-LEVEL: YOLO-only vs VLM-only Component Counts")
    
    yolo_ens = load_json(ENSEMBLE_FILES['YOLO-only'])
    vlm_ens = load_json(ENSEMBLE_FILES['VLM-only'])
    
    # Gather all component names across both
    all_comps = set()
    for r in yolo_ens['results']:
        all_comps.update(r['ensemble_output']['fused_counts'].keys())
    for r in vlm_ens['results']:
        all_comps.update(r['ensemble_output']['fused_counts'].keys())
    all_comps = sorted(all_comps)
    
    yolo_counts = defaultdict(list)
    vlm_counts = defaultdict(list)
    for r in yolo_ens['results']:
        for c in all_comps:
            yolo_counts[c].append(r['ensemble_output']['fused_counts'].get(c, 0))
    for r in vlm_ens['results']:
        for c in all_comps:
            vlm_counts[c].append(r['ensemble_output']['fused_counts'].get(c, 0))
    
    print(f"\n  Mean Fused Component Counts (N=70 images):")
    print(f"  {'Component':<20s}{'YOLO-only':>12s}{'VLM-only':>12s}{'Diff':>10s}")
    print("  " + "-" * 57)
    for c in all_comps:
        ym = np.mean(yolo_counts[c])
        vm = np.mean(vlm_counts[c])
        print(f"  {c:<20s}{ym:>12.2f}{vm:>12.2f}{ym-vm:>+10.2f}")
    
    # Disagreement rate
    disagree = 0
    total = 0
    for i in range(len(yolo_ens['results'])):
        for c in all_comps:
            yc = yolo_ens['results'][i]['ensemble_output']['fused_counts'].get(c, 0)
            vc = vlm_ens['results'][i]['ensemble_output']['fused_counts'].get(c, 0)
            total += 1
            if yc != vc:
                disagree += 1
    print(f"\n  Component-level disagreement: {disagree}/{total} ({disagree/total*100:.1f}%)")
    
    # Which components disagree most?
    print(f"\n  Per-Component Disagreement Rate:")
    print(f"  {'Component':<20s}{'Disagree':>10s}{'Rate':>10s}")
    print("  " + "-" * 43)
    for c in all_comps:
        d = sum(1 for i in range(len(yolo_ens['results'])) 
                if yolo_ens['results'][i]['ensemble_output']['fused_counts'].get(c, 0) != 
                   vlm_ens['results'][i]['ensemble_output']['fused_counts'].get(c, 0))
        print(f"  {c:<20s}{d:>10d}{d/70*100:>9.1f}%")

def analysis_8_latex(stats_results):
    section("8. LATEX TABLE (copy-paste ready)")
    print(r"""
\begin{table}[h]
\centering
\caption{Ablation study on 70 synthetic car images. Each row removes one component from the full PCMDE pipeline.}
\label{tab:ablation_car}
\begin{tabular}{lcccccc}
\toprule
Variant & Mean & Std & Min & Max & Range & CV(\%) \\
\midrule""")
    for name, st in stats_results.items():
        cn = name.replace('_', ' ')
        print(f"{cn} & {st['Mean']} & {st['Std']} & {st['Min']} & {st['Max']} & {st['Range']} & {st['CV(%)']} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

def analysis_9_csv(ablation):
    section("9. PER-IMAGE CSV EXPORT")
    out = r'Q:\SOL\SyntheticDataGen\Ablation\car_ablation_per_image.csv'
    ids = get_image_ids(ablation['Full Pipeline'])
    names = list(ablation.keys())
    all_scores = {n: get_scores(d) for n, d in ablation.items()}
    all_rules = {n: get_rules_scores(d) for n, d in ablation.items()}
    all_llm = {n: get_llm_scores(d) for n, d in ablation.items()}
    all_verdicts = {n: get_verdicts(d) for n, d in ablation.items()}
    
    with open(out, 'w') as f:
        h = ['image_id']
        for n in names:
            sn = n.replace(' ', '_')
            h += [f'{sn}_final', f'{sn}_rules', f'{sn}_llm', f'{sn}_verdict']
        f.write(','.join(h) + '\n')
        for i, img in enumerate(ids):
            row = [img]
            for n in names:
                row += [str(all_scores[n][i]), str(all_rules[n][i]), str(all_llm[n][i]), all_verdicts[n][i]]
            f.write(','.join(row) + '\n')
    print(f"  Saved to: {out}")

def analysis_10_key_insights(ablation, stats_results):
    section("10. KEY INSIGHTS FOR REBUTTAL")
    
    full_scores = np.array(get_scores(ablation['Full Pipeline']))
    full_v = get_verdicts(ablation['Full Pipeline'])
    
    print("\n  A) Which component contributes most to discriminative power?")
    pass_idx = [i for i, x in enumerate(full_v) if x == 'PASS']
    fail_idx = [i for i, x in enumerate(full_v) if x == 'FAIL']
    
    for name, data in ablation.items():
        scores = get_scores(data)
        ps = np.mean([scores[i] for i in pass_idx])
        fs = np.mean([scores[i] for i in fail_idx])
        cv = stats_results[name]['CV(%)']
        print(f"    {name:<25s}: Gap={ps-fs:>+7.2f}, CV={cv}%")
    
    print("\n  B) Does removing LLM or Rules hurt more?")
    rules_only_cv = stats_results['Rules Only']['CV(%)']
    llm_only_cv = stats_results['LLM Only']['CV(%)']
    full_cv = stats_results['Full Pipeline']['CV(%)']
    print(f"    Full Pipeline CV: {full_cv}%")
    print(f"    Rules Only CV:   {rules_only_cv}%")
    print(f"    LLM Only CV:     {llm_only_cv}%")
    
    print("\n  C) Detection ablation impact:")
    for det_name in ['VLM-only Detection', 'YOLO-only Detection']:
        if det_name in ablation:
            abl_s = np.array(get_scores(ablation[det_name]))
            diff = abl_s - full_scores
            print(f"    {det_name}: Mean shift={np.mean(diff):+.2f}, RMSE={np.sqrt(np.mean(diff**2)):.2f}")
    
    print("\n  D) Score correlation structure:")
    if HAS_SCIPY:
        for name in ['Rules Only', 'LLM Only']:
            if name in ablation:
                r, _ = scipy_stats.pearsonr(full_scores, np.array(get_scores(ablation[name])))
                print(f"    Full vs {name}: r={r:.4f}")

# ========== MAIN ==========

def main():
    print("=" * 90)
    print("  PCMDE CAR DATASET ABLATION ANALYSIS — ICML 2026 Rebuttal")
    print("=" * 90)
    
    ablation = load_all()
    
    stats = analysis_1_summary(ablation)
    analysis_2_passfail(ablation)
    analysis_3_diffs(ablation)
    analysis_4_stats_tests(ablation)
    analysis_5_subscores(ablation)
    analysis_6_discrimination(ablation)
    analysis_7_detection(ablation)
    analysis_8_latex(stats)
    analysis_9_csv(ablation)
    analysis_10_key_insights(ablation, stats)
    
    section("ANALYSIS COMPLETE")

if __name__ == '__main__':
    main()
