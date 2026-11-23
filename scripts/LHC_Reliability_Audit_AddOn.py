"""Read-only reliability audit for an ADC2021 strict-baseline results ZIP or directory.

This does not train models, access HDF5, modify original CSVs or claim novelty.
Works in Kaggle (with attached original results directory) and locally with a results ZIP.
"""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import json
import os
import pandas as pd
from scipy.stats import binomtest


def audit(source=None, output_dir=None):
    if source is None:
        candidates = [
            Path('/kaggle/working/lhc_adc2021_strict_baseline'),
            Path('/kaggle/working/LHC_Research_Results.zip'),
            Path('/mnt/data/LHC_Research_Results.zip'),
        ]
        source = next((p for p in candidates if p.exists()), None)
        if source is None:
            raise FileNotFoundError('No results found. Pass source= to the results ZIP or directory.')
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f'Results not found: {source}')

    with ZipFile(source) if source.is_file() else _NullZip() as z:
        def read_bytes(filename):
            if z is None:
                p = source / filename
                if not p.is_file():
                    raise FileNotFoundError(f'Missing required result: {p}')
                return p.read_bytes()
            names = z.namelist()
            matches = [n for n in names if n == filename or n.endswith('/' + filename)]
            if len(matches) != 1:
                raise ValueError(f'Expected one {filename} in ZIP, found {len(matches)}')
            return z.read(matches[0])

        metrics = pd.read_csv(BytesIO(read_bytes('signal_metrics.csv')))
        cal = pd.read_csv(BytesIO(read_bytes('calibration_thresholds.csv')))
        metadata = json.loads(read_bytes('run_metadata.json'))
        isol = pd.read_csv(BytesIO(read_bytes('isolation_forest_metrics.csv')))
        required = {'signal','n_signal','roc_auc','average_precision','positive_prevalence_in_pair'}
        if not required.issubset(metrics.columns):
            raise ValueError(f'Missing signal metric columns: {sorted(required-set(metrics.columns))}')
        cr = []
        for _, r in cal.iterrows():
            target = float(r['target_fpr'])
            n = int(r['test_background_n'])
            k = int(r['test_false_positives'])
            if n <= 0 or not 0 <= k <= n or not 0 < target < 1:
                raise ValueError('Invalid calibration/test FPR data')
            ci = binomtest(k,n).proportion_ci(0.95,method='exact')
            # One-sided test against FPR target. Condition on the independently selected threshold.
            p_greater = float(binomtest(k,n,target,alternative='greater').pvalue)
            status = ('above target (95% CI)' if ci.low > target else
                      'below target (95% CI)' if ci.high < target else 'inconclusive')
            cr.append({
                'nominal_target_fpr': target,
                'calibration_tail_events': int(r['calibration_tail_events']),
                'n_cal': int(r['calibration_background_n']),
                'test_false_positives': k,
                'n_test': n,
                'observed_fpr': k / n,
                'test_fpr_ci95_lower': float(ci.low),
                'test_fpr_ci95_upper': float(ci.high),
                'test_fpr_relative_to_target': (k / n) / target,
                'binomial_one_sided_p_for_excess': p_greater,
                'status': status,
            })
        fpr_audit = pd.DataFrame(cr).sort_values('nominal_target_fpr',ascending=False)

        if 'signal' not in isol.columns or 'roc_auc' not in isol.columns:
            raise ValueError('Isolation Forest results missing signal/roc_auc')
        comps = metrics[['signal','n_signal','roc_auc','average_precision','positive_prevalence_in_pair']].merge(
            isol[['signal','roc_auc','average_precision']].rename(columns={
                'roc_auc':'iforest_roc_auc','average_precision':'iforest_average_precision'
            }), on='signal', how='left',validate='one_to_one'
        )
        comps['iforest_minus_ae_auc'] = comps['iforest_roc_auc'] - comps['roc_auc']
        comps['ae_ap_lift_over_random'] = comps['average_precision']/comps['positive_prevalence_in_pair']

        # Mark achieved FPR next to reported TPR; distinguish nominal from achieved.
        long_rows=[]
        for _, r in metrics.iterrows():
            for target in sorted(fpr_audit['nominal_target_fpr'],reverse=True):
                key=f'{target:.0e}'
                col=f'calibrated_signal_tpr_{key}'
                if col in metrics.columns:
                    row=fpr_audit.loc[fpr_audit.nominal_target_fpr==target].iloc[0]
                    long_rows.append({
                        'signal':r['signal'],
                        'nominal_target_fpr':target,
                        'achieved_test_fpr':row['observed_fpr'],
                        'signal_tpr_at_nominally_calibrated_threshold':r[col],
                        'estimated_signal_pass_count_rounded':round(float(r[col])*int(r['n_signal'])),
                        'signal_n':int(r['n_signal']),
                    })
        tpr = pd.DataFrame(long_rows)

    if output_dir is None:
        output_dir = source/'reliability_audit' if source.is_dir() else source.parent/'reliability_audit'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    fpr_audit.to_csv(output_dir/'fpr_reliability.csv',index=False)
    comps.to_csv(output_dir/'model_comparison.csv',index=False)
    tpr.to_csv(output_dir/'signal_efficiencies_with_achieved_fpr.csv',index=False)

    verified = bool(metadata.get('official_background_v2_verified'))
    split_counts = metadata.get('background_split_counts',{})
    input_dim = 57
    bottleneck = 64
    lines = [
        '# ADC2021 strict baseline — independent reliability audit',
        '',
        f'- Source: `{source}`',
        f'- Background checksum verified in notebook metadata: **{verified}** (file not rehashed by this audit).',
        f'- Dataset overlap flag in notebook metadata: background_equals_blackbox = `{metadata.get("background_equals_blackbox")}`.',
        f'- Background counts: `{split_counts}`. Split indices not included in uploaded ZIP.',
        f'- Architecture bottleneck width = **{bottleneck}**, input width = **{input_dim}**. This is not dimensional compression.',
        '- Isolation Forest is compared here only by ROC-AUC and AP, because it lacks independently calibrated low-FPR results.',
        '- Low-FPR uncertainty below assumes approximately independent test background events, with fixed thresholds learned from independent calibration data.',
        '',
        '## Independent-background FPR checks',
    ]
    for _,r in fpr_audit.iterrows():
        lines.append(f'- Nominal {r.nominal_target_fpr:.0e}: observed **{r.observed_fpr:.7g}** '
                     f'({r.test_false_positives}/{r.n_test}); exact 95% CI '
                     f'[{r.test_fpr_ci95_lower:.7g}, {r.test_fpr_ci95_upper:.7g}]; '
                     f'**{r.status}**; one-sided excess p={r.binomial_one_sided_p_for_excess:.4g}.')
    lines.extend(['','## Model comparison, exploratory',''])
    for _,r in comps.iterrows():
        lines.append(f'- {r.signal}: AE ROC-AUC {r.roc_auc:.4f}; IF ROC-AUC {r.iforest_roc_auc:.4f}; '
                     f'IF−AE = {r.iforest_minus_ae_auc:+.4f}.')
    lines.extend(['','## Scientific interpretation and next checks',
                  '',
                  '1. Do not label calibrated TPR as achieved TPR at a target FPR if the realized independent-test FPR is higher.',
                  '2. Repeat full training with at least 3 independent training seeds; retain exact split assignment, input checksums, environment and loss history.',
                  '3. Calibrate and test Isolation Forest at the same 3 FPR targets using disjoint background sets before claiming superiority.',
                  '4. Examine extreme false positives for per-object reconstruction contributions, zero padding, and extreme pT; this is diagnostic, not established novel research.',
                  '5. If testing reduced bottleneck size (e.g., 16 or 32), treat it as an ablation and reserve the signal/test data for final evaluation.',
                  '6. No raw per-event score vectors or original dataset files are included, so resampling, duplicate-event detection and signal-file checksums cannot be independently verified here.',
                  '7. Blackbox candidate ranks are unlabelled: they cannot establish detection sensitivity or discovery.',
                  '',
                  'This audit is not itself a novel algorithm or a claim of publishable improvement.',
                  ''])
    (output_dir/'AUDIT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    summary={'source':str(source),'verified_background_metadata':verified,
             'background_split_counts':split_counts,
             'n_signal_metrics':len(metrics),
             'model_auc_comparison_n':len(comps),
             'fpr_audit':fpr_audit.to_dict(orient='records')}
    (output_dir/'AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Read-only audit complete. No HDF5 or models loaded and no input files changed.')
    print(fpr_audit[['nominal_target_fpr','test_false_positives','observed_fpr','test_fpr_ci95_lower','test_fpr_ci95_upper','status']].to_string(index=False))
    print('\nIsolation Forest minus AE ROC-AUC:')
    print(comps[['signal','roc_auc','iforest_roc_auc','iforest_minus_ae_auc']].to_string(index=False))
    print('\nSaved audit to:',output_dir)
    return output_dir


class _NullZip:
    def __enter__(self): return None
    def __exit__(self,exc_type,exc,tb):return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=None)
    parser.add_argument('--output-dir',default=None)
    a=parser.parse_args()
    audit(a.source,a.output_dir)
