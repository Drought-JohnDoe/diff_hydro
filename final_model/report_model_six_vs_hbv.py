#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE_ROOT = ROOT / 'code' / 'dPLHBVrelease' / 'hydroDL-dev'
sys.path.append(str(CODE_ROOT))
from hydroDL.data import camels

FOR_TYPE = 'daymet'
DEFAULT_HBV_EVA = ROOT / 'outputs' / 'rnnStreamflow' / 'CAMELSDemo' / 'dPLHBV' / 'ALL' / 'Testforc' / FOR_TYPE / 'BuffOpt0' / 'RMSE_para0.25' / '111111' / 'Train19801001_19951001Test19951001_20101001Buff5478Nmul16_HBVAll671_BS32_HS64_MaxIter100' / 'Eva10.npy'
BINS = [(-np.inf, 0.0, '<0'), (0.0, 0.2, '0-0.2'), (0.2, 0.4, '0.2-0.4'), (0.4, 0.5, '0.4-0.5'),
        (0.5, 0.65, '0.5-0.65'), (0.65, 0.8, '0.65-0.8'), (0.8, np.inf, '>=0.8')]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--result-suffix', type=str, required=True)
    p.add_argument('--analysis-dir', type=str, required=True)
    p.add_argument('--out-dir', type=str, required=True)
    p.add_argument('--subset-tag', type=str, default='AllBasins')
    p.add_argument('--seed', type=int, default=111111)
    p.add_argument('--forcing', type=str, default=FOR_TYPE)
    p.add_argument('--hbv-eva-path', type=str, default=str(DEFAULT_HBV_EVA))
    return p.parse_args()


def assign_bins(vals):
    labels = []
    for v in vals:
        lab = None
        for lo, hi, name in BINS:
            if v >= lo and v < hi:
                lab = name
                break
        labels.append(lab)
    return labels


def add_bar_labels(ax):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h):
            ax.annotate(f'{int(h)}' if abs(h - round(h)) < 1e-6 else f'{h:.2f}',
                        (p.get_x() + p.get_width() / 2, h),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')


def plot_binned_map(df, col, title, out_path):
    palette = {
        '<0': ('#6e1f2f', 26),
        '0-0.2': ('#c85a54', 32),
        '0.2-0.4': ('#e8a34f', 40),
        '0.4-0.5': ('#f4d35e', 50),
        '0.5-0.65': ('#8ec07c', 60),
        '0.65-0.8': ('#3fa7a3', 72),
        '>=0.8': ('#2b6cb0', 86),
    }
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.set_facecolor('#f7f4ef')
    for label in [b[2] for b in BINS]:
        sub = df[df[col] == label]
        if len(sub) == 0:
            continue
        color, size = palette[label]
        ax.scatter(sub['lon'], sub['lat'], s=size, c=color, edgecolors='black', linewidths=0.2, alpha=0.9,
                   label=f'{label} (n={len(sub)})')
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(alpha=0.18)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hbv_eva = np.load(Path(args.hbv_eva_path), allow_pickle=True)[0]

    model_base = ROOT / 'outputs' / 'rnnStreamflow' / 'CAMELSMODELSIX' / 'DynamicSimHydModelSix' / args.subset_tag / args.forcing / str(args.seed)
    model_result = model_base / args.result_suffix
    model_analysis = Path(args.analysis_dir)

    camels.initcamels(str(ROOT / 'Camels'))
    g = camels.gageDict
    meta_all = pd.DataFrame({
        'gage_id': g['id'].astype(int),
        'gage_name': g['name'],
        'lat': g['lat'].astype(float),
        'lon': g['lon'].astype(float),
        'huc': g['huc'].astype(int),
    })
    hbv_nse_map = dict(zip(meta_all['gage_id'].tolist(), hbv_eva['NSE']))

    model_eva = np.load(model_result / f'Eva{args.epoch}.npy', allow_pickle=True)[0]
    model_analysis_df = pd.read_csv(model_analysis / 'per_basin_model_six_metrics.csv')

    out = model_analysis_df[['gage_id']].copy()
    out = out.merge(meta_all, on='gage_id', how='left')
    out['NSE_model_six'] = model_eva['NSE']
    out['NSE_hbv'] = out['gage_id'].map(hbv_nse_map).astype(float)
    out['NSE_diff_model_six_minus_hbv'] = out['NSE_model_six'] - out['NSE_hbv']
    out['nse_bin_model_six'] = assign_bins(out['NSE_model_six'].to_numpy(dtype=float))
    out['nse_bin_hbv'] = assign_bins(out['NSE_hbv'].to_numpy(dtype=float))
    cols = ['gage_id', 'KGE', 'logNSE', 'lowflow_NSE', 'highflow_NSE', 'fdc_error', 'BFI_error',
            'ET_ratio', 'runoff_ratio', 'water_balance_closure', 'BFI', 'BFI_obs']
    out = out.merge(model_analysis_df[cols], on='gage_id', how='left')
    out.to_csv(out_dir / 'per_basin_model_six_vs_hbv.csv', index=False)

    summary = pd.DataFrame([{
        'median_nse_model_six': float(np.nanmedian(out['NSE_model_six'])),
        'median_nse_hbv': float(np.nanmedian(out['NSE_hbv'])),
        'mean_nse_model_six': float(np.nanmean(out['NSE_model_six'])),
        'mean_nse_hbv': float(np.nanmean(out['NSE_hbv'])),
        'median_kge_model_six': float(np.nanmedian(out['KGE'])),
        'median_lognse_model_six': float(np.nanmedian(out['logNSE'])),
        'median_lowflow_nse_model_six': float(np.nanmedian(out['lowflow_NSE'])),
        'median_highflow_nse_model_six': float(np.nanmedian(out['highflow_NSE'])),
        'median_fdc_error_model_six': float(np.nanmedian(out['fdc_error'])),
        'median_bfi_error_model_six': float(np.nanmedian(out['BFI_error'])),
        'median_runoff_ratio_model_six': float(np.nanmedian(out['runoff_ratio'])),
        'median_et_ratio_model_six': float(np.nanmedian(out['ET_ratio'])),
        'median_water_balance_closure_model_six': float(np.nanmedian(out['water_balance_closure'])),
        'model_six_nse_wins': int(np.sum(out['NSE_model_six'] > out['NSE_hbv'])),
        'hbv_nse_wins': int(np.sum(out['NSE_hbv'] > out['NSE_model_six'])),
    }])
    summary.to_csv(out_dir / 'summary.csv', index=False)

    rows = []
    for label in [b[2] for b in BINS]:
        rows.append({
            'bin': label,
            'hbv_count': int(np.sum(out['nse_bin_hbv'] == label)),
            'model_six_count': int(np.sum(out['nse_bin_model_six'] == label)),
        })
    bins_df = pd.DataFrame(rows)
    bins_df.to_csv(out_dir / 'test_nse_bins_hbv_vs_model_six.csv', index=False)

    x = np.arange(len(bins_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.bar(x - width / 2, bins_df['hbv_count'], width, label='HBV Ep10', color='#4c78a8')
    ax.bar(x + width / 2, bins_df['model_six_count'], width, label=f'Model Six Ep{args.epoch}', color='#f58518')
    ax.set_xticks(x)
    ax.set_xticklabels(bins_df['bin'])
    ax.set_ylabel('Number of basins')
    ax.set_title(f'NSE bin counts: HBV Ep10 vs Model Six Ep{args.epoch}')
    ax.legend()
    add_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(out_dir / 'test_nse_bins_side_by_side.png', dpi=240, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for vals, label, color in [
        (np.sort(out['NSE_hbv'].dropna().to_numpy()), 'HBV Ep10', '#4c78a8'),
        (np.sort(out['NSE_model_six'].dropna().to_numpy()), f'Model Six Ep{args.epoch}', '#f58518'),
    ]:
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=label, color=color, linewidth=2)
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel('Basin NSE')
    ax.set_ylabel('CDF')
    ax.set_title('CDF of basin NSE')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'cdf_basin_nse_hbv_vs_model_six.png', dpi=240, bbox_inches='tight')
    plt.close(fig)

    plot_binned_map(out, 'nse_bin_hbv', 'HBV Ep10 basin NSE bins', out_dir / 'spatial_hbv_nse_conus_binned.png')
    plot_binned_map(out, 'nse_bin_model_six', f'Model Six Ep{args.epoch} basin NSE bins',
                    out_dir / 'spatial_model_six_nse_conus_binned.png')

    diff_bins = []
    for v in out['NSE_diff_model_six_minus_hbv'].to_numpy(dtype=float):
        if v < -0.2:
            diff_bins.append('<-0.2')
        elif v < -0.05:
            diff_bins.append('-0.2 to -0.05')
        elif v < 0.05:
            diff_bins.append('-0.05 to 0.05')
        elif v < 0.2:
            diff_bins.append('0.05 to 0.2')
        else:
            diff_bins.append('>0.2')
    out['nse_diff_bin'] = diff_bins
    palette = {
        '<-0.2': ('#7f1d1d', 60),
        '-0.2 to -0.05': ('#dc2626', 55),
        '-0.05 to 0.05': ('#d4d4d8', 44),
        '0.05 to 0.2': ('#16a34a', 55),
        '>0.2': ('#14532d', 60),
    }
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.set_facecolor('#f7f4ef')
    for label in ['<-0.2', '-0.2 to -0.05', '-0.05 to 0.05', '0.05 to 0.2', '>0.2']:
        sub = out[out['nse_diff_bin'] == label]
        if len(sub) == 0:
            continue
        color, size = palette[label]
        ax.scatter(sub['lon'], sub['lat'], s=size, c=color, edgecolors='black', linewidths=0.2, alpha=0.9,
                   label=f'{label} (n={len(sub)})')
    ax.set_title(f'Model Six Ep{args.epoch} minus HBV Ep10 NSE')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(alpha=0.18)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'spatial_nse_difference_model_six_minus_hbv_binned.png', dpi=240, bbox_inches='tight')
    plt.close(fig)

    report_lines = [
        f'Model Six Ep{args.epoch} median NSE: {np.nanmedian(out["NSE_model_six"]):.4f}',
        f'HBV Ep10 median NSE: {np.nanmedian(out["NSE_hbv"]):.4f}',
        f'Model Six wins on NSE in {int(np.sum(out["NSE_model_six"] > out["NSE_hbv"]))} / {len(out)} basins',
        f'Median Model Six KGE: {np.nanmedian(out["KGE"]):.4f}',
        f'Median Model Six logNSE: {np.nanmedian(out["logNSE"]):.4f}',
        f'Median Model Six lowflow NSE: {np.nanmedian(out["lowflow_NSE"]):.4f}',
        f'Median Model Six highflow NSE: {np.nanmedian(out["highflow_NSE"]):.4f}',
        f'Median Model Six FDC error: {np.nanmedian(out["fdc_error"]):.4f}',
        f'Median Model Six BFI error: {np.nanmedian(out["BFI_error"]):.4f}',
        f'Median Model Six ET/P: {np.nanmedian(out["ET_ratio"]):.4f}',
        f'Median Model Six Q/P: {np.nanmedian(out["runoff_ratio"]):.4f}',
    ]
    (out_dir / 'report.txt').write_text('\n'.join(report_lines))


if __name__ == '__main__':
    main()
