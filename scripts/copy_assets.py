#!/usr/bin/env python3
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'paper_project'
SRC = ROOT / 'results_paperlock' / 'figures'
DST = PROJECT / 'figs'

MAP = {
    'fig_system_model_p3clr': 'system_model_p3clr',
    'fig_algorithm_flow_p3clr': 'algorithm_flow_p3clr',
    'fig_delay_cdf_paperlock': 'delay_cdf_paperlock',
    'fig_energy_by_regime_paperlock': 'energy_by_regime_paperlock',
    'fig_handover_by_regime_paperlock': 'handover_by_regime_paperlock',
    'fig_cache_hit_breakdown_paperlock': 'cache_hit_breakdown_paperlock',
    'fig_outage_by_weather_paperlock': 'outage_by_weather_paperlock',
    'fig_predictor_scatter_paperlock': 'predictor_scatter_paperlock',
    'fig_predictor_residual_by_weather_paperlock': 'predictor_residual_by_weather_paperlock',
    'fig_oracle_gap_paperlock': 'oracle_gap_paperlock',
    'fig_ablation_paperlock': 'ablation_paperlock',
    'fig_stability_negative_result_paperlock': 'stability_negative_result_paperlock',
}

DST.mkdir(parents=True, exist_ok=True)
missing = []
for src_stem, dst_stem in MAP.items():
    copied = False
    for ext in ['.pdf', '.png']:
        src = SRC / f'{src_stem}{ext}'
        if src.exists():
            shutil.copy2(src, DST / f'{dst_stem}{ext}')
            copied = True
    if not copied:
        missing.append(src_stem)
if missing:
    raise SystemExit('Missing figures: ' + ', '.join(missing))
print(f'Copied {len(MAP)} figure groups to {DST}')
