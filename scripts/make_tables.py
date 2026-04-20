#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import math

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'paper_project'
SRC = ROOT / 'results_paperlock' / 'tables'
DST = PROJECT / 'tables'
DST.mkdir(parents=True, exist_ok=True)

METHOD_RENAME = {
    'P3C-LR': 'P3C-LR Core',
    'Always-On Stability': 'P3C-SR',
    'Event-Triggered Stability': 'ET-P3C',
    'OracleLink Upper Bound': 'OracleLink Upper Bound',
    'OracleSelector Upper Bound': 'OracleSelector Upper Bound',
}
PRACTICAL = {'Random','Nearest','RateMax','MUCCO-like','Reactive-3C','P3C-LR Core','P3C-SR','ET-P3C'}
MINIMIZE = {'Avg Delay','p95 Delay','Energy','Handovers/100','Outage','Drop Rate','Objective'}
MAXIMIZE = {'Useful Cache Hit','Throughput'}


def esc(s):
    return str(s).replace('&','\\&').replace('%','\\%').replace('_','\\_')


def method_name(m):
    return METHOD_RENAME.get(str(m), str(m))


def tex_table(label, caption, headers, rows, align=None, size='\\scriptsize', star=True, note=None):
    env = 'table*' if star else 'table'
    if align is None:
        align = 'l' + 'c'*(len(headers)-1)
    lines = [f'\\begin{{{env}}}[!t]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}', size, f'\\begin{{tabular}}{{{align}}}', '\\toprule']
    lines.append(' & '.join(headers) + r' \\')
    lines.append('\\midrule')
    for row in rows:
        lines.append(' & '.join(row) + r' \\')
    lines += ['\\bottomrule', '\\end{tabular}']
    if note:
        lines.append(f'\\vspace{{1mm}}\\footnotesize{{{note}}}')
    lines.append(f'\\end{{{env}}}')
    return '\n'.join(lines) + '\n'


def pm(mean, ci, digits=2, scale=1.0):
    return f'{mean*scale:.{digits}f} $\\pm$ {ci*scale:.{digits}f}'


def fmt(x, digits=2, scale=1.0):
    return f'{float(x)*scale:.{digits}f}'


def best_map(df, cols):
    out = {}
    prac = df[df['Method'].map(method_name).isin(PRACTICAL)].copy()
    for c in cols:
        if c in MINIMIZE:
            idx = prac[c].astype(float).idxmin()
        else:
            idx = prac[c].astype(float).idxmax()
        out[c] = method_name(prac.loc[idx,'Method'])
    return out


def maybe_bold(method, col, text, best):
    name = method_name(method)
    if 'Oracle' in name:
        return r'\textit{' + text + '}'
    if best.get(col) == name:
        return r'\textbf{' + text + '}'
    return text

# Notation table
notation_rows = [
    ['$\\mathcal{S}$', 'Set of UAV swarms'],
    ['$s$', 'Index of a UAV swarm'],
    ['$\\mathcal{K}_t$', 'Active demand clusters at slot $t$'],
    ['$k$', 'Index of a demand cluster'],
    ['$x_{s,k,t}$', 'Binary assignment variable'],
    ['$\\mathcal{C}_{s,t}$', 'Cache state of swarm $s$'],
    ['$Q_{k,t}$', 'Backlog of demand cluster $k$'],
    ['$A_{k,t}$', 'New arrival demand'],
    ['$\\mu_{s,k,t}$', 'Expected service by swarm $s$ for cluster $k$'],
    ['$\\hat{m}_{s,k,t+1}$', 'Predicted next-slot link margin'],
    ['$\\hat{p}^{\\mathrm{out}}_{s,k,t}$', 'Predicted outage probability'],
    ['$V^{\\mathrm{cache}}_{s,k,t}$', 'Useful cache value'],
    ['$C^{\\mathrm{en}}_{s,k,t}$', 'Energy cost'],
    ['$D_{s,k,t}$', 'Delay cost'],
    ['$C^{\\mathrm{sw}}_{s,k,t}$', 'Switch/handover cost'],
    ['$B_{s,k,t}$', 'P3C-LR Core scheduling score'],
]
(DST/'notation_table.tex').write_text(tex_table('tab:notation','Main notation used in the system model.', ['Symbol','Meaning'], notation_rows, align='ll', star=False), encoding='utf-8')

# Related work comparison
rel_rows = [
    ['MUCCO~\\cite{wei2025mucco}', 'Energy-efficient data delivery', 'Yes', 'Coverage and grouping', 'No', 'Mobility-aware deployment', 'Partial', 'No ANN link-risk prediction or dwell-aware multi-swarm predictive 3C scheduling'],
    ['Post-disaster UAV swarm~\\cite{zheng2025postdisaster}', 'Transmission-rate maximization', 'No', 'Collaborative beamforming and routing', 'No', 'Placement optimization', 'Partial', 'No cache-value-aware dynamic content scheduling'],
    ['Dynamic task collaboration~\\cite{zhou2026dynamic}', 'Task collaboration and consensus', 'No', 'Control network', 'No', 'Two-layer asynchronous control', 'No', 'Focuses on task-volume collaboration, not service-level 3C orchestration'],
    ['P3C-Orch', 'Predictive service orchestration', 'Useful cache value', 'Link-risk-aware assignment', 'ANN link-risk', 'Dwell guard', 'Yes', 'Simulation-based; real UAV traces still needed'],
]
(DST/'comparison_related_work.tex').write_text(tex_table('tab:related_work','Comparison with closely related UAV-swarm and LAE works.', ['Work','Main focus','Caching','Routing/communication','Prediction','Mobility/handover','3C orchestration','Main limitation'], rel_rows, align='p{0.105\\textwidth}p{0.105\\textwidth}p{0.065\\textwidth}p{0.115\\textwidth}p{0.065\\textwidth}p{0.105\\textwidth}p{0.065\\textwidth}p{0.155\\textwidth}', size='\\tiny', star=True), encoding='utf-8')

# Predictor tables
pred = pd.read_csv(SRC/'table_predictor_metrics_paperlock.csv').iloc[0]
pred_rows = [[fmt(pred['MSE'],3), fmt(pred['RMSE'],3), fmt(pred['MAE'],3), fmt(pred['R2'],3), fmt(pred['Pearson_R'],3), fmt(pred['prediction_time_ms_per_sample'],4)]]
(DST/'predictor_metrics.tex').write_text(tex_table('tab:predictor_metrics','Overall ANN link-margin predictor metrics on the paper-lock predictor evaluation set.', ['MSE','RMSE','MAE','$R^2$','Pearson $R$','Time/sample (ms)'], pred_rows, star=False), encoding='utf-8')

weather = pd.read_csv(SRC/'table_predictor_by_weather_paperlock.csv')
weather_rows=[]
for _,r in weather.iterrows():
    weather_rows.append([esc(r['weather_state']), fmt(r['MSE'],3), fmt(r['RMSE'],3), fmt(r['MAE'],3), fmt(r['R2'],3), str(int(r['count'])), fmt(r['sigma_db'],3)])
(DST/'predictor_weather.tex').write_text(tex_table('tab:predictor_weather','Weather-wise link predictor error and calibrated residual scale.', ['Weather','MSE','RMSE','MAE','$R^2$','Samples','$\\sigma_w$ (dB)'], weather_rows, star=False), encoding='utf-8')

# Main results
main = pd.read_csv(SRC/'table_main_results_paperlock.csv')
cols = ['Avg Delay','p95 Delay','Energy','Handovers/100','Useful Cache Hit','Outage','Drop Rate','Throughput','Objective']
best = best_map(main, cols)
main_rows=[]
for _,r in main.iterrows():
    m=method_name(r['Method'])
    cells=[r'\textit{'+m+'}' if 'Oracle' in m else esc(m)]
    vals = {
        'Avg Delay': pm(r['Avg Delay'], r['Avg Delay CI95'], 2),
        'p95 Delay': pm(r['p95 Delay'], r['p95 Delay CI95'], 2),
        'Energy': pm(r['Energy']/1000.0, r['Energy CI95']/1000.0, 2),
        'Handovers/100': pm(r['Handovers/100'], r['Handovers/100 CI95'], 2),
        'Useful Cache Hit': pm(r['Useful Cache Hit'], r['Useful Cache Hit CI95'], 2, 100),
        'Outage': pm(r['Outage'], r['Outage CI95'], 2, 100),
        'Drop Rate': pm(r['Drop Rate'], r['Drop Rate CI95'], 2, 100),
        'Throughput': pm(r['Throughput'], r['Throughput CI95'], 1),
        'Objective': pm(r['Objective'], r['Objective CI95'], 3),
    }
    for c in cols:
        cells.append(maybe_bold(r['Method'], c, vals[c], best))
    main_rows.append(cells)
(DST/'main_results.tex').write_text(tex_table('tab:main_results','Combined-stress paper-lock results on blind seeds 4001--4080. Energy is in kJ. Cache, outage, and drop are in percent. Oracle methods are upper bounds and excluded from practical ranking.', ['Method','Avg Delay','p95 Delay','Energy','Handovers/100','Useful Cache','Outage','Drop','Throughput','Objective'], main_rows, align='lccccccccc', size='\\tiny', star=True), encoding='utf-8')

# External comparison from known improvements computed from main
main_idx = {method_name(r['Method']): r for _,r in main.iterrows()}
metric_defs=[('Avg Delay','min'),('p95 Delay','min'),('Energy','min'),('Handovers/100','min'),('Useful Cache Hit','max'),('Outage','min'),('Drop Rate','min'),('Throughput','max'),('Objective','min')]
def imp(prop, base, metric, direction):
    p=float(main_idx[prop][metric]); b=float(main_idx[base][metric])
    if abs(b) < 1e-12:
        return 'n/a'
    val = 100*(b-p)/b if direction=='min' else 100*(p-b)/b
    return f'{val:+.2f}\\%'

ext_rows=[]
for b in ['Reactive-3C','MUCCO-like','RateMax']:
    ext_rows.append([b] + [imp('P3C-LR Core', b, m, d) for m,d in metric_defs])
(DST/'external_baseline_comparison.tex').write_text(tex_table('tab:external_baseline','Percentage change of P3C-LR Core against external baselines under combined stress. Positive values favour P3C-LR Core; negative values show worsening.', ['Baseline','Avg Delay','p95 Delay','Energy','Handovers','Useful Cache','Outage','Drop','Throughput','Objective'], ext_rows, align='lccccccccc', size='\\scriptsize', star=True), encoding='utf-8')

int_rows=[]
base='P3C-LR Core'
for m in ['P3C-LR Core','P3C-SR','ET-P3C']:
    r=main_idx[m]
    int_rows.append([m, fmt(r['Avg Delay'],2), fmt(r['p95 Delay'],2), fmt(r['Energy']/1000,2), fmt(r['Handovers/100'],2), fmt(r['Useful Cache Hit']*100,2), fmt(r['Outage']*100,2), fmt(r['Drop Rate']*100,2), fmt(r['Throughput'],1), fmt(r['Objective'],3)])
(DST/'internal_variant_comparison.tex').write_text(tex_table('tab:internal_variant','Internal P3C-Orch variant comparison under combined stress. Energy is in kJ and ratio metrics are in percent.', ['Variant','Avg Delay','p95 Delay','Energy','Handovers','Useful Cache','Outage','Drop','Throughput','Objective'], int_rows, align='lccccccccc', size='\\scriptsize', star=True), encoding='utf-8')

# Regime table compact: all regimes for main practical methods
reg = pd.read_csv(SRC/'table_regime_results_paperlock.csv')
keep = ['Reactive-3C','MUCCO-like','RateMax','P3C-LR','Always-On Stability','Event-Triggered Stability']
reg_rows=[]
for _,r in reg[reg['Method'].isin(keep)].iterrows():
    reg_rows.append([esc(r['Regime'].replace('_',' ')), method_name(r['Method']), fmt(r['Avg Delay'],2), fmt(r['p95 Delay'],2), fmt(r['Energy']/1000,2), fmt(r['Handovers/100'],2), fmt(r['Useful Cache Hit']*100,2), fmt(r['Outage']*100,2), fmt(r['Drop Rate']*100,2), fmt(r['Objective'],3)])
(DST/'regime_results.tex').write_text(tex_table('tab:regime_results','Regime-wise paper-lock results for main practical methods. Energy is in kJ and ratio metrics are in percent.', ['Regime','Method','Avg Delay','p95 Delay','Energy','Handovers','Useful Cache','Outage','Drop','Objective'], reg_rows, align='llcccccccc', size='\\tiny', star=True), encoding='utf-8')

# Ablation
abl = pd.read_csv(SRC/'table_ablation_paperlock.csv')
var_rename={'P3C-LR full':'P3C-LR Core','Always-On Stability':'P3C-SR','Event-Triggered Stability':'ET-P3C'}
abl_rows=[]
for _,r in abl.iterrows():
    v=var_rename.get(r['Variant'], r['Variant'])
    abl_rows.append([esc(v), fmt(r['Avg Delay'],2), fmt(r['p95 Delay'],2), fmt(r['Energy']/1000,2), fmt(r['Handovers/100'],2), fmt(r['Useful Cache Hit']*100,2), fmt(r['Outage']*100,2), fmt(r['Drop Rate']*100,2), fmt(r['Throughput'],1)])
(DST/'ablation_results.tex').write_text(tex_table('tab:ablation','Ablation results under combined stress. Energy is in kJ and ratio metrics are in percent.', ['Variant','Avg Delay','p95 Delay','Energy','Handovers','Useful Cache','Outage','Drop','Throughput'], abl_rows, align='lcccccccc', size='\\scriptsize', star=True), encoding='utf-8')

# Significance: all comparisons, compact labels
sig = pd.read_csv(SRC/'table_significance_paperlock.csv')
metric_map={
 'avg_delay':'Avg delay','p95_delay':'p95 delay','total_energy':'Energy','handovers_per_100_active_cluster_slots':'Handovers','useful_cache_hit_ratio':'Useful cache','outage_probability':'Outage','drop_rate':'Drop rate','throughput':'Throughput'}
sig_rows=[]
for _,r in sig.iterrows():
    base = METHOD_RENAME.get(r['baseline'], r['baseline'])
    metric = metric_map.get(r['metric'], r['metric'])
    p = float(r['holm_wilcoxon_p'])
    supported = 'Yes' if p < 0.05 else 'No'
    sig_rows.append([base, metric, f'{p:.3g}', f'{float(r["effect_size_dz"]):+.3f}', supported])
(DST/'significance_results.tex').write_text(tex_table('tab:significance','Holm-corrected Wilcoxon tests comparing P3C-LR Core with each baseline under combined stress. The Yes column only marks corrected $p<0.05$ and does not imply that the direction favours P3C-LR Core.', ['Baseline','Metric','Corrected $p$','Effect $d_z$','$p<0.05$'], sig_rows, align='llccc', size='\\tiny', star=True), encoding='utf-8')
print('Generated LaTeX tables in', DST)
