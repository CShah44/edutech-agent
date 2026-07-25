"""Paired Wilcoxon signed-rank tests: baseline vs multi-agent per model.
Uses 50-sample GPT-4.1 judge (answer_accuracy) and Llama-3.3-70B judge (overall) CSVs."""
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
import re

gpt_dir = Path('/Users/drd01/projects/MSR-thesis/EDUCATION_MULTI_AGENT/PROJECT_FILES/EDUCATION_MULTI_AGENT/llm-metrics-50-samples-gpt4.1')
llama_dir = Path('/Users/drd01/projects/MSR-thesis/EDUCATION_MULTI_AGENT/PROJECT_FILES/EDUCATION_MULTI_AGENT/llm_metrics_llama3.3')

def parse_gpt(f):
    m = re.search(r'(arch1|arch_1|baseline)_([a-zA-Z0-9._-]+?)_0_30000_ragas_llm\.csv', f.name)
    if not m: return None, None
    cond = 'agent' if m.group(1).startswith('arch') else 'baseline'
    return m.group(2), cond

def parse_llama(f):
    m = re.search(r'(arch1|arch_1|baseline)_([a-zA-Z0-9._-]+?)_0_30000_ragas_llm_gptoss_judge\.csv', f.name)
    if not m: return None, None
    cond = 'agent' if m.group(1).startswith('arch') else 'baseline'
    return m.group(2), cond

def run_paired(d, parser, score_col):
    groups = {}
    for f in sorted(d.glob('*.csv')):
        model, cond = parser(f)
        if model is None: continue
        df = pd.read_csv(f)
        if score_col not in df.columns: continue
        groups.setdefault(model, {})[cond] = df
    print(f'\n=== {d.name} ({score_col}) ===')
    for m, c in sorted(groups.items()):
        if 'baseline' not in c or 'agent' not in c:
            print(f'  {m}: missing condition (only {list(c)})')
            continue
        b, a = c['baseline'], c['agent']
        common = sorted(set(b['question_id']) & set(a['question_id']))
        if len(common) < 5:
            print(f'  {m}: overlap too small ({len(common)})')
            continue
        b = b.set_index('question_id').loc[common]
        a = a.set_index('question_id').loc[common]
        bv = pd.to_numeric(b[score_col], errors='coerce').dropna()
        av = pd.to_numeric(a[score_col], errors='coerce').dropna()
        idx = bv.index.intersection(av.index)
        bv, av = bv.loc[idx], av.loc[idx]
        diff = av - bv
        nonzero = (diff != 0).sum()
        if nonzero == 0:
            print(f'  {m}: n={len(idx)} all differences zero')
            continue
        try:
            stat, p = wilcoxon(av, bv, zero_method='wilcox', alternative='greater')
            cliff = ((av > bv).sum() - (av < bv).sum()) / len(av)
            print(f'  {m}: n={len(idx)} baseline_mean={bv.mean():.3f} agent_mean={av.mean():.3f} Δ={av.mean()-bv.mean():+.3f} W={stat:.1f} p(one-sided)={p:.4f} Cliff_δ={cliff:+.2f}')
        except Exception as e:
            print(f'  {m}: error {e}')

run_paired(gpt_dir, parse_gpt, 'answer_accuracy')
run_paired(llama_dir, parse_llama, 'overall')
