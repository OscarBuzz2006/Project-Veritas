import os
import json
import pandas as pd
import numpy as np

# Find the file
files = os.listdir('.')
print("Files in cwd:", files)
target_file = [f for f in files if 'numeric_runs' in f][0]

records = []
with open(target_file, 'r') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"Total records loaded: {len(records)}")

# Extract task metadata from first record's run_spec
run_spec = records[0]['run_spec']
tasks_meta = {t['id']: t for t in run_spec['tasks']}

rows = []
for r in records:
    qid = r['question_id']
    tmeta = tasks_meta.get(qid, {})
    truth = tmeta.get('numeric_truth')
    truth_log10 = np.log10(truth) if truth and truth > 0 else np.nan
    rows.append({
        'condition_id': r['condition_id'],
        'question_id': qid,
        'domain': tmeta.get('domain'),
        'stratum': tmeta.get('selection_stratum'),
        'bias_dir': tmeta.get('popular_misconception_bias'),
        'replicate': r['replicate'],
        'round_idx': r['round_idx'],
        'agent_id': r['agent_id'],
        'parsed_value': r['parsed_value'],
        'assistant_log10': r['assistant_log10'],
        'agent_log10_before': r['agent_log10_before'],
        'agent_log10_after': r['agent_log10_after'],
        'pop_median_log10': r['pop_median_log10'],
        'truth': truth,
        'truth_log10': truth_log10,
        'miss': r['miss'],
        'input_tokens': r['input_tokens'],
        'output_tokens': r['output_tokens'],
        'cost_usd': r['cost_usd']
    })

df = pd.DataFrame(rows)
df['is_control'] = df['domain'] == 'control'
df['asst_err_log10'] = df['assistant_log10'] - df['truth_log10']
df['agent_err_before_log10'] = df['agent_log10_before'] - df['truth_log10']
df['agent_err_after_log10'] = df['agent_log10_after'] - df['truth_log10']
df['belief_shift_log10'] = df['agent_log10_after'] - df['agent_log10_before']
df['anchor_diff_log10'] = df['agent_log10_before'] - df['truth_log10']

print("\n--- OVERVIEW ---")
print("Conditions:", df['condition_id'].unique())
print("Rounds:", df['round_idx'].unique())
print("Questions:", df['question_id'].nunique())
print("Miss rate:", df['miss'].mean())
print("Total cost USD:", df['cost_usd'].sum())
print("Total input tokens:", df['input_tokens'].sum(), "Total output tokens:", df['output_tokens'].sum())

print("\n--- BY CONDITION & ROUND (LOG10 RMSE & MEAN ERROR) ---")
def summarize(g):
    return pd.Series({
        'n': len(g),
        'asst_rmse_log10': np.sqrt(np.mean(g['asst_err_log10']**2)),
        'asst_bias_log10': np.mean(g['asst_err_log10']),
        'agent_rmse_before': np.sqrt(np.mean(g['agent_err_before_log10']**2)),
        'agent_rmse_after': np.sqrt(np.mean(g['agent_err_after_log10']**2)),
        'mean_abs_shift': np.mean(np.abs(g['belief_shift_log10'])),
        'pop_sd_after': g.groupby(['question_id'])['agent_log10_after'].std().mean()
    })

print(df.groupby(['condition_id', 'is_control', 'round_idx']).apply(summarize).to_string())

print("\n--- PER QUESTION SUMMARY (POST-CUTOFF vs CONTROL) ---")
q_sum = df.groupby(['condition_id', 'is_control', 'question_id', 'round_idx']).agg(
    truth=('truth', 'first'),
    asst_med=('parsed_value', 'median'),
    asst_rmse=('asst_err_log10', lambda x: np.sqrt(np.mean(x**2))),
    agent_rmse_before=('agent_err_before_log10', lambda x: np.sqrt(np.mean(x**2))),
    agent_rmse_after=('agent_err_after_log10', lambda x: np.sqrt(np.mean(x**2))),
    pop_med_log10=('pop_median_log10', 'first')
).reset_index()
print(q_sum.to_string())
