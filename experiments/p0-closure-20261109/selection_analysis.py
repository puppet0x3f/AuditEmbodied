"""Choice-level analysis of already executed candidates; no new policy scores."""
from collections import defaultdict
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def longest(flags):
    count=maximum=0
    for value in flags:
        count=count+1 if value else 0;maximum=max(maximum,count)
    return maximum


def main():
    groups=defaultdict(dict)
    for line in (ROOT/'outputs/p0-20261109/intervention-trials.jsonl').open():
        row=json.loads(line);key=(row['seed'],row['scenario'])
        arm=0 if row['candidate']=='R_left' else 1
        assert row['post_fault']['holds'][1-arm]
        c=bool(row['task_complete']);p=not row['protection_violation']
        row['label']='A' if c and p else 'B' if c else 'C' if p else 'D'
        groups[key][(row['planned_wait_steps'],row['planned_duration_steps'])]=row
    assert len(groups)==16 and all(len(g)==6 for g in groups.values())
    cases=[];costs=[]
    for (seed,scenario),g in sorted(groups.items()):
        assert all(r['prefix_state']==next(iter(g.values()))['prefix_state'] for r in g.values())
        choose=lambda w:next((d for d in (30,40,60) if g[w,d]['label']=='A'),None)
        before,after=choose(0),choose(19);assert before is not None
        actual=g[19,before];costs.append(actual['total_post_fault_seconds'])
        cases.append({'seed':seed,'scenario':scenario,'zero_wait_choice_steps':before,
            'delayed_outcome_of_original_choice':actual['label'],'delayed_choice_steps':after,
            'choice_changed':before!=after,'original_choice_total_delayed_seconds':actual['total_post_fault_seconds'],
            'changed_choice_total_seconds':g[19,after]['total_post_fault_seconds'] if after else None})
    protection=json.loads((ROOT/'outputs/protection-validity-20261109/report.json').read_text())
    protocol=[]
    for cohort in ('scope_fault','supervision_fault'):
        for method,m in protection['methods'][cohort].items():
            protocol.append({'cohort':cohort,'method':method,'episodes':m['episodes'],
                'completion_only':m['terminal_completions'],'completion_with_shared_guards':m['physical_guard_only_completions'],
                'completion_with_full_obligations':m['strict_safe_completions']})
    last_attempt=[]
    for line in (ROOT/'outputs/agentchord-style-20261109/trials.jsonl').open():
        r=json.loads(line)
        if r['method']!='agentchord_style' or not r['protection_violation']:continue
        event=[e for e in r['skill_events'] if e['candidate']!='continue'][-1]
        frames=[f for f in r['trace'] if f['time']>event['start_time'] and f['phase'] in ('recovery','final_hold')]
        gap=longest([not(any(f['evaluator']['holds']) or f['evaluator']['table_contact']) for f in frames])*50
        last_attempt.append({'seed':r['seed'],'scenario':r['scenario'],'last_recovery':event['candidate'],
            'terminal_completion':r['task_complete'],'full_protected':False,
            'last_attempt_registered_support_gap_ms':gap,'last_attempt_also_fails_shared_guard':gap>100})
    result={'source':'96 frozen physical candidate executions, plus existing460episode protection decomposition',
        'initialization_seeds':8,'states':16,'candidate_executions':96,'matched_prefixes':True,
        'choice_rule':'First recorded A in nominal30/40/60step order; uses truth labels as an oracle diagnostic.',
        'zero_wait_selected_A':16,'same_choices_after_wait_A':sum(c['delayed_outcome_of_original_choice']=='A' for c in cases),
        'same_choices_after_wait_B':sum(c['delayed_outcome_of_original_choice']=='B' for c in cases),
        'wait_conditioned_observed_A_available':sum(c['delayed_choice_steps'] is not None for c in cases),
        'different_duration_available':sum(c['choice_changed'] and c['delayed_choice_steps'] is not None for c in cases),
        'no_registered_safe_candidate':sum(c['delayed_choice_steps'] is None for c in cases),
        'original_selected_delayed_mean_seconds':sum(costs)/len(costs),
        'cases':cases,'protocol_views':protocol,'graph_last_attempt_checks':last_attempt,
        'interpretation':['This is an oracle diagnostic over executed candidates, not a deployed policy or an estimated unknown counterfactual.',
          'One state has no observed A after waiting. Refusal was not executed, so its protected outcome and full cost remain unknown.',
          'A correctly specified whole-trajectory temporal monitor can reproduce the full verdict. This is not a logic-expressiveness claim.',
          'Both violating graph paths also violate a shared guard inside the last recovery attempt. This cohort does not demonstrate a verdict gain attributable only to retaining earlier history.']}
    (OUT/'selection-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for filename,rows in [('selection-cases.csv',cases),('protocol-views.csv',protocol)]:
        with (OUT/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'State & $W{=}0$ choice & Outcome after wait & New observed A \\',r'\midrule']
    for c in cases:
        if c['choice_changed']:
            label='L' if c['scenario']=='left_loss' else 'R';alternative=str(c['delayed_choice_steps']) if c['delayed_choice_steps'] else 'None'
            lines.append(f"{c['seed']}{label} & {c['zero_wait_choice_steps']} & {c['delayed_outcome_of_original_choice']} & {alternative} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'paper/tables/selection-change.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','protocol_views')},indent=2))


if __name__=='__main__':main()
