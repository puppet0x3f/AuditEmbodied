"""Summarize the frozen 96-episode run and its physical suffix diagnosis."""
from pathlib import Path
from collections import Counter
from itertools import combinations
import ast
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def read(path):
    return [json.loads(line) for line in path.open()]


def label(row):
    return ('B' if row['protection_violation'] else 'A') if row['success'] else ('D' if row['protection_violation'] else 'C')


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def costs(rows):
    return {name: mean(r[key] for r in rows) for name, key in (
        ('total_s', 'total_post_fault_seconds'), ('wait_s', 'inference_wait_seconds'),
        ('skill_s', 'skill_seconds'), ('hold_s', 'hold_seconds'),
        ('wall_s', 'total_post_fault_wall_seconds'), ('pipeline_s', 'total_pipeline_wall_seconds'))}


def distribution(values):
    values = list(values)
    return {'n': len(values), 'median': statistics.median(values), 'max': max(values)} if values else None


report = json.loads((ROOT / 'closed-loop96/report.json').read_text())
rows = read(ROOT / 'closed-loop96/trials.jsonl')
counter = json.loads((ROOT / 'counterfactual/report.json').read_text())
suffixes = read(ROOT / 'counterfactual/cases.jsonl')
diagnostic = json.loads((ROOT / 'diagnostic22/report.json').read_text())
offline = json.loads((ROOT / 'offline-comparison.json').read_text())
training = json.loads((ROOT / 'new1024/report.json').read_text())
methods = ['fixed60', 'current_legacy', 'current_consistent', 'new_consistent']
by = {m: {(r['seed'], r['scenario']): r for r in rows if r['method'] == m} for m in methods}
keys = set.intersection(*(set(by[m]) for m in methods))
assert len(rows) == 96 and len(keys) == 24 and all(len(by[m]) == 24 for m in methods)
assert {r['seed'] for r in rows} == set(range(54000, 54012))
assert all(not r.get('error') and not r['blocked_calls'] and all(r[k] == 0 for k in
    ('state_edit_count_after_initialization', 'contact_anchor_count', 'external_wrench_at_end')) for r in rows)
assert all(abs(r['total_post_fault_seconds'] - r['inference_wait_seconds'] - r['skill_seconds'] - r['hold_seconds']) < 1e-7 for r in rows)
assert all(not r['diagnostic_errors'] for r in read(ROOT / 'counterfactual/trials.jsonl'))
for path in ('frozen.json', 'pretest-source-hashes.json'):
    for name, digest in json.loads((ROOT / path).read_text()).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
old = ast.parse((ROOT / 'source-before/l4_lift.py').read_text())
now = ast.parse((PROJECT / 'auditembodied/l4_lift.py').read_text())
for name in ('LiftConfig', 'TwoArmLiftEnv'):
    select = lambda tree: next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    assert ast.dump(select(old)) == ast.dump(select(now)), name

results = {'independent_episodes': 96, 'physical_states': 24, 'seeds': list(range(54000, 54012)),
    'methods': report['methods'], 'counterfactual': counter['methods'], 'costs': {}, 'timing': {},
    'common_safe': {}, 'paired_outcomes': {}, 'upgrade_attribution': {}, 'labels': {},
    'initial_duration_outcomes': {}, 'counterfactual_physical_suffixes': counter['new_physical_suffixes'],
    'training': training, 'offline_comparison': offline, 'diagnostic22': diagnostic,
    'checks': {'all_waits_charged': True, 'physical_prefix_pairing': True, 'cumulative_guards': True,
        'original_models_threshold_gains_force_unchanged': True, 'frozen_test_sources_unchanged': True,
        'performance_gate': False, 'diagnostics_used_for_training': False},
    'scope': 'Native closed loops use actual measured latency. Frozen initial duration weights can still select different durations after different actual waits. Middle-policy comparisons are conditional on actual N60 entries; do not attribute whole-loop differences only to the middle LoRA.'}
for method in methods:
    group = list(by[method].values())
    windows = [w for r in group for w in r.get('mid_selection', {}).get('windows', [])]
    timing = [t for r in group for t in r.get('async_timing', []) if t['kind'].startswith('mid_tail')]
    results['costs'][method] = costs(group)
    results['labels'][method] = dict(Counter(label(r) for r in group))
    results['initial_duration_outcomes'][method] = {
        str(n): dict(Counter(label(r) for r in group if r.get('local_steps', {}).get('local_relevel_steps') == n))
        for n in (30, 40, 60)}
    results['timing'][method] = {'windows': len(windows),
        'ready_and_matching': sum(w['forecast_ready'] and w['continuation_matches_forecast'] for w in windows),
        'not_ready': sum(not w['forecast_ready'] for w in windows),
        'stale': sum(not w['continuation_matches_forecast'] for w in windows),
        'client_tail_s': distribution(t['pipeline_wall_seconds'] for t in timing),
        'worker_tail_s': distribution(t['worker_seconds'] for t in timing),
        'exit_reasons': dict(Counter(r.get('abstention_reason', r.get('failure_reason')) for r in group if r.get('abstained')))}
    cases = [r for r in suffixes if r['method'] == method]
    needed = [r for r in cases if r['labels']['LOCAL60'] != 'A' and any(r['labels'][k] == 'A' for k in ('GLOBAL30', 'GLOBAL40'))]
    def source(case):
        if not case['windows']: return 'fixed'
        w = case['windows'][-1]
        if not w['forecast_ready']: return 'deadline'
        if not w['continuation_matches_forecast']: return 'stale'
        if w['baseline_fallback']: return 'unadmitted_fallback'
        if w['decision'].get('model_admitted'): return 'model'
        return 'no_admissible_plan'
    results['upgrade_attribution'][method] = {
        'needed_misses_by_source': dict(Counter(source(r) for r in needed if r['actual_label'] != 'A')),
        'necessary_safe_upgrades_by_source': dict(Counter(source(r) for r in needed if r['actual_label'] == 'A' and r['actual_branch'].startswith('GLOBAL'))),
        'unnecessary_upgrades_by_source': dict(Counter(source(r) for r in cases if r['actual_branch'].startswith('GLOBAL') and r['labels']['LOCAL60'] == 'A')),
        'recoverable_exits': sum(r['actual_branch'] == 'reject' and 'A' in r['labels'].values() for r in cases),
        'bad_local_choices': [{k: r[k] for k in ('seed', 'scenario', 'actual_label', 'labels')} for r in cases if r['actual_branch'] == 'LOCAL60' and r['actual_label'] != 'A']}

for subset in [methods, *[list(pair) for pair in combinations(methods, 2)]]:
    common = sorted(k for k in keys if all(label(by[m][k]) == 'A' for m in subset))
    results['common_safe']['/'.join(subset)] = {'states': len(common),
        'costs': {m: costs([by[m][k] for k in common]) for m in subset}}
for first, second in combinations(methods, 2):
    results['paired_outcomes'][first + '/' + second] = dict(Counter(label(by[first][k]) + '/' + label(by[second][k]) for k in keys))

cfby = {(r['seed'], r['scenario'], r['method']): r for r in suffixes}
fields = ['seed', 'scenario', 'method', 'label', 'N', 'window_actions', 'timely', 'abstained',
    'failure_reason', 'total_s', 'wait_s', 'skill_s', 'hold_s', 'pipeline_s', 'LOCAL60', 'GLOBAL30', 'GLOBAL40']
with (ROOT / 'cases.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for r in rows:
        windows = r.get('mid_selection', {}).get('windows', [])
        cf = cfby.get((r['seed'], r['scenario'], r['method']), {})
        writer.writerow({'seed': r['seed'], 'scenario': r['scenario'], 'method': r['method'], 'label': label(r),
            'N': r.get('local_steps', {}).get('local_relevel_steps'),
            'window_actions': ';'.join(str(w['decision_step']) + ':' + w['selected_branch'] for w in windows),
            'timely': sum(w['forecast_ready'] and w['continuation_matches_forecast'] for w in windows),
            'abstained': bool(r.get('abstained')), 'failure_reason': r.get('failure_reason'),
            **{k: r[v] for k, v in [('total_s', 'total_post_fault_seconds'), ('wait_s', 'inference_wait_seconds'),
                ('skill_s', 'skill_seconds'), ('hold_s', 'hold_seconds'), ('pipeline_s', 'total_pipeline_wall_seconds')]},
            **cf.get('labels', {})})
(ROOT / 'RESULTS.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')

display = {'fixed60': '固定60步', 'current_legacy': '当前校正LoRA＋旧接管',
    'current_consistent': '当前校正LoRA＋准入修复', 'new_consistent': '本轮LoRA＋准入修复'}
lines = [
    '# 第48轮：一致准入修复、分类概率校正与96回合独立闭环', '',
    '完成11状态22回合冻结诊断、从原中途512步检查点开始的1024步训练，以及54000–54011种子、左右故障、四方法共96回合物理仿真。'
    f'另从实际中途前缀执行{counter["new_physical_suffixes"]}条未选物理续段，覆盖{counter["states"]}个方法条件下的中途状态，用于标签与归因诊断。没有新增训练采集，没有以训练分类或拒绝率作为开跑门槛。', '',
    '**结论：未准入自动转全局的问题已修复；新训练改善了离线训练覆盖，但本轮没有同时保持在线安全覆盖并取得含全部等待的成本优势。**'
    '新LoRA安全完成10/24、过程违例0；仅4/14个中途窗口及时且匹配，10次超时退出损失覆盖。固定60步安全完成18/24。', '',
    '## 控制修复与冻结条件', '',
    '中途续段统一采用同一个正准入条件joint P(A)≥0.999667229515218。候选均未准入时执行有限退出与最终保持；'
    '不再要求所有候选同时达到极强负置信度才拒绝，也不把不确定性自动转换为全局动作。迟到、过期或明确未准入的全局请求同样不能直接执行。'
    '当前没有经过确认、仍有效的备用续段，因此本轮超时使用退出。退出仍保留此前全部保护记录，且不等于任务完成。', '',
    '起点LoRA、旧后果分支、初始时长分支、准入阈值、动作预算、控制增益、力限和桌面模板冻结；LiftConfig与TwoArmLiftEnv的AST保持一致。'
    '旧接管仅作为显式对照保留。四方法使用同一故障物理前缀，轮换执行顺序，分别计入自身实测等待。', '',
    '## 11状态22回合冻结诊断', '',
    '|控制逻辑|终态完成|安全完成|过程违例|拒绝退出|完整20Hz成本均值/秒|',
    '|---|---:|---:|---:|---:|---:|']
for name, title in [('legacy_global', '旧自动全局'), ('consistent_exit', '一致准入退出')]:
    d = diagnostic['methods'][name]
    lines.append(f'|{title}|{d["successes"]}/11|{d["safe_successes"]}/11|{d["process_violations"]}|{d["abstentions"]}|{d["mean_total_post_fault_seconds"]:.3f}|')
lines += ['', '22回合固定同一组当前LoRA预测，重放原始完整动作前缀及全部既有等待，边界状态与目标逐项相同。'
    '4个状态在边界前已有违例；旧自动全局使另外5个状态新增违例，退出没有新增违例状态，但也放弃了2个本可由全局安全完成的状态。'
    '11组中6组所有候选均非A，5组存在A。这里只验证控制逻辑，不将冻结读出重放当作新的在线推理延迟测量。', '',
    '## 同预算训练与离线结果', '',
    '初始化固定为outputs/vlm-mid-continuation-20261109/set512/adapter；严格配对采样与1024项索引和上轮一致，共2560次候选曝光。'
    '本轮损失为w×(CE＋0.5完成＋0.5过程)＋0.1配对＋0.1集合准入；只有前三项做采样概率校正。'
    f'训练耗时{training["training_seconds"]:.2f}秒，峰值显存{training["max_allocated_gib"]:.3f}GiB，使用zh-server L20；仅评估最终检查点。', '',
    '以下两模型均使用修复后的一致选择器，避免把拒绝语义变化算作训练收益。物理家族只计一次，单窗口可选择LOCAL60、GLOBAL30或预先安排GLOBAL40。', '',
    '|离线指标|当前校正LoRA|本轮LoRA|', '|---|---:|---:|']
oldtrain = offline['current']['train']['consistent']['policies']['single']
newtrain = offline['new']['train']['consistent']['policies']['single']
olddev = offline['current']['development']['consistent']['policies']['single']
newdev = offline['new']['development']['consistent']['policies']['single']
for text, key in [('训练安全覆盖（112个可恢复家族）', 'safe_state_coverage'),
        ('训练已选非A', 'model_selected_unsafe_choices'), ('训练可恢复家族被拒绝', 'safe_states_rejected'),
        ('训练全非A拒绝（6组）', 'no_safe_rejected'), ('训练必要升级漏判（6组）', 'needed_upgrade_misses')]:
    lines.append(f'|{text}|{oldtrain[key]}|{newtrain[key]}|')
lines += [f'|开发安全覆盖（38组）|{olddev["safe_state_coverage"]}|{newdev["safe_state_coverage"]}|',
    f'|开发多余升级|{olddev["unnecessary_upgrades"]}|{newdev["unnecessary_upgrades"]}|', '',
    '候选分类训练566/590→570/590，开发185/190→184/190；候选误放行训练均1、开发均0。'
    '候选误放行与实际选中非A不是同一指标。开发所有候选真值均为A，因此38/38不能证明危险或无安全候选状态的拒绝泛化。', '',
    '## 96回合真实在线比较', '',
    '本机原物理环境执行，zh-server GPU3持久进程推理；图像获取与传输、初始及中途推理、控制超周期、验证保持均计入。'
    '完整20Hz成本为实际技能＋物理等待＋最终保持；流水线墙钟另外包含实际前后处理和回合结束后的后台推理排空。一次性的加载与训练池预热单列于setup.json。', '',
    '|方法|终态完成/24|安全完成/24|过程违例|拒绝退出|20Hz总成本/秒|其中等待/秒|完整流水线墙钟/秒|',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for m in methods:
    d = report['methods'][m]
    lines.append(f'|{display[m]}|{d["successes"]}|{d["safe_successes"]}|{d["process_violations"]}|{d["abstentions"]}|{d["mean_total_post_fault_seconds"]:.3f}|{d["mean_wait_seconds"]:.3f}|{d["mean_pipeline_wall_seconds"]:.3f}|')
lines += ['', '所有行均含失败与退出，不能把更快退出解释为恢复效率提高。初始时长分支虽然权重不变，实际等待后的输入不同，选出的30/40/60步也会不同。'
    '当前LoRA修复版的两次过程违例均来自初始40步分支，并未进入中途窗口。整段差异不能全归因于中途LoRA。', '',
    '|方法|及时且匹配/中途窗口|迟到|过期|未准入全局实际执行|尾段往返中位数/秒|', '|---|---:|---:|---:|---:|---:|']
for m in methods[1:]:
    d = report['methods'][m]; t = results['timing'][m]
    lines.append(f'|{display[m]}|{t["ready_and_matching"]}/{t["windows"]}|{t["not_ready"]}|{t["stale"]}|{d["unadmitted_mid_global_executions"]}|{t["client_tail_s"]["median"]:.3f}|')
lines += ['', '第20步查询供第30步消费，承诺执行区间为10步／0.5秒。尾段往返中位数超过该窗口，导致大量结果未能实际消费。'
    '本轮LoRA的4次及时动作是1次继续与3次GLOBAL30；当前LoRA修复版有12次迟到与1次过期退出，新LoRA有10次迟到退出。', '',
    '## 相同实际前缀的未选物理续段', '',
    f'从{counter["states"]}个实际N60中途状态出发，重放各自全部既有等待与动作，执行LOCAL60、GLOBAL30、GLOBAL40的未选分支；新增{counter["new_physical_suffixes"]}条物理续段。'
    '保护从原故障后一直累计，无状态恢复、接触锚或外力。下表只覆盖实际进入N60中途窗口的状态，搜索耗尽和初始短时长分支仍在整段96回合中计分。', '',
    '|方法|N60状态|存在A|安全覆盖|必要升级漏判/必要状态|误放行候选/已准入候选|可恢复状态退出|无A退出/无A状态|',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for m in methods:
    d = counter['methods'][m]; u = results['upgrade_attribution'][m]
    lines.append(f'|{display[m]}|{d["N60_states"]}|{d["recoverable_states"]}|{d["safe_coverage"]}|{d["needed_upgrade_misses"]}/{d["needed_upgrade_states"]}|{d["false_admitted_candidates"]}/{d["candidate_admissions"]}|{u["recoverable_exits"]}|{d["no_safe_rejections"]}/{d["no_safe_states"]}|')
lines += ['', '误放行只统计及时且匹配的模型候选；旧未准入接管另列，不能算成模型主动准入。'
    '必要升级漏掉的原因、模型决定与接管归因、全部逐状态标签见RESULTS.json和cases.csv。若无A状态为0，拒绝能力的独立泛化在该组没有可估计的分母。', '',
    '## 共同安全状态成本', '',
    '以下只比较四方法都安全完成的相同故障状态；仍使用每种方法自身真实等待。', '',
    '|方法|共同安全状态数|20Hz成本/秒|完整流水线墙钟/秒|', '|---|---:|---:|---:|']
common = results['common_safe']['/'.join(methods)]
for m in methods:
    c = common['costs'][m]
    lines.append(f'|{display[m]}|{common["states"]}|{c["total_s"]:.3f}|{c["pipeline_s"]:.3f}|' if common['states'] else f'|{display[m]}|0|—|—|')
lines += ['', '两两共同安全成本与标签交叉表同时保存在RESULTS.json；共同安全子集不能替代全24状态覆盖。', '',
    '## 实现、检查与复现', '',
    '实现位于vlm_mid_recovery.py、l4_lift.py、mid_recovery_train.py及现有持久推理与闭环入口；新增admission_diagnostic.py只负责冻结诊断。'
    '新增检查验证一致准入、迟到／过期处理以及分类概率权重不缩放配对和集合准入梯度。CPU检查两组分别6通过2跳过、12通过2跳过；Torch两组各2通过。'
    '模型与阈值哈希、控制类AST、96回合物理前缀、全部成本闭合及续段原等待重放检查通过；无执行异常、状态编辑、接触锚或外力。', '',
    '工程试跑8回合仅使用训练池种子40，不算独立测试。diagnostic22-incomplete目录保留了首次诊断中因固定重放接口遗漏reject选项而中止的尝试；'
    '补齐执行接口后重新完成全部22回合，不将中止尝试混入正式统计。既有诊断、54000测试及其续段均未回填训练。', '',
    f'- [完整机器汇总]({ROOT}/RESULTS.json)', f'- [逐回合表]({ROOT}/cases.csv)',
    f'- [96回合原始结果]({ROOT}/closed-loop96/report.json)', f'- [真实续段报告]({ROOT}/counterfactual/report.json)',
    f'- [22回合诊断]({ROOT}/diagnostic22/report.json)', f'- [最终训练结果]({ROOT}/new1024/report.json)',
    f'- [最终LoRA]({ROOT}/new1024/adapter)', f'- [闭环命令]({ROOT}/closed-loop-command.txt)',
    f'- [训练命令]({ROOT}/train-command.txt)', f'- [可复算汇总脚本]({ROOT}/summarize-results.py)', '',
    '本轮按固定协议完成，保留新LoRA为实验检查点；未额外调阈值、补采训练或追选检查点，也未改动论文与其他会话独立实验。']
(ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({k: results[k] for k in ('methods', 'counterfactual', 'timing', 'upgrade_attribution', 'common_safe')}, ensure_ascii=False, indent=2))
