"""Record the completed round and prepend measured results to the research log."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
plan=read(root/'experiment-plan.json');a=read(root/'assessment.json');s=read(root/'final-serving.json')
bench=read(root/'benchmark-shared.json');live=read(root/'closed-loop/report.json')
before=a['metrics']['dev_before']['W66'];after=a['metrics']['dev_after']['W66']
old_before=a['metrics']['old_train_before']['all'];old_after=a['metrics']['old_train_after']['all']
new=a['metrics']['new_train_after']['all'];b=live['methods']['fixed60'];m=live['methods']['vlm_shared']
decision={'status':'ROUND_COMPLETE_FINAL512_AND_INDEPENDENT_PHYSICS','new_train_physical_episodes':96,
    'training_examples':168,'training_steps':512,'candidate_presentations':1024,
    'threshold':plan['threshold'],'threshold_changed':False,'frozen_unchanged':a['frozen_unchanged'] and live['frozen_unchanged'],
    'inference_optimization':'Cache the same h1 prerequisite; encode the same8 images once; score ascending-cost candidates and return the first admitted one; unchanged selection and logits.',
    'frozen_training_benchmark':{k:bench[k] for k in ('requests','same_admission','same_selection','probability_max_error','batch_pipeline_mean','batch_deadline_misses')},
    'final_serving':s,'development_before':before,'development_after':after,
    'old_training_before':old_before,'old_training_after':old_after,'new_training_after':new,
    'development_reduced_false_admission_with_preserved_state_coverage':after['false_admitted_candidates']<before['false_admitted_candidates'] and after['safe_covered_families']>=before['safe_covered_families'],
    'independent_closed_loop':live,'training_accuracy_gate':None,
    'assessment':str(root/'assessment.json'),'scope':'The independent pilot is8 new states and16 actual executions, with no repair-role oracle. Matrix and live costs remain separate. All live refusals and late waits count.'}
decision['main_finding']='Old-state safe choices retained16/16. Dev W66 false admissions4->3 but safe coverage4/4->3/4 and safe selection stays1/4. Independent VLM responses all on time; task success7/8 ties fixed60, safe success6/8 versus7/8, process violations2 versus0. Mean post-fault cost12.53125s versus9.34375s. The simultaneous generalization/coverage goal and closed-loop benefit were not achieved.'
decision['independent_cost_delta_seconds']=m['mean_total_post_fault_seconds']-b['mean_total_post_fault_seconds']
decision['independent_cost_increase_fraction']=m['mean_total_post_fault_seconds']/b['mean_total_post_fault_seconds']-1
decision['new_checkpoint_selected_for_default_policy']=False
(root/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
headers=['指标','微调前','本轮最终512步']
table=['| '+' | '.join(headers)+' |','| --- | ---: | ---: |']
for title,key,den in [('开发W66安全候选放行','safe_admitted_candidates','safe_candidates'),
    ('开发W66非A误放行','false_admitted_candidates',None),
    ('开发W66可恢复状态覆盖','safe_covered_families','recoverable_families'),
    ('开发W66实际安全选择','safe_selected_families','families'),
    ('旧72条安全候选放行','safe_admitted_candidates','safe_candidates'),
    ('旧72条非A误放行','false_admitted_candidates',None)]:
    pair=(old_before,old_after) if title.startswith('旧') else (before,after)
    fmt=lambda r:str(r[key])+(f'/{r[den]}' if den else '')
    table.append(f'| {title} | {fmt(pair[0])} | {fmt(pair[1])} |')
text=f'''## 2026-11-09 第三十七轮：新增96条训练干预、均衡回放512步与独立闭环完成

**本轮未达到“同时降低误放行并保持安全覆盖”的目标。旧训练状态安全选择16/16保持，开发W66误放行4→3但覆盖4/4→3/4，安全选择仍1/4。新种子8次推理全部按时返回；VLM安全完成6/8，固定60步7/8，平均故障后耗时12.531秒与9.344秒。推理执行时序已经在这组独立试验中打通，泛化和包含等待的控制收益仍未成立。**

本轮从当前联合家族full512检查点继续训练后果LoRA，使用旧72条＋新96条，固定512步、1024次候选呈现，旧/新各512次。起点LoRA、旧分支、损失、阈值 **{plan['threshold']}** 和补抓后66步/3.3秒预算固定；只评估最终检查点，没有训练分类100%门槛。

新增种子48–55、左右故障共16个有效故障起点，每个起点实际执行N30/40/60×W0/66，共96条（70 A、18 B、6 C、2 D）。故障打开时间按事前规则取8/16步，未改变恢复几何、控制增益或力限。共同前缀和观察已逐一核对。新增4个无安全候选家族；两组严格40/60对照为40步B、60步A。H1采图耗时额外转为12–14步物理保持；同一起点六个分支重放相同保持，未将旧轨迹标签移接到新等待状态。

普通三候选批处理首次测试超时18/36次，最大概率差0.218，未用于本轮闭环。随后采用**H1起点缓存＋同一8幅图的视觉特征复用＋原样单候选语言前向**：36次训练请求放行和选择全部一致、概率差0；同轮原串行平均3.060秒、优化后{bench['batch_pipeline_mean']:.3f}秒，优化后超时{bench['batch_deadline_misses']}/36。最终模型完整开发评分概率最大差{s['max_probability_error']}，W66完整评分超时{s['W66_deadline_misses']}/4。因此实际部署进一步按成本顺序评分，首个通过即返回，避免计算不会改变选择的后续候选；仍保留完整离线评分。这一执行优化的开发选择逐一等同完整评分，实测超时{s['deployment_deadline_misses']}/4。共享GPU负载会影响耗时，历史时延不作为同负载因果比较。

{chr(10).join(table)}

新96条训练的安全放行为{new['safe_admitted_candidates']}/{new['safe_candidates']}，非A误放行{new['false_admitted_candidates']}；无安全家族拒绝{new['no_safe_families_rejected']}/{new['no_safe_families']}。开发集没有无安全家族，其拒绝能力不能由该开发集证明。矩阵费用使用真实候选轨迹，拒绝只报告已观察到的前缀费用，不虚构终止轨迹。

独立闭环使用此前未执行的3000–3003种子×左右故障，共8个状态、16条实际轨迹。两方法使用相同冻结范围选择模型，修复臂由模型决定；固定60基线无故障角色Oracle。学习策略安全完成 **{m['safe_successes']}/8**、任务成功{m['successes']}/8、过程违例{m['process_violations']}/8、拒绝{m['abstentions']}/8、超时{m['deadline_misses']}/8；固定60安全完成 **{b['safe_successes']}/8**、任务成功{b['successes']}/8、过程违例{b['process_violations']}/8。两者平均故障后总耗时分别为 **{m['mean_total_post_fault_seconds']:.3f}秒、{b['mean_total_post_fault_seconds']:.3f}秒**。采图、排队、传输、推理、超时保持及拒绝后的最终保持全部纳入物理计时与保护审计，逐条满足总耗时恒等式。此为小规模仿真独立试验。

结果目录：`{root}`。最终模型为`mixed512/adapter`，分类与候选选择详见`assessment.json`，独立实测详见`closed-loop/report.json`，完整决策记录见`decision.json`。

'''
route=Path('/home/lc/AuditEmbodied/AuditEmbodied 闭环优先研究路线.md')
original=route.read_text();marker='## 2026-11-09 第三十七轮：'
if marker in original:raise ValueError('round already documented')
anchor='Oct 7, 2026 · @hiker-jing\n\n'
route.write_text(original.replace(anchor,anchor+text,1))
week=Path('/home/lc/AuditEmbodied/outputs/l4-duration-pilot-20261108/week-plan.json')
w=read(week);w['as_of']='2026-11-09';w['latest_diagnostic']=str(root/'decision.json')
week.write_text(json.dumps(w,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'round_complete':True,'decision':str(root/'decision.json')}),flush=True)
