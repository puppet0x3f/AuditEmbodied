"""AgentChord graph runtime adapted to the frozen robot-only lift interface.

The upstream scheduler is reused; offline authoring is a fixed task template.
No upstream LLM authoring or EmbodiChain atomic action is claimed as reproduced.
"""
from __future__ import annotations

from .agentchord_graph import AgentTaskGraph
from .l4_offline_selection import features
from .l4_selector import ProprioRule


UPSTREAM_COMMIT = '74fe7dd756964b737952d26e9e65d4d5e2e29ec3'


def graph_spec(rule):
    """Compile the task template before accessing the held-out cohort."""
    edges = [{'id': 'task', 'source': 'start', 'target': 'goal', 'skill': 'continue',
              'monitors': ['neither_loaded', 'left_missing', 'right_missing', 'height_mismatch']}]
    branches = []
    for side, skill, index in [('left', 'R_left', 1), ('right', 'R_right', 2), ('global', 'R_global', 0)]:
        repair, check = f'{side}_repair', f'{side}_check'
        edges += [
            {'id': repair, 'source': 'start', 'target': side, 'skill': skill,
             'monitors': [f'{side}_keeper_missing'] if side != 'global' else []},
            {'id': check, 'source': side, 'target': 'goal', 'skill': 'continue',
             'monitors': ['incomplete']},
            {'id': f'{side}_finish_global', 'source': side, 'target': side,
             'skill': 'R_global', 'monitors': []}]
        branches += [{'edge': 'task', 'monitor': index, 'recovery': [repair, check]},
                     {'edge': check, 'monitor': 0, 'recovery': [f'{side}_finish_global']}]
        if side != 'global':
            fallback = f'{side}_interrupted_global'
            edges.append({'id': fallback, 'source': 'start', 'target': side,
                          'skill': 'R_global', 'monitors': []})
            branches.append({'edge': repair, 'monitor': 0, 'recovery': [fallback]})
    branches.append({'edge': 'task', 'monitor': 3, 'recovery': ['right_repair', 'right_check']})
    return {'upstream_commit': UPSTREAM_COMMIT, 'authoring': 'fixed task template; no LLM calls',
            'start': 'start', 'goal': 'goal', 'max_transitions': 8, 'rule': rule,
            'nodes': ['start', 'left', 'right', 'global', 'goal'],
            'edges': edges, 'branches': branches}


class AgentChordStyle:
    def __init__(self, spec):
        self.spec = spec
        self.rule = ProprioRule(**spec['rule'])
        self.graph = AgentTaskGraph(spec['start'], spec['goal'], spec['max_transitions'])
        for node in spec['nodes']:
            self.graph.add_node(node)
        for edge in spec['edges']:
            self.graph.add_edge(edge['id'], edge['source'], edge['target'],
                left_arm_action=edge['skill'], monitor_sequences=edge['monitors'],
                is_recovery=edge['id'] != 'task')
        for branch in spec['branches']:
            self.graph.add_recovery(branch['edge'], branch['monitor'], branch['recovery'])

    def monitor(self, name, port):
        x = features(port, self.rule)
        left, right = bool(x[2]), bool(x[3])
        height_bad = x[6] > self.rule.height_difference_limit
        return {'neither_loaded': not left and not right,
                'left_missing': not left and right,
                'right_missing': left and not right,
                'height_mismatch': left and right and height_bad,
                'left_keeper_missing': not right,
                'right_keeper_missing': not left,
                'incomplete': not (left and right) or height_bad}[name]

    def first_trigger(self, names, port):
        return next((i for i, name in enumerate(names) if self.monitor(name, port)), None)

    def __call__(self, port):
        names = self.spec['edges'][0]['monitors']
        index = self.first_trigger(names, port)
        candidate = {None: 'continue', 0: 'R_global', 1: 'R_left', 2: 'R_right', 3: 'R_right'}[index]
        return {'candidate': candidate, 'monitor_index': index,
                'reason': names[index] if index is not None else 'no_failure_monitor',
                'observation_keys': sorted(port)}

    def execute(self, execute_skill):
        events = []

        def drive(*, left_arm_action, right_arm_action, monitor_sequences, **kwargs):
            if right_arm_action is not None:
                raise ValueError('a registered skill already controls both arms')
            names = monitor_sequences or []
            result = execute_skill(left_arm_action,
                (lambda port: self.first_trigger(names, port)) if names else None)
            events.append({'skill': left_arm_action, 'monitors': names,
                           'monitor_index': result['monitor_index']})
            return result

        completed = True
        try:
            self.graph.run(drive=drive)
        except RuntimeError as error:
            if str(error) != 'Agent task graph exceeded max_transitions.':
                raise
            completed = False
        return {'completed': completed, 'transitions': len(events), 'events': events,
                'max_transitions': self.spec['max_transitions'],
                'executed_skills': [e['skill'] for e in events if e['skill'] != 'continue']}
