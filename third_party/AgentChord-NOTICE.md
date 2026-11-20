# AgentChord runtime attribution

`auditembodied/agentchord_graph.py` derives from EDEM-AI/AgentChord,
`embodichain/agents/agentchord/agent_graph.py`, commit
`74fe7dd756964b737952d26e9e65d4d5e2e29ec3` (Apache-2.0).

Source: https://github.com/EDEM-AI/AgentChord/blob/74fe7dd756964b737952d26e9e65d4d5e2e29ec3/embodichain/agents/agentchord/agent_graph.py

Copyright (c) 2021-2026 DexForce Technology Co., Ltd.

Local change: the EmbodiChain `drive` import is replaced by an explicit keyword
argument to `AgentTaskGraph.run`. Graph transitions, monitor branch dispatch,
continuation stacks, and the transition limit retain the upstream implementation.
The task graph and robot-observation monitors are supplied by the local adapter;
the upstream LLM authoring agents and EmbodiChain atomic actions are not run.
