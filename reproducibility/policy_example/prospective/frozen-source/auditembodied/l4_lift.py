"""Physical TwoArmLift initialization and forced recovery candidates for L4."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
import multiprocessing
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from .l0 import L0Config, TwoArmL0Env, _import_robosuite, _jsonable
from .recovery_skills import actual_eef_rotations, actual_gripper_positions
from .continuous import ContinuousActionExpert


@dataclass(frozen=True)
class LiftConfig:
    lift_height: float = .16
    minimum_clearance: float = .10
    maximum_tilt_deg: float = 15.
    maximum_handle_height_difference: float = .03
    relative_speed_limit: float = .2
    close_steps: int = 10
    lift_steps: int = 50
    hold_steps: int = 20
    grasp_height_offset: float = -.0036
    initial_finger_gap: float = .011
    fault_open_steps: int = 12
    fault_withdraw_steps: int = 12
    fault_withdrawal: float = .04
    coupled_height_offset: float = .04
    coupled_steps: int = 20
    local_approach_steps: int = 16
    local_close_steps: int = 22
    table_lower_steps: int = 60
    table_press_offset: float = .004
    gripper_force_limit: float = 40.
    gripper_position_gain: float = 4000.
    local_search_angles_deg: tuple[float, ...] = (12., 24., 36.)
    local_relevel_steps: int = 40


class TwoArmLiftEnv:
    """Reuse robot IK and actuator setup only at the initialization boundary."""

    sim = TwoArmL0Env.sim
    action_dim = TwoArmL0Env.action_dim
    _eef_pos = TwoArmL0Env._eef_pos
    _move_arm_to_pose = TwoArmL0Env._move_arm_to_pose
    _sync_after_reset = TwoArmL0Env._sync_after_reset
    _sync_after_kinematic_edit = TwoArmL0Env._sync_after_kinematic_edit
    sync_gripper_controller = TwoArmL0Env.sync_gripper_controller
    close = TwoArmL0Env.close

    def __init__(self, seed, config=None):
        self.settings = config or LiftConfig()
        self.config = L0Config(render=False, horizon=2000, expert_mode="continuous")
        make, load = _import_robosuite(False)
        controllers = [copy.deepcopy(load(robot="Panda")) for _ in range(2)]
        for controller in controllers:
            controller["body_parts"]["right"]["input_ref_frame"] = "world"
        self.raw = make("TwoArmLift", robots=["Panda","Panda"], env_configuration="opposed",
            controller_configs=controllers, use_object_obs=False, use_camera_obs=False,
            has_renderer=False, has_offscreen_renderer=False, control_freq=20, lite_physics=False,
            horizon=2000, seed=int(seed))
        self.obs = {}

    def reset(self):
        self.obs = self.raw.reset()
        # New-task actuator settings are fixed before grasp initialization.
        # The native pot's mass, friction and collision model are unchanged.
        for robot in self.raw.robots:
            ids=np.asarray(robot._ref_joint_gripper_actuator_indexes["right"])
            self.sim.model.actuator_forcerange[ids]=[-self.settings.gripper_force_limit,self.settings.gripper_force_limit]
            self.sim.model.actuator_gainprm[ids,0]=self.settings.gripper_position_gain
            self.sim.model.actuator_biasprm[ids,1]=-self.settings.gripper_position_gain
        self._sync_after_reset()

    def bootstrap(self):
        """Keep native sampled pot placement; place open robots then close physically."""
        settings = self.settings
        rotation = np.asarray(self.sim.data.body_xmat[self.raw.pot_body_id]).reshape(3,3)
        yaw = float(np.arctan2(rotation[1,0],rotation[0,0]))
        angle = yaw-np.pi/2
        c,s = np.cos(angle),np.sin(angle)
        rz = np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        targets = np.stack([self.raw._handle0_xpos.copy(),self.raw._handle1_xpos.copy()])
        targets[:,2] += settings.grasp_height_offset
        for i, base in enumerate((np.diag([-1.,1.,-1.]),np.diag([1.,-1.,-1.]))):
            self._move_arm_to_pose(i,targets[i],rz@base)
            robot = self.raw.robots[i]
            self.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]] = [settings.initial_finger_gap,-settings.initial_finger_gap]
        self._sync_after_kinematic_edit()
        self.table_targets = np.stack([self._eef_pos(i) for i in range(2)])
        self.rotation_targets = actual_eef_rotations(self)
        for i in range(2):self.sync_gripper_controller(i,closed=True)
        self.bootstrap_info = {"native_pot_placement_preserved":True,"registered_table_robot_targets":self.table_targets.tolist(),
            "contact_anchor_steps":0,"post_close_state_edits":0}

    def step(self, action):
        self.obs,_,_,_ = self.raw.step(np.asarray(action))

    def action(self, targets, grippers=(1.,1.)):
        from robosuite.utils import transform_utils as transforms
        action = ContinuousActionExpert.action_towards(self,targets,grippers,gain=.8)
        for i,current in enumerate(actual_eef_rotations(self)):
            error = transforms.quat2axisangle(transforms.mat2quat(self.rotation_targets[i]@current.T))
            action[i*7+3:i*7+6] = np.clip(error/.5,-1,1)
        return action


def evaluate(env):
    cfg = env.settings
    raw = env.raw
    holds = [bool(raw._check_grasp(gripper=raw.robots[i].gripper["right"],
                 object_geoms=raw.pot.handle0_geoms if i==0 else raw.pot.handle1_geoms)) for i in range(2)]
    center = np.asarray(env.sim.data.site_xpos[raw.pot_center_id])
    bottom = float(center[2]-raw.pot.top_offset[2])
    table = float(env.sim.data.site_xpos[raw.table_top_id][2])
    matrix = np.asarray(env.sim.data.body_xmat[raw.pot_body_id]).reshape(3,3)
    tilt = float(np.degrees(np.arccos(np.clip(matrix[2,2],-1,1))))
    handles = np.stack([raw._handle0_xpos.copy(),raw._handle1_xpos.copy()])
    speeds = []
    for i,site in enumerate((raw.handle0_site_id,raw.handle1_site_id)):
        a = env.sim.model.site_id2name(site)
        b = env.sim.model.site_id2name(raw.robots[i].eef_site_id["right"])
        speeds.append(float(np.linalg.norm(env.sim.data.get_site_xvelp(a)-env.sim.data.get_site_xvelp(b))))
    supports = [holds[i] and speeds[i]<=cfg.relative_speed_limit for i in range(2)]
    contact = False
    geoms = set(raw.pot.contact_geoms)
    for item in env.sim.data.contact[:env.sim.data.ncon]:
        a,b=(env.sim.model.geom_id2name(int(g)) for g in (item.geom1,item.geom2))
        if (a in geoms and b.startswith("table")) or (b in geoms and a.startswith("table")):contact=True
    difference = float(abs(handles[0,2]-handles[1,2]))
    ready = all(supports) and bottom-table>=cfg.minimum_clearance and tilt<=cfg.maximum_tilt_deg and difference<=cfg.maximum_handle_height_difference
    return {"holds":holds,"supports":supports,"relative_speeds":speeds,"clearance":bottom-table,
            "tilt_deg":tilt,"handle_height_difference":difference,"table_contact":contact,"ready":bool(ready),
            "pot_center":center.tolist(),"handle_positions":handles.tolist()}


SCENARIOS = ("left_loss","right_loss","coupled_height")
CANDIDATES = ("continue","R_left","R_right","R_global")


def snapshot(env):
    return {**{k:np.asarray(getattr(env.sim.data,k)).copy().tolist() for k in ("qpos","qvel","ctrl","act")},
            "time":float(env.sim.data.time)}


def wait_control_steps(seconds, control_timestep=.05):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("decision latency must be finite and nonnegative")
    return math.ceil(seconds/control_timestep)


def run_trial(seed, scenario="nominal", candidate="continue", config=None, selector=None, account_selector_latency=False,
              duration_selector=None, fixed_selector_wait_steps=None, refresh_duration_observation=False,
              fixed_duration_wait_steps=None,account_prefix_observation_latency=False,
              fixed_prefix_observation_wait_steps=None,duration_lookahead_steps=None,pace_recovery=False,
              recovery_graph=None,regrasp_hook=None,global_executor=None):
    import mujoco
    if fixed_selector_wait_steps is not None:
        if (not isinstance(fixed_selector_wait_steps,int) or fixed_selector_wait_steps<0 or
            selector is None or account_selector_latency):
            raise ValueError('fixed wait requires nonnegative integer steps, a selector and no measured-latency replay')
    if fixed_duration_wait_steps is not None and (not isinstance(fixed_duration_wait_steps,int) or
            fixed_duration_wait_steps<0 or not refresh_duration_observation or duration_selector is None):
        raise ValueError('post-prefix wait needs refreshed duration observations and nonnegative integer steps')
    if fixed_prefix_observation_wait_steps is not None and (not isinstance(fixed_prefix_observation_wait_steps,int)
            or fixed_prefix_observation_wait_steps<0 or account_prefix_observation_latency):
        raise ValueError('prefix observation wait must be nonnegative and cannot also be measured')
    if duration_lookahead_steps is not None and (not isinstance(duration_lookahead_steps,int) or
            duration_lookahead_steps<=0 or duration_selector is None or not refresh_duration_observation or
            fixed_duration_wait_steps is not None):
        raise ValueError('lookahead requires a positive step count, refreshed observations and no post-prefix budget hold')
    cfg = config or LiftConfig()
    if scenario not in ("nominal","sham",*SCENARIOS) or candidate not in CANDIDATES:raise ValueError((scenario,candidate))
    env = TwoArmLiftEnv(seed,cfg)
    row = {"seed":seed,"scenario":scenario,"candidate":candidate,"success":False,"failure_reason":None,
           "selector_wall_seconds":0.,"inference_wait_steps":0,"inference_wait_seconds":0.}
    trace,blocked=[],[]
    category,stage="initialization","reset"
    previous=None;path=0.;flight=0;max_flight=0;collisions=0;keeper_lost=0
    warning_before=None;recovery_start=None
    lookahead_hook=None
    active_monitor=None

    class MonitorTriggered(Exception):
        def __init__(self, index):
            self.index=index

    def robot_observation():
        return {"positions":np.stack([env._eef_pos(i) for i in range(2)]).tolist(),
                "fingers":actual_gripper_positions(env).tolist(),"reference_positions":registered.tolist()}

    def check_monitor():
        if active_monitor is None:return
        started=time.perf_counter()
        index=active_monitor(copy.deepcopy(robot_observation()))
        elapsed=time.perf_counter()-started
        row['monitor_calls']=row.get('monitor_calls',0)+1
        row['monitor_wall_seconds']=row.get('monitor_wall_seconds',0.)+elapsed
        row['max_monitor_wall_seconds']=max(row.get('max_monitor_wall_seconds',0.),elapsed)
        # A synchronous feedback callback has the same 50 ms control-period
        # budget as the shared controller. Replay any excess as physical holds.
        extra_steps=wait_control_steps(max(0.,elapsed-.05))
        for _ in range(extra_steps):
            action=np.asarray(trace[-1]['robot']['action'])
            env.step(action);capture('monitor_overrun_wait',action)
        if extra_steps:
            extra=extra_steps/20
            row['monitor_wait_seconds']=row.get('monitor_wait_seconds',0.)+extra
            row['inference_wait_seconds']+=extra
            row['inference_wait_steps']+=extra_steps
            row['continuation_wait_seconds']+=extra
        if index is not None:raise MonitorTriggered(index)

    def capture(name,action):
        nonlocal stage,previous,path,flight,max_flight,collisions,keeper_lost
        stage=name
        warning={k:int(env.sim.data.warning[int(getattr(mujoco.mjtWarning,k))].number)-warning_before[k]
                 for k in ("mjWARN_BADQPOS","mjWARN_BADQVEL","mjWARN_BADQACC","mjWARN_BADCTRL")}
        if any(warning.values()) or (trace and float(env.sim.data.time)<=trace[-1]["time"]):
            row["bad_state_warnings"]=warning
            raise FloatingPointError(f"numerical instability during {name}")
        positions=np.stack([env._eef_pos(i) for i in range(2)])
        if category=="recovery" and previous is not None:path+=float(np.linalg.norm(positions-previous,axis=1).sum())
        previous=positions
        truth=evaluate(env)
        if category in ("recovery","final_hold"):
            flight=flight+1 if not (any(truth["holds"]) or truth["table_contact"]) else 0
            max_flight=max(max_flight,flight)
            if category=="recovery" and candidate in ("R_left","R_right"):
                keeper=1 if candidate=="R_left" else 0
                keeper_lost+=int(not truth["holds"][keeper])
            for contact in env.sim.data.contact[:env.sim.data.ncon]:
                a,b=(env.sim.model.geom_id2name(int(g)) for g in (contact.geom1,contact.geom2))
                if contact.dist<-.01 and ((a.startswith("robot0") and b.startswith("robot1")) or (a.startswith("robot1") and b.startswith("robot0"))):
                    collisions+=1;break
        trace.append({"time":float(env.sim.data.time),"stage":name,"phase":category,
            "robot":{"positions":positions.tolist(),"rotations":actual_eef_rotations(env).tolist(),
                     "fingers":actual_gripper_positions(env).tolist(),"action":np.asarray(action).tolist()},"evaluator":truth})
        if recovery_graph is not None and category=='recovery':trace[-1]['skill']=candidate

    def phase(name,targets,steps,grippers=(1.,1.)):
        env.record_stage=name
        env.record_phase=category
        for step in range(steps):
            control_started=time.perf_counter()
            action=env.action(targets,grippers);env.step(action);capture(name,action)
            if category=='recovery':check_monitor()
            if lookahead_hook is not None and name=='local_arc_close' and steps-step-1==duration_lookahead_steps:
                lookahead_hook(name,steps-step-1)
            if pace_recovery and category=='recovery':
                elapsed=time.perf_counter()-control_started
                row['max_recovery_control_wall_seconds']=max(row.get('max_recovery_control_wall_seconds',0.),elapsed)
                time.sleep(max(0.,.05-elapsed))

    def move(name,targets,steps,grippers=(1.,1.)):
        start=np.stack([env._eef_pos(i) for i in range(2)])
        for j in range(steps):phase(name,start+(np.asarray(targets)-start)*(j+1)/steps,1,grippers)

    def forbidden(name):
        def reject(*args,**kwargs):
            blocked.append({"stage":stage,"method":name})
            raise AssertionError(f"post-initialization edit or anchor forbidden: {name}")
        return reject

    try:
        env.reset()
        warning_before={k:int(env.sim.data.warning[int(getattr(mujoco.mjtWarning,k))].number)
                        for k in ("mjWARN_BADQPOS","mjWARN_BADQVEL","mjWARN_BADQACC","mjWARN_BADCTRL")}
        row["initialization_state"]=snapshot(env)
        env.bootstrap()
        row["preclose_state"]=snapshot(env)
        with ExitStack() as guards:
            # Start before physical closing. No subsequent pose edit is allowed.
            for name in ("reset","bootstrap","_move_arm_to_pose","_sync_after_reset","_sync_after_kinematic_edit","sync_gripper_controller"):
                guards.enter_context(patch.object(env,name,side_effect=forbidden(name)))
            guards.enter_context(patch.object(env.raw,"reset",side_effect=forbidden("raw.reset")))
            guards.enter_context(patch.object(ContinuousActionExpert,"_set_contact_state",side_effect=forbidden("_set_contact_state")))
            phase("initial_close",env.table_targets,cfg.close_steps)
            row["post_close"]=evaluate(env)
            table_targets=np.stack([env._eef_pos(i) for i in range(2)])
            goal=table_targets.copy();goal[:,2]+=cfg.lift_height
            move("nominal_lift",goal,cfg.lift_steps)
            nominal_hold_start=float(env.sim.data.time)
            phase("nominal_stable_hold",goal,cfg.hold_steps)
            row["nominal_hold_seconds"]=float(env.sim.data.time)-nominal_hold_start
            nominal_frames=[f["evaluator"] for f in trace if f["stage"]=="nominal_stable_hold"]
            row["nominal_ready"]=all(f["ready"] for f in nominal_frames) and row["nominal_hold_seconds"]>=1.-1e-9
            row["initialization_end_state"]=snapshot(env)
            row["bootstrap"]=env.bootstrap_info
            if not row["nominal_ready"]:
                row["failure_reason"]="nominal_lift_or_hold_failed";return row
            registered=np.stack([env._eef_pos(i) for i in range(2)])
            rotations=actual_eef_rotations(env)
            registered_rotation_targets=env.rotation_targets.copy()
            if scenario=="nominal":
                row["success"]=True;row["final"]=evaluate(env);return row
            category="fault"
            targets=registered.copy()
            if scenario in ("left_loss","right_loss"):
                arm=0 if scenario=="left_loss" else 1
                grips=[1.,1.];grips[arm]=-1.
                phase("fault_open",targets,cfg.fault_open_steps,grips)
                targets[arm,2]+=cfg.fault_withdrawal
                move("fault_withdraw",targets,cfg.fault_withdraw_steps,grips)
            elif scenario=="sham":
                phase("fault_hold",targets,cfg.fault_open_steps+cfg.fault_withdraw_steps)
            else:
                targets[0,2]+=cfg.coupled_height_offset;targets[1,2]-=cfg.coupled_height_offset
                move("fault_differential_height",targets,cfg.coupled_steps)
                phase("fault_settle",targets,4)
            row["post_fault"]=evaluate(env);row["prefix_state"]=snapshot(env)
            h=row["post_fault"]["holds"]
            row["fault_class_valid"]=bool((not h[0] and h[1]) if scenario=="left_loss" else
                (h[0] and not h[1]) if scenario=="right_loss" else
                all(h) and row["post_fault"]["ready"] if scenario=="sham" else all(h) and not row["post_fault"]["ready"])
            category="recovery";recovery_start=float(env.sim.data.time)
            waiting_targets=targets.copy()
            waiting_grippers=tuple(trace[-1]["robot"]["action"][i] for i in (6,13))
            parallel_local=False
            if selector is not None:
                # The policy gets robot observations only, never env, phase or truth.
                observation={"positions":np.stack([env._eef_pos(i) for i in range(2)]).tolist(),
                    "fingers":actual_gripper_positions(env).tolist(),"reference_positions":registered.tolist()}
                decision_started=time.perf_counter()
                decision=selector(copy.deepcopy(observation))
                row["selector_wall_seconds"]=time.perf_counter()-decision_started
                candidate=decision["candidate"]
                if candidate not in (*CANDIDATES,"abstain"):raise ValueError(f"invalid selected candidate: {candidate}")
                if candidate in ("R_left","R_right") and "local_relevel_steps" in decision:
                    n=decision["local_relevel_steps"]
                    if n not in (30,40,60):raise ValueError(f"unregistered relevel duration: {n}")
                    cfg=replace(cfg,local_relevel_steps=n)
                    env.settings=cfg
                row["candidate"]=candidate
                row["selection"]={"input":observation,"decision":decision,"rgb_frames":0,"robot_observations":1}
                parallel_local=bool(decision.get('parallel_local'))
                if parallel_local and (duration_selector is None or candidate not in ('R_left','R_right')):
                    raise ValueError('parallel local prefix requires a duration selector and local role')
                if account_selector_latency or fixed_selector_wait_steps is not None:
                    row['wait_source']='fixed_intervention' if fixed_selector_wait_steps is not None else 'measured_pipeline'
                    row["inference_wait_steps"]=(fixed_selector_wait_steps if fixed_selector_wait_steps is not None
                                                  else wait_control_steps(row["selector_wall_seconds"]))
                    wait_started=float(env.sim.data.time)
                    phase("inference_wait",waiting_targets,row["inference_wait_steps"],waiting_grippers)
                    row["inference_wait_seconds"]=float(env.sim.data.time)-wait_started
            row["post_wait_state"]=snapshot(env)
            row["post_wait_evaluator"]=evaluate(env)
            row['pre_skill_protection']={'max_unsupported_seconds':max_flight/20,
                'deep_interarm_contact_steps':collisions,'keeper_grasp_lost_steps':keeper_lost}
            skill_started=float(env.sim.data.time)
            row['continuation_wait_seconds']=0.
            def measured_duration_observation():
                obs={"positions":np.stack([env._eef_pos(i) for i in range(2)]).tolist(),
                    "fingers":actual_gripper_positions(env).tolist(),"reference_positions":registered.tolist()}
                if refresh_duration_observation:
                    frames=[f for f in trace if f['phase']=='recovery']
                    indices=np.linspace(0,len(frames)-1,min(8,len(frames)),dtype=int) if frames else []
                    obs['observed_history']=[{'seconds_before_observation':round(float(env.sim.data.time)-frames[i]['time'],6),
                        'stage':frames[i]['stage'],
                        'positions':np.round(frames[i]['robot']['positions'],6).tolist(),
                        'fingers':np.round(frames[i]['robot']['fingers'],6).tolist()} for i in indices]
                return obs
            if parallel_local:
                observation=measured_duration_observation()
                row['duration_selection']={'input':{k:v for k,v in observation.items() if k!='observed_history'},'observation_state':snapshot(env)}
                observation_started=time.perf_counter()
                duration_selector('start',copy.deepcopy(observation),candidate,0.)
                observation_wall=time.perf_counter()-observation_started
                if account_prefix_observation_latency or fixed_prefix_observation_wait_steps is not None:
                    steps=(fixed_prefix_observation_wait_steps if fixed_prefix_observation_wait_steps is not None
                        else wait_control_steps(observation_wall))
                    row['duration_selection'].update(prefix_observation_wall_seconds=observation_wall,
                        prefix_observation_wait_steps=steps)
                    wait_start=float(env.sim.data.time)
                    phase('prefix_observation_wait',np.stack([env._eef_pos(i) for i in range(2)]),steps,waiting_grippers)
                    extra=float(env.sim.data.time)-wait_start
                    row['continuation_wait_seconds']+=extra
                    row['inference_wait_seconds']+=extra
                    row['inference_wait_steps']+=steps
            duration_consumed=False
            if parallel_local and duration_lookahead_steps is not None:
                if duration_lookahead_steps>=cfg.local_close_steps:raise ValueError('lookahead must fit the closing phase')
                row['duration_selection']['lookahead_requests']=[]
                def lookahead_hook(name,remaining):
                    if duration_consumed:return
                    observation=measured_duration_observation()
                    recent=[f for f in trace if f['phase']=='recovery'][-5:]
                    previous_robot=recent[0]['robot']
                    attempt=len(row['duration_selection']['lookahead_requests'])+1
                    observation.update(attempt_id=attempt,stage=name,remaining_close_steps=remaining,
                        seconds_after_h0=float(env.sim.data.time)-row['prefix_state']['time'],
                        previous_robot={k:previous_robot[k] for k in ('positions','fingers')},
                        delta_seconds=float(env.sim.data.time)-recent[0]['time'],
                        prefix_observation_wait_steps=row['duration_selection'].get('prefix_observation_wait_steps',0))
                    observation['previous_robot']['reference_positions']=registered.tolist()
                    issued=float(env.sim.data.time)
                    duration_selector('lookahead',copy.deepcopy(observation),candidate,remaining/20)
                    row['duration_selection']['lookahead_requests'].append({'attempt_id':attempt,
                        'issued_sim_time':issued,'remaining_close_steps':remaining,'input':observation})

            def consume_duration():
                nonlocal cfg,duration_consumed
                if not parallel_local or duration_consumed:return True
                prefix_seconds=float(env.sim.data.time)-skill_started
                boundary=measured_duration_observation() if refresh_duration_observation else None
                if boundary is not None and 'prefix_observation_wait_steps' in row['duration_selection']:
                    boundary['prefix_observation_wait_steps']=row['duration_selection']['prefix_observation_wait_steps']
                if boundary is not None:
                    row['duration_selection'].update(boundary_input={k:v for k,v in boundary.items() if k!='observed_history'},
                        boundary_state=snapshot(env),prefix_protection={'max_unsupported_seconds':max_flight/20,
                        'keeper_grasp_lost_steps':keeper_lost,'deep_interarm_contact_steps':collisions})
                if duration_lookahead_steps is not None:
                    requests=row['duration_selection']['lookahead_requests']
                    boundary['attempt_id']=requests[-1]['attempt_id'] if requests else None
                    result=duration_selector('finish',copy.deepcopy(boundary),candidate,prefix_seconds)
                    duration_consumed=True
                    if result['candidate']!=candidate:raise ValueError('lookahead changed repair role')
                    ready=bool(result.get('forecast_ready'))
                    same_attempt=result.get('attempt_id')==boundary['attempt_id']
                    fallback=not ready or not same_attempt
                    row['duration_selection'].update(decision=result,prefix_seconds=prefix_seconds,
                        forecast_origin='before_close_end',forecast_ready=ready and same_attempt,
                        deadline_missed=bool(requests and not ready),residual_wait_steps=0,
                        baseline_fallback=fallback or bool(result.get('baseline_fallback')),
                        pipeline_wall_seconds=result.get('pipeline_wall_seconds'))
                    if not fallback and not result['accepted']:
                        row['abstained']=True;row['abstention_reason']='no_acceptable_duration'
                        row['local_steps']['local_relevel_steps']=None
                        return False
                    n=60 if fallback else result['local_relevel_steps']
                    if n not in (30,40,60):raise ValueError('unregistered lookahead duration')
                    cfg=replace(cfg,local_relevel_steps=n);env.settings=cfg
                    row['local_steps']['local_relevel_steps']=n
                    return True
                result=duration_selector('finish',copy.deepcopy(boundary),candidate,prefix_seconds)
                duration_consumed=True
                if result['candidate']!=candidate:raise ValueError('duration selector changed repair role')
                elapsed=result['pipeline_wall_seconds']
                planned=0 if fixed_duration_wait_steps is None else fixed_duration_wait_steps
                overlap=prefix_seconds if fixed_duration_wait_steps is None else planned/20
                residual=max(0.,elapsed-overlap)
                steps=wait_control_steps(residual)
                row['duration_selection'].update(decision=result,prefix_seconds=prefix_seconds,
                    pipeline_wall_seconds=elapsed,overlapped_seconds=min(elapsed,overlap),
                    residual_wait_steps=steps,deadline_missed=bool(steps))
                if fixed_duration_wait_steps is not None:
                    row['duration_selection'].update(planned_wait_steps=planned,forecast_origin='post_prefix')
                if planned:
                    wait_start=float(env.sim.data.time)
                    phase('duration_budget_wait',np.stack([env._eef_pos(i) for i in range(2)]),planned)
                    extra=float(env.sim.data.time)-wait_start
                    row['continuation_wait_seconds']+=extra
                    row['inference_wait_seconds']+=extra
                    row['inference_wait_steps']+=planned
                if steps:
                    wait_start=float(env.sim.data.time)
                    phase('duration_deadline_wait',np.stack([env._eef_pos(i) for i in range(2)]),steps)
                    extra=float(env.sim.data.time)-wait_start
                    row['continuation_wait_seconds']+=extra
                    row['inference_wait_seconds']+=extra
                    row['inference_wait_steps']+=steps
                if steps or not result['accepted']:
                    row['abstained']=True
                    row['abstention_reason']='duration_deadline_missed' if steps else 'no_acceptable_duration'
                    row['local_steps']['local_relevel_steps']=None
                    return False
                n=result['local_relevel_steps']
                if n not in (30,40,60):raise ValueError(f'unregistered duration: {n}')
                cfg=replace(cfg,local_relevel_steps=n);env.settings=cfg
                row['local_steps']['local_relevel_steps']=n
                return True

            def execute_skill(skill, monitor=None):
                nonlocal candidate,active_monitor
                if skill!='continue' or not row.get('skill_events'):
                    candidate=skill
                    row['candidate']=skill
                active_monitor=monitor
                event=None
                if recovery_graph is not None:
                    event={'candidate':skill,'start_time':float(env.sim.data.time),
                           'input':robot_observation(),'start_evaluator':evaluate(env)}
                    row.setdefault('skill_events',[]).append(event)
                try:
                    check_monitor()
                    if skill=='continue':return {'monitor_index':None,'actions':[]}
                    if candidate in ("R_left","R_right"):
                        row["local_steps"]={k:getattr(cfg,k) for k in ("local_approach_steps","local_close_steps","local_relevel_steps")}
                        arm=0 if candidate=="R_left" else 1;keeper=1-arm
                        targets=np.stack([env._eef_pos(i) for i in range(2)])
                        delta=actual_eef_rotations(env)[keeper]@rotations[keeper].T
                        span=delta@(registered[arm]-registered[keeper])
                        aligned=targets.copy();aligned[arm]=targets[keeper]+span

                        def fingers_loaded():
                            # Encoder evidence only; actual contact is evaluated separately.
                            width=float(np.mean(np.abs(actual_gripper_positions(env)[arm])))
                            return .004<width<.028

                        if fingers_loaded():
                            # Do not discard an existing grasp just to correct arm balance.
                            move("local_closed_realign",aligned,cfg.local_approach_steps)
                            phase("local_closed_settle",aligned,cfg.local_close_steps)
                            consume_duration()
                        else:
                            # A one-sided load can rotate the pot within its remaining grip.
                            # Probe a fixed gravity arc around the measured keeper pose.
                            axis=np.cross(span,np.array([0.,0.,-1.]))
                            axis/=np.linalg.norm(axis)
                            x,y,z=axis
                            skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
                            def gravity_rotation(angle):
                                return np.eye(3)+np.sin(angle)*skew+(1.-np.cos(angle))*(skew@skew)
                            grips=[1.,1.];grips[arm]=-1.
                            for degrees in cfg.local_search_angles_deg:
                                angle=np.radians(degrees)
                                rotation=gravity_rotation(angle)
                                targets[arm]=targets[keeper]+rotation@span
                                env.rotation_targets[arm]=rotation@delta@rotations[arm]
                                move("local_arc_approach",targets,cfg.local_approach_steps,grips)
                                phase("local_arc_close",targets,cfg.local_close_steps)
                                if fingers_loaded():
                                    if not consume_duration():break
                                    targets=np.stack([env._eef_pos(i) for i in range(2)])
                                    current_span=targets[arm]-targets[keeper]
                                    horizontal=current_span.copy();horizontal[2]=0.
                                    tilt=np.arctan2(-current_span[2],np.linalg.norm(horizontal))
                                    horizontal*=np.linalg.norm(current_span)/np.linalg.norm(horizontal)
                                    # Rotate the grasp pair along an arc without stretching it.
                                    for j in range(cfg.local_relevel_steps):
                                        angle=tilt*(1.-(j+1)/cfg.local_relevel_steps)
                                        targets[arm]=targets[keeper]+gravity_rotation(angle)@horizontal
                                        env.rotation_targets[arm]=gravity_rotation(angle)@delta@rotations[arm]
                                        phase("local_relevel",targets,1)
                                    phase("local_relevel_settle",targets,8)
                                    break
                            consume_duration()
                    elif candidate=="R_global" and global_executor is not None:
                        def set_rotations(value):
                            env.rotation_targets=np.asarray(value).copy()
                        row['global_execution']=global_executor({
                            'robot':lambda:{**robot_observation(),
                                'rotations':actual_eef_rotations(env).tolist()},
                            'time':lambda:float(env.sim.data.time),
                            'move':move,'phase':phase,'set_rotations':set_rotations,
                            'observe':lambda:regrasp_hook(env,table_targets,registered_rotation_targets),
                            'table_targets':table_targets.copy(),
                            'registered':registered.copy(),
                            'registered_rotations':registered_rotation_targets.copy(),
                            'gripper_commands':tuple(trace[-1]['robot']['action'][i] for i in (6,13)),
                        })
                        extra=row['global_execution']['service_steps']
                        row['continuation_wait_seconds']+=extra/20
                        row['inference_wait_seconds']+=extra/20
                        row['inference_wait_steps']+=extra
                    elif candidate=="R_global":
                        # Restore this registered skill's controller targets after
                        # an interrupted local edge; this never edits robot state.
                        env.rotation_targets=registered_rotation_targets.copy()
                        targets=table_targets.copy();targets[:,2]-=cfg.table_press_offset
                        move("global_lower_to_table",targets,cfg.table_lower_steps)
                        phase("global_table_loaded",targets,8)
                        phase("global_release",targets,14,(-1.,-1.))
                        phase("global_table_settle",targets,6,(-1.,-1.))
                        targets=np.stack([env._eef_pos(i) for i in range(2)]);targets[:,2]+=.10
                        move("global_clear",targets,16,(-1.,-1.))
                        pickup=table_targets.copy()
                        if regrasp_hook is not None:
                            observed_at=float(env.sim.data.time)
                            callback_started=time.perf_counter()
                            result=regrasp_hook(env,table_targets,registered_rotation_targets)
                            elapsed=time.perf_counter()-callback_started
                            result.update(service_wall_seconds=elapsed,
                                service_wait_steps=max(40,wait_control_steps(elapsed)),observed_time=observed_at)
                            row['regrasp_decision']=result
                            hold=np.stack([env._eef_pos(i) for i in range(2)])
                            steps=result['service_wait_steps']
                            phase('regrasp_observation_wait',hold,steps,(-1.,-1.))
                            row['inference_wait_steps']+=steps
                            row['inference_wait_seconds']+=steps/20
                            row['continuation_wait_seconds']+=steps/20
                            if result['apply'] and result['estimate']['valid']:
                                pickup=np.asarray(result['estimate']['targets'])
                        targets=table_targets.copy();targets[:,2]+=.08
                        if regrasp_hook is not None and result['apply'] and result['estimate']['valid']:
                            targets=pickup+np.array([0.,0.,.08])
                        move("global_hover",targets,24,(-1.,-1.))
                        move("global_descend",pickup,20,(-1.,-1.))
                        phase("global_close",pickup,22)
                        env.rotation_targets=registered_rotation_targets.copy()
                        move("global_lift",registered,cfg.lift_steps)
                        phase("global_lift_settle",registered,12)
                    return {'monitor_index':None,'actions':[]}
                except MonitorTriggered as trigger:
                    if event is not None:event['monitor_index']=trigger.index
                    return {'monitor_index':trigger.index,'actions':[]}
                finally:
                    active_monitor=None
                    if event is not None:
                        event.update(end_time=float(env.sim.data.time),end_input=robot_observation())
            if recovery_graph is None:
                execute_skill(candidate)
            else:
                row['graph']=recovery_graph.execute(execute_skill)
                if not row['graph']['completed']:
                    row['abstained']=True;row['abstention_reason']='graph_transition_limit'
            row["recovery_seconds"]=float(env.sim.data.time)-recovery_start
            row["skill_seconds"]=float(env.sim.data.time)-skill_started-row['continuation_wait_seconds']
            row["recovery_path_m"]=path
            row["post_recovery"]=evaluate(env)
            table=[f["evaluator"] for f in trace if f["stage"]=="global_table_settle"]
            row["table_setdown_verified"]=bool(table and any(f["table_contact"] and not any(f["holds"]) for f in table))
            lifted=[f["evaluator"] for f in trace if f["stage"] in ("global_lift","global_lift_settle")]
            row["table_regrasp_and_lift_verified"]=bool(lifted and any(all(f["holds"]) and f["clearance"]>=cfg.minimum_clearance for f in lifted))
            row["table_cycle_verified"]=row["table_setdown_verified"] and row["table_regrasp_and_lift_verified"]
            category="final_hold"
            finish=np.stack([env._eef_pos(i) for i in range(2)])
            hold_start=float(env.sim.data.time)
            phase("final_stable_hold",finish,cfg.hold_steps,waiting_grippers if candidate=="abstain" else (1.,1.))
            row["hold_seconds"]=float(env.sim.data.time)-hold_start
            held=[f["evaluator"] for f in trace if f["stage"]=="final_stable_hold"]
            row["task_complete"]=all(f["ready"] for f in held) and row["hold_seconds"]>=1.-1e-9
            row["success"]=bool(candidate!="abstain" and not row.get('abstained',False) and row["task_complete"] and (candidate!="R_global" or row["table_cycle_verified"]))
            row["final"]=evaluate(env)
            row["total_post_fault_seconds"]=float(env.sim.data.time)-row["prefix_state"]["time"]
            row["episode_simulated_seconds"]=float(env.sim.data.time)-row["initialization_state"]["time"]
            if not row["success"]:
                row["failure_reason"]=(row['abstention_reason'] if row.get('abstained') else "no_acceptable_plan" if candidate=="abstain"
                    else "table_setdown_not_verified" if candidate=="R_global" and not row["table_setdown_verified"]
                    else "table_regrasp_not_verified" if candidate=="R_global" and not row["table_regrasp_and_lift_verified"]
                    else "missing_left_grasp" if not all(f["holds"][0] for f in held)
                    else "missing_right_grasp" if not all(f["holds"][1] for f in held)
                    else "insufficient_lift_height" if not all(f["clearance"]>=cfg.minimum_clearance for f in held)
                    else "imbalance" if not all(f["tilt_deg"]<=cfg.maximum_tilt_deg and f["handle_height_difference"]<=cfg.maximum_handle_height_difference for f in held)
                    else "relative_motion_or_hold_failed")
    except Exception as e:
        row["failure_reason"]="numerical_instability" if isinstance(e,FloatingPointError) else "forbidden_state_edit" if blocked else "initialization_or_execution_error"
        row["error"]=repr(e)
        row["failed_stage"]=stage
    finally:
        row["trace"]=trace
        row["blocked_calls"]=blocked
        row["contact_anchor_count"]=0
        row["state_edit_count_after_initialization"]=0
        row["external_wrench_at_end"]=float(np.linalg.norm(env.sim.data.xfrc_applied))
        row["protection"]={"max_unsupported_seconds":max_flight/20,"deep_interarm_contact_steps":collisions,"keeper_grasp_lost_steps":keeper_lost}
        row["protection_violation"]=bool(max_flight>2 or collisions or keeper_lost or blocked or row["failure_reason"]=="numerical_instability")
        env.close()
    return row


def nominal(seed,config=None):return run_trial(seed,"nominal","continue",config)


def _job(args):return run_trial(*args)


def summary(rows):
    return {"episodes":len(rows),"successes":sum(r["success"] for r in rows),
        "safe_successes":sum(r["success"] and not r["protection_violation"] for r in rows),
        "mean_recovery_seconds":float(np.mean([r.get("recovery_seconds",0.) for r in rows])),
        "mean_recovery_path_m":float(np.mean([r.get("recovery_path_m",0.) for r in rows])),
        "protection_violation_rate":float(np.mean([r["protection_violation"] for r in rows])),
        "numerical_instabilities":sum(r["failure_reason"]=="numerical_instability" for r in rows),
        "failure_reasons":dict(Counter(r["failure_reason"] for r in rows if not r["success"]))}


def analyze_matrix(rows):
    """Paired candidate evidence, including an offline safe-cost oracle."""
    groups={}
    for row in rows:groups.setdefault((row["seed"],row["scenario"]),{})[row["candidate"]]=row
    mismatches=[];faults={q:{"scenes":0,"valid":0,"invalid_seeds":[]} for q in SCENARIOS}
    selected=[];uncovered=[];paired=[]
    for (seed,scenario),candidates in sorted(groups.items()):
        reference=next(iter(candidates.values()))
        fault=faults[scenario];fault["scenes"]+=1
        fault["valid"]+=int(reference.get("fault_class_valid",False))
        if not reference.get("fault_class_valid",False):fault["invalid_seeds"].append(seed)
        if set(candidates)!=set(CANDIDATES):mismatches.append([seed,scenario,"incomplete_candidates"])
        for name in ("initialization_state","preclose_state","initialization_end_state","prefix_state"):
            if name not in reference or any(r.get(name)!=reference[name] for r in candidates.values()):
                mismatches.append([seed,scenario,name])
        safe=[r for r in candidates.values() if r["success"] and not r["protection_violation"]]
        if not safe:
            uncovered.append([seed,scenario]);continue
        best=min(safe,key=lambda r:(r["recovery_seconds"],r["recovery_path_m"]))
        selected.append(best)
        global_=candidates.get("R_global",{})
        if global_.get("success") and not global_["protection_violation"]:
            paired.append((best,global_))
    oracle={"kind":"offline envelope of actually executed safe candidates; not an online selector",
        "scenes":len(groups),"safe_successes":len(selected),"uncovered":uncovered,
        "selected_candidates":dict(Counter(r["candidate"] for r in selected)),
        "mean_recovery_seconds_on_safe":float(np.mean([r["recovery_seconds"] for r in selected])) if selected else None,
        "mean_recovery_path_m_on_safe":float(np.mean([r["recovery_path_m"] for r in selected])) if selected else None}
    costs={"paired_safe_scenes":len(paired)}
    for metric in ("recovery_seconds","recovery_path_m"):
        a=float(np.mean([x[metric] for x,y in paired])) if paired else None
        b=float(np.mean([y[metric] for x,y in paired])) if paired else None
        costs[metric]={"oracle":a,"fixed_global":b,"reduction_fraction":1.-a/b if b else None}
    return {"prefix_pairing":{"groups":len(groups),"all_equal":not mismatches,"mismatches":mismatches},
        "fault_class_verification":faults,"oracle":oracle,"oracle_vs_global_paired_cost":costs}


def run_batch(output,stage,seeds=10,workers=4,config=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);cfg=config or LiftConfig()
    if (output/"trials.jsonl").exists():raise ValueError("data exists; do not overwrite experiments")
    jobs=[(s,q,k,cfg) for s in range(seeds) for q in (("nominal",) if stage=="nominal" else SCENARIOS)
          for k in (("continue",) if stage=="nominal" else CANDIDATES)]
    rows=[]
    (output/"config.json").write_text(json.dumps(asdict(cfg),indent=2)+"\n")
    with (output/"trials.jsonl").open("w") as log:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(_job,jobs):
                log.write(json.dumps(_jsonable(row))+"\n");log.flush();rows.append(row)
                print(json.dumps({k:row.get(k) for k in ("seed","scenario","candidate","success","failure_reason")}),flush=True)
    report={"stage":stage,"seeds":list(range(seeds)),"overall":summary(rows),
        "by_candidate":{k:summary([r for r in rows if r["candidate"]==k]) for k in CANDIDATES if any(r["candidate"]==k for r in rows)},
        "by_scenario":{q:{k:summary([r for r in rows if r["scenario"]==q and r["candidate"]==k]) for k in CANDIDATES if any(r["scenario"]==q and r["candidate"]==k for r in rows)} for q in sorted({r["scenario"] for r in rows})},
        "contact_anchors":sum(r["contact_anchor_count"] for r in rows),"state_edits_after_initialization":sum(r["state_edit_count_after_initialization"] for r in rows),
        "forbidden_calls":sum(len(r["blocked_calls"]) for r in rows)}
    if stage=="matrix":report.update(analyze_matrix(rows))
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    fields=("seed","scenario","candidate","success","protection_violation","recovery_seconds","recovery_path_m","failure_reason")
    with (output/"cases.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in rows)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--output")
    parser.add_argument("--output-dir")
    parser.add_argument("--stage",choices=("nominal","matrix"),default="nominal")
    parser.add_argument("--seeds",type=int,default=10)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--config")
    args=parser.parse_args()
    cfg=LiftConfig(**json.loads(Path(args.config).read_text())) if args.config else LiftConfig()
    if args.output_dir:
        print(json.dumps(run_batch(args.output_dir,args.stage,args.seeds,args.workers,cfg)));return
    if not args.output:parser.error("--output or --output-dir is required")
    result=nominal(args.seed,cfg)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(_jsonable(result),indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="trace"}))


if __name__=="__main__":main()
