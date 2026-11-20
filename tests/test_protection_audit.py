from auditembodied.protection_audit import longest_run, role_signal, analyze_episode


def test_separated_interruptions_are_not_added_to_a_single_gap():
    assert longest_run([True,False,True,False,True]) == 1


def test_role_change_breaks_gap_without_erasing_previous_violation():
    assert longest_run([True,True,True,True], [0,0,1,1]) == 2
    assert longest_run([True,True,False,False], [0,0,None,None]) == 2


def test_intentional_global_release_is_not_a_local_role_loss():
    frames=[{'phase':'recovery','skill':skill,'evaluator':{'holds':holds}}
            for skill,holds in [('R_left',[True,False]),('R_global',[False,False]),('R_global',[False,False])]]
    flags,keys=role_signal({'candidate':'R_global','trace':frames})
    assert flags == [True,False,False]
    assert longest_run(flags,keys) == 1


def sample_row(keeper):
    frames=[]
    for i,holds in enumerate(keeper):
        truth={'holds':holds,'table_contact':False,'pot_center':[0.,0.,1.], 'tilt_deg':0.,'clearance':.1}
        frames.append({'phase':'recovery','stage':'local_relevel','time':(i+1)*.05,'evaluator':truth})
    unsupported=[not any(h) for h in keeper]
    return {'trace':frames,'seed':1,'method':'test','scenario':'left_loss','candidate':'R_left',
        'prefix_state':{'time':0.},'post_fault':{'holds':[False,True],'pot_center':[0.,0.,1.]},
        'success':True,'task_complete':True,'blocked_calls':[], 'failure_reason':None,
        'total_post_fault_seconds':len(frames)*.05,
        'protection':{'keeper_grasp_lost_steps':sum(not h[1] for h in keeper),
            'max_unsupported_seconds':longest_run(unsupported)*.05,'deep_interarm_contact_steps':0},
        'protection_violation':any(not h[1] for h in keeper)}


def test_one_sample_boundary_is_role_only_and_changes_only_secondary_metric():
    result=analyze_episode(sample_row([[True,True],[True,False],[True,True]]),'test')
    assert result['role_only_violation'] and not result['shared_physical_violation']
    assert not result['strict_safe']
    assert result['protected_completion_grace50ms'] and result['protected_completion_grace100ms']


def test_physical_unsupported_limit_stays_fixed_under_role_grace():
    row=sample_row([[False,False]]*3)
    result=analyze_episode(row,'test')
    assert result['shared_physical_violation']
    assert not result['protected_completion_grace100ms']
