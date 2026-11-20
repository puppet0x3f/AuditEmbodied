"""Change only the initial CAD executor's touchdown targets; retain its visual services."""
import numpy as np
from closed_loop import recover as original_recover


def recover(port):
    changed=[];target=None
    original_move=port['move'];original_phase=port['phase']
    def move(name,targets,steps,grippers=(1.,1.)):
        nonlocal target
        if name=='global_lower_to_table':
            at=np.asarray(port['robot']()['positions'])
            target=at.copy();target[:,2]-=.09
            changed.append({'time':port['time'](),'original_target':np.asarray(targets).tolist(),
                'replacement_target':target.tolist(),'original_steps':steps,'replacement_steps':30})
            return original_move(name,target,30,grippers)
        return original_move(name,targets,steps,grippers)
    def phase(name,targets,steps,grippers=(1.,1.)):
        if name=='global_table_loaded' and target is not None and len(changed)<=2:
            # Only loaded holds immediately following the two initial strokes
            # use the changed target. Clear it after each loaded hold.
            return initial_loaded(name,targets,steps,grippers)
        return original_phase(name,targets,steps,grippers)
    def initial_loaded(name,targets,steps,grippers):
        nonlocal target
        replacement=target;target=None
        return original_phase(name,replacement,steps,grippers)
    adapted=dict(port,move=move,phase=phase)
    log=original_recover(adapted)
    log['policy']='Frozen CAD executor with two fixed0.09m initial touchdown targets'
    log['touchdown_intervention']=changed
    return log
