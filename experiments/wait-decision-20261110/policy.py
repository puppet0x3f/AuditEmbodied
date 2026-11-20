"""Three frozen-MLP uses of the same permitted waiting observation stream."""
import numpy as np
from auditembodied.admissible_mlp import StartGatedMLP, start_mask
from auditembodied.l4_offline_selection import features
from auditembodied.l4_lift import CANDIDATES

MODES=('cached','refreshed','wait_conditioned')


class WaitingPolicy:
    def __init__(self,values,mode):
        assert mode in MODES
        self.backend=StartGatedMLP(values)
        self.values=values
        self.mode=mode

    def initial(self,port):
        self.cached=self.backend(port)
        return self.cached

    def __call__(self,history):
        if self.mode=='cached':
            return {'candidate':self.cached['candidate'],'mode':self.mode,'threshold':self.values['threshold'],
                'safety_scores':self.cached['safety_scores'],'start_admissible':self.cached['start_admissible'],
                'scored_observations':0,'observed_hold_seconds':history[-1]['elapsed_seconds'],
                'aggregation':'Reuse initial decision without another model invocation'}
        model=self.backend.model
        use=history[:1] if self.mode=='cached' else history[-1:] if self.mode=='refreshed' else history
        x=np.stack([features(h['observation'],model.rule) for h in use])
        probabilities=model.probabilities(x)
        masks=np.stack([start_mask(row) for row in x])
        score=probabilities.min(0)
        mask=masks.all(0)
        order=np.argsort(self.values['action_cost_seconds'],kind='stable')
        candidate=next((CANDIDATES[j] for j in order if mask[j] and score[j]>=self.values['threshold']),'abstain')
        return {'candidate':candidate,'mode':self.mode,'threshold':self.values['threshold'],
            'safety_scores':score.tolist(),'start_admissible':mask.tolist(),
            'scored_observations':len(use),'observed_hold_seconds':history[-1]['elapsed_seconds'],
            'aggregation':'minimum frozen candidate score and conjunction of observable start masks'}
