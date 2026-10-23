"""Event-level computation, durable writes, failures and recovery."""
import json
import math
from pathlib import Path
import numpy as np

def load_worlds(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


class Environment:
    def __init__(self, world, config):
        self.w, self.config = world, config
        self.work, self.cap = world['work'], world['horizon']
        self.t=self.progress=self.saved=self.last_failure=0.
        self.pointer=0;self.done=False;self.completed=False
        self.last_checkpoint=None;self.checkpoint_id=0;self.used=set()
        self.attempts=self.successes=self.failed=self.unfinished=0
        self.writing=self.failed_write=self.lost=self.recovery=0.
        self.active_failures=self.recovery_failures=self.usable=self.protected=0
        self.gaps=[];self.actions=[0,0];self.return_sum=0.

    def cost(self):
        return self.w['costs'][min(int(self.t/self.w['block']),len(self.w['costs'])-1)]

    def mask(self):
        return np.array([True, self.progress>self.saved+1e-8 and not self.done],dtype=bool)

    def observation(self):
        unsaved=max(0.,self.progress-self.saved)
        telemetry=self.w['telemetry'][min(int(self.t/self.w['block']),len(self.w['telemetry'])-1)]
        age=max(0.,self.t-self.last_failure)
        if self.w['aging_telemetry']:
            telemetry=float(np.clip(.75*age/self.w['public_mtbf']+.25*telemetry,0,1))
        cost=self.cost()
        return np.array([self.progress/self.work,unsaved/self.work,
            math.log1p(unsaved)/12, math.log1p(self.work)/12,
            cost/self.work,math.log1p(cost)/8,math.log1p(self.w['public_mtbf'])/14,
            self.w['recovery']/self.work,self.t/self.cap,math.log1p(age)/14,
            min(4.,(self.active_failures+self.recovery_failures)*3600/(self.t+3600))/4,
            telemetry,float(self.saved>0)],dtype=np.float32)

    def potential(self):
        return 0. if self.done else -(self.work-self.progress)/self.work

    def step(self, action):
        if self.done or not self.mask()[action]: raise ValueError('Illegal action')
        before=self.t;phi=self.potential();self.actions[action]+=1
        duration=self.cost() if action else min(self.config['compute_quantum_seconds'],self.work-self.progress)
        if action: self.attempts+=1
        end=min(self.cap,self.t+duration)
        next_failure=(self.w['failures'][self.pointer] if self.pointer<len(self.w['failures']) else math.inf)
        fail=next_failure<=end
        delta=(next_failure if fail else end)-self.t
        self.t+=delta
        if action:self.writing+=delta
        else:self.progress+=delta
        if fail:
            self.pointer+=1;self.active_failures+=1;self.last_failure=self.t
            loss=self.progress-self.saved;self.lost+=loss
            if self.saved>0:
                self.usable+=1;self.used.add(self.checkpoint_id)
                self.gaps.append(self.t-self.last_checkpoint)
                self.protected+=int(loss<=self.config['protection_budget_seconds']+1e-8)
            if action:self.failed+=1;self.failed_write+=delta
            self.progress=self.saved
            recovery_end=min(self.cap,self.t+self.w['recovery'])
            while self.pointer<len(self.w['failures']) and self.w['failures'][self.pointer]<=recovery_end:
                ft=self.w['failures'][self.pointer];self.pointer+=1
                self.recovery_failures+=1;self.last_failure=ft
                recovery_end=min(self.cap,ft+self.w['recovery'])
            self.recovery+=recovery_end-self.t;self.t=recovery_end
        elif action:
            if delta+1e-8>=duration:
                self.saved=self.progress;self.successes+=1;self.checkpoint_id+=1;self.last_checkpoint=self.t
            else:self.unfinished+=1
        self.completed=self.progress>=self.work-1e-7
        self.done=self.completed or self.t>=self.cap-1e-7
        penalty=(self.config['timeout_penalized_work_multiple']*self.work-self.t
                 if self.done and not self.completed else 0.)
        reward=-(self.t-before+penalty)/self.work+self.potential()-phi
        self.return_sum+=reward
        return self.observation(),float(reward),self.done,self.mask()

    def metrics(self):
        if not self.done:raise ValueError('Episode not finished')
        score=self.t if self.completed else self.config['timeout_penalized_work_multiple']*self.work
        return {'world_id':self.w['id'],**self.w['cell'],'work_seconds':self.work,
                'elapsed_seconds':self.t,'penalized_seconds':score,'normalized_time':score/self.work,
                'completed':int(self.completed),'progress_seconds':self.progress,
                'attempts':self.attempts,'successful_writes':self.successes,'failed_writes':self.failed,
                'unfinished_writes':self.unfinished,'writing_seconds':self.writing,
                'failed_write_seconds':self.failed_write,'lost_work_seconds':self.lost,
                'recovery_seconds':self.recovery,'active_failures':self.active_failures,
                'recovery_failures':self.recovery_failures,'usable_failures':self.usable,
                'protected_failures':self.protected,'conditional_gaps_seconds':self.gaps,
                'used_writes':len(self.used),'observed_unused_writes':self.successes-len(self.used),
                'compute_actions':self.actions[0],'checkpoint_actions':self.actions[1],
                'episode_return':self.return_sum,
                'accounting_residual':self.t-self.progress-self.lost-self.writing-self.recovery}
