"""Double DQN with a shared MLP and uniform/proportional prioritized replay.

No supervised examples, baseline action targets, search or fallback controller.
"""
import json
import numpy as np


class Network:
    def __init__(self, seed, inputs, actions):
        self.inputs,self.n_actions=inputs,actions
        rng=np.random.default_rng(seed); self.params=[]
        for i,(a,b) in enumerate([(inputs,96),(96,96),(96,actions)]):
            self.params.extend([rng.normal(0,np.sqrt((2 if i<2 else .02)/a),(a,b)).astype(np.float32), np.zeros(b,np.float32)])

    def forward(self,x,cache=False):
        w1,b1,w2,b2,w3,b3=self.params
        h1=np.maximum(0,x@w1+b1);h2=np.maximum(0,h1@w2+b2);q=h2@w3+b3
        return (q,(x,h1,h2)) if cache else q

    def actions(self,x,masks): return np.argmax(np.where(masks,self.forward(x),-np.inf),axis=1)

    def copy_from(self,other):
        for a,b in zip(self.params,other.params): a[:]=b

    def save(self,path,metadata):
        np.savez_compressed(path,**{f'p{i}':p for i,p in enumerate(self.params)},metadata=json.dumps(metadata))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            net=cls(0,d['p0'].shape[0],d['p5'].shape[0])
            for i,p in enumerate(net.params):p[:]=d[f'p{i}']
        return net


class Replay:
    def __init__(self,capacity,inputs,actions,alpha=0.):
        self.capacity=capacity;self.size=self.cursor=0;self.alpha=alpha;self.max_priority=1.
        self.x=np.zeros((capacity,inputs),np.float32);self.nx=np.zeros_like(self.x)
        self.a=np.zeros(capacity,np.int64);self.r=np.zeros(capacity,np.float32)
        self.done=np.zeros(capacity,bool);self.mask=np.zeros((capacity,actions),bool)
        self.leaves=1 << (capacity-1).bit_length()
        self.tree=np.zeros(2*self.leaves,np.float64)

    def priorities(self,ids,p):
        # Collapse repeated samples before updating parents, so sums remain exact.
        unique,inverse=np.unique(ids,return_inverse=True)
        values=np.zeros(len(unique));np.maximum.at(values,inverse,p)
        nodes=unique+self.leaves;self.tree[nodes]=values
        while nodes[0]>1:
            nodes=np.unique(nodes//2)
            self.tree[nodes]=self.tree[2*nodes]+self.tree[2*nodes+1]

    def add(self,x,a,r,nx,done,mask):
        ids=(np.arange(len(x))+self.cursor)%self.capacity
        for k,v in [('x',x),('a',a),('r',r),('nx',nx),('done',done),('mask',mask)]:getattr(self,k)[ids]=v
        if self.alpha:self.priorities(ids,np.full(len(ids),self.max_priority))
        self.cursor=(self.cursor+len(x))%self.capacity;self.size=min(self.capacity,self.size+len(x))

    def sample(self,rng,n,beta=1.):
        if self.alpha:
            v=(np.arange(n)+rng.random(n))*self.tree[1]/n
            nodes=np.ones(n,dtype=np.int64)
            while nodes[0]<self.leaves:
                left=nodes*2;go_right=v>=self.tree[left]
                v-=np.where(go_right,self.tree[left],0.);nodes=left+go_right
            ids=nodes-self.leaves
            probs=self.tree[nodes]/self.tree[1]
            weights=(self.size*probs)**(-beta)
            # Normalize by sampled maximum; conventional minibatch normalization.
            weights/=weights.max()
        else:
            ids=rng.integers(self.size,size=n);weights=np.ones(n)
        return [getattr(self,k)[ids] for k in ['x','a','r','nx','done','mask']]+[weights.astype(np.float32)],ids

    def update_priorities(self,ids,error):
        if self.alpha:
            p=(np.abs(error)+1e-4)**self.alpha
            self.max_priority=max(self.max_priority,float(p.max()));self.priorities(ids,p)


def double_targets(online,target,r,nx,done,mask):
    chosen=online.actions(nx,mask)
    return r+np.where(done,0.,target.forward(nx)[np.arange(len(chosen)),chosen])


class Optimizer:
    def __init__(self,network,lr,clip):
        self.m=[np.zeros_like(p) for p in network.params];self.v=[p.copy() for p in self.m]
        self.t=0;self.lr=lr;self.clip=clip

    def update(self,network,batch,target):
        x,a,r,nx,done,mask,weights=batch
        y=double_targets(network,target,r,nx,done,mask)
        q,(x,h1,h2)=network.forward(x,True);err=q[np.arange(len(a)),a]-y
        loss=float(np.mean(weights*.5*err**2))
        dq=np.zeros_like(q);dq[np.arange(len(a)),a]=weights*err/len(a)
        w1,b1,w2,b2,w3,b3=network.params
        g3=h2.T@dq;gb3=dq.sum(axis=0);d2=(dq@w3.T)*(h2>0)
        g2=h1.T@d2;gb2=d2.sum(axis=0);d1=(d2@w2.T)*(h1>0)
        grads=[x.T@d1,d1.sum(axis=0),g2,gb2,g3,gb3]
        norm=float(np.sqrt(sum(np.sum(g*g) for g in grads)))
        if not np.isfinite(loss+norm):raise FloatingPointError('Nonfinite update')
        self.t+=1;factor=min(1.,self.clip/max(norm,1e-12))
        for p,g,m,v in zip(network.params,grads,self.m,self.v):
            g=g*factor;m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
            p-=self.lr*(m/(1-.9**self.t))/(np.sqrt(v/(1-.999**self.t))+1e-8)
        return loss,norm,err


class SupportedActor:
    def __init__(self, actor, excluded):
        self.actor, self.excluded = actor, excluded

    def actions(self, x, masks):
        legal = masks.copy()
        legal[:, self.excluded] = False
        if not legal.any(axis=1).all():
            raise ValueError('No legal action remains')
        return self.actor.actions(x, legal)
