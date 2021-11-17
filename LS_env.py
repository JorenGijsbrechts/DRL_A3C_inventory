import numpy as np
import itertools


def actions(args):
    actions = np.arange(args.max_order)
    return actions

def transition_stochLT(s, a, d, q_arrivals, args,LT):
    arrived = np.sum(s[1:][q_arrivals[1:]==0])
    s[1:][q_arrivals[1:]==0] = 0
    s[0] += arrived

    q_arrivals -= 1
    q_arrivals = np.roll(q_arrivals,-1)
    q_arrivals[-1] = LT-1
    q_arrivals = np.clip(q_arrivals,0,np.inf)

    s1 = np.roll(s, -1)

    s1[-1] = a
    s1[0] = np.clip(max(s[0] - d,0) + s[1], 0,args.inv_max - 1)
    reward = args.c*a + max(s[0] - d, 0) * args.h + min(s[0] - d, 0) * -args.p
    return reward, s1, q_arrivals