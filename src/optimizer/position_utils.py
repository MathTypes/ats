import logging
import cvxpy as cp
import numpy as np

class Optimizer(object):
    def __init__(self, name: str, max_loss, gamma, initial_positions: None):
        super().__init__()
        self.initial_positions = initial_positions
        self.max_loss = max_loss
        self.gamma = gamma
        self.n = initial_positions.shape[0]
        self.l_norm = 0.5
        
    def optimize(self, returns_fcst, quantile_returns_fcst):
        w = cp.Variable(self.n)
        logging.info(f"returns_fcst:{returns_fcst}, w:{w}")
        ret = np.dot(np.sum(returns_fcst), w)
        logging.info(f"ret:{ret}, w:{w}")
        #buy_risk = np.dot(quantile_returns_fcst[:,:,0], w*maximum(w,0))
        #sell_risk = np.dot(quantile_returns_fcst[:,:,1], w*minimum(w,0))
        #risk = buy_risk + sell_risk
        risk = 0
        objective = cp.Maximize(ret - self.gamma * risk)
        #wors_hour = cp.sum_smallest(np.dot(np.sum(quantile_returns_fcst), w))

        constraints = [
            cp.norm(w, self.l_norm) <= self.gamma,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()
        logging.info(f"w:{w.value}, problem.value:{problem.value}, ret:{ret.value}")
        return (
            w.value.round(1).ravel(),
            ret.value,
            problem.value,
        )
