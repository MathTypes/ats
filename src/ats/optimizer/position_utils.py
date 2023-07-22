import logging
import cvxpy as cp
import numpy as np


def min_max_rets(returns_fcst):
    cumsum = returns_fcst.cumsum()
    return cumsum.min(), cumsum.max()


class Optimizer(object):
    def __init__(
        self,
        name: str,
        max_loss,
        gamma=0,
        sigma=0,
        l_norm=1.5,
        pnl_risk_norm=3,
        pnl_risk=10,
        initial_positions=None,
    ):
        super().__init__()
        self.initial_positions = initial_positions
        self.max_loss = max_loss
        self.gamma = gamma
        self.sigma = sigma
        self.n = initial_positions.shape[0]
        self.l_norm = l_norm
        self.pnl_risk_norm = pnl_risk_norm
        self.pnl_risk = pnl_risk

    def optimize(self, returns_fcst, min_neg_fcst, max_pos_fcst):
        w = cp.Variable(self.n)
        logging.info(f"returns_fcst:{returns_fcst}, w:{w}")
        cum_rets = np.sum(returns_fcst, axis=-1)
        logging.info(f"cum_rets:{cum_rets}")
        ret = cp.sum(cum_rets @ w)
        logging.info(f"ret:{ret}, w:{w}")
        # We need a bit power to limit positions.
        buy_risk = cp.sum_squares(cp.maximum(w,0) @ cp.minimum(min_neg_fcst, 0))
        sell_risk = cp.sum_squares(cp.minimum(w,0) @ cp.maximum(max_pos_fcst, 0))
        risk = buy_risk + sell_risk
        objective = cp.Minimize(self.sigma * risk-ret)
        logging.info(f"ret:{ret}, risk:{risk}, objective:{objective}")
        # objective = cp.Maximize(ret)
        # wors_hour = cp.sum_smallest(np.dot(np.sum(quantile_returns_fcst), w))

        constraints = [
            cp.norm(w, self.l_norm) <= self.gamma,
            cp.norm(risk, self.pnl_risk) <= self.pnl_risk_norm,
            risk <= self.max_loss,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()
        logging.info(f"w:{w.value}, problem.value:{problem.value}, ret:{ret.value}")
        return (
            w.value.round(1).ravel(),
            ret.value,
            problem.value,
        )
