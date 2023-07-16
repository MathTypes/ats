import cvxpy as cp

# initial_position: [instr_size]
# return_fcst: [instr_size, time_step]
# quantile_fcast: [instr_size, time_step, 2]
def maximize_trade_constrain_downside(
        self,
        initial_positions,
        return_fcst,
        quantile_fcst,
        max_loss,
        gamma
):
    n = initial_positions.shape[0]
    w = cp.Variable(n)
    ret = return_fcst.T @ w
    buy_risk = quantile_fcst[:,:,0].T @ (w*(w>0))
    sell_risk = quantile_fcst[:,:,1].T @ (w*(w<0))
    risk = buy_risk + sell_risk
    
    objective = cp.Maximize(ret - gamma * risk)
    wors_hour = cp.sum_smallest(quantile_rets)

    constraints = [
        cp.norm(weights, self.l_norm) <= self.gamma,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return (
        weights.value.round(1).ravel(),
        bid_return,
        problem.value,
    )
