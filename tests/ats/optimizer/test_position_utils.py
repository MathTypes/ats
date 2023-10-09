import math
import logging
import unittest.mock as mock

import numpy as np
import torch

from ats.optimizer.position_utils import Optimizer, min_max_rets
from ats.util import logging_utils


@mock.patch("ats.optimizer.position_utils.Optimizer")
def test_valid_init(mock_optimizer):
    optimizer = mock_optimizer.return_value
    optimizer.name = "futures"
    assert optimizer.name == "futures"


def test_portfolio_opt():
    ret_fcst = np.array([0, 0, 0, 1])
    w = np.array([0, 0, 0, 1])
    ret = np.dot(ret_fcst, w)
    assert ret == 1
    w = np.array([0, 0, 1, 0])
    ret = np.dot(ret_fcst, w)
    assert ret == 0
    logging.info(f"ret:{ret}")


def test_max_pos_fcst():
    ret_fcst = np.array([0, 1, 2, -1])
    min_ret, max_ret = min_max_rets(ret_fcst)
    assert math.isclose(min_ret, 0, rel_tol=0.01)
    assert math.isclose(max_ret, 3, rel_tol=0.01)


def test_single_asset_gamma_zero():
    initial_positions = torch.tensor([0])
    optimizer = Optimizer(
        name="opt", max_loss=0, gamma=0, initial_positions=initial_positions
    )
    returns_fcst = np.array([[0.1, 0.1, 0.1, 0.1]])
    min_neg_fcst = np.array([-0.1])
    max_pos_fcst = np.array([0.2])
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    assert math.isclose(new_positions, 0, rel_tol=0.01)
    # assert math.isclose(ret[0], 0, rel_tol=0.01)
    # assert math.isclose(val, 0, rel_tol=0.01)


def test_single_asset_gamma_two():
    initial_positions = torch.tensor([0])
    optimizer = Optimizer(
        name="opt", max_loss=0.2, gamma=2, sigma=1,
        initial_positions=initial_positions
    )
    returns_fcst = np.array([[0.1, 0.1, 0.1, 0.1]])
    min_neg_fcst = np.array([-0.2])
    max_pos_fcst = np.array([0.2])
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    logging.error(f"new_positions:{new_positions}, ret:{ret}, val:{val}")
    assert math.isclose(new_positions, 2, rel_tol=0.01)


def test_single_asset_gamma_two_neg_fcst_med_med_risk():
    initial_positions = torch.tensor([0])
    optimizer = Optimizer(
        name="opt", max_loss=0.2, gamma=2, sigma=2,
        initial_positions=initial_positions
    )
    returns_fcst = np.array([[0.1, 0.1, 0.1, 0.1]])
    min_neg_fcst = np.array([-0.2])
    max_pos_fcst = np.array([0.2])
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    logging.error(f"new_positions:{new_positions}, ret:{ret}, val:{val}")
    assert math.isclose(new_positions, 2, rel_tol=0.01)

def test_single_asset_gamma_two_neg_fcst_med():
    initial_positions = torch.tensor([0])
    optimizer = Optimizer(
        name="opt", max_loss=0.2, gamma=2, sigma=1,
        initial_positions=initial_positions
    )
    returns_fcst = np.array([[0.1, 0.1, 0.1, 0.1]])
    min_neg_fcst = np.array([-0.2])
    max_pos_fcst = np.array([0.2])
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    logging.error(f"new_positions:{new_positions}, ret:{ret}, val:{val}")
    assert math.isclose(new_positions, 2, rel_tol=0.01)

def test_single_asset_gamma_two_neg_fcst_large():
    initial_positions = torch.tensor([0])
    optimizer = Optimizer(
        name="opt", max_loss=0.2, gamma=2, sigma=1,
        initial_positions=initial_positions
    )
    returns_fcst = np.array([[0.1, 0.1, 0.1, 0.1]])
    min_neg_fcst = np.array([-0.4])
    max_pos_fcst = np.array([0.2])
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    logging.error(f"new_positions:{new_positions}, ret:{ret}, val:{val}")
    assert math.isclose(new_positions, 1.1, rel_tol=0.01)

def test_single_asset_large_pos_no_max_loss_hit_gamma():
    initial_positions = torch.tensor([0])
    returns_fcst = np.array([[0.1, 0.3, 0.2, 0.7]])
    min_neg_fcst = np.array([0])
    max_pos_fcst = np.array([0.7])
    optimizer = Optimizer(
        name="opt", max_loss=0, gamma=2, sigma=2,
        initial_positions=initial_positions
    )
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    assert math.isclose(new_positions, 2, rel_tol=0.01)


def test_single_asset_large_neg_no_max_loss_hit_gamma():
    initial_positions = torch.tensor([0])
    returns_fcst = np.array([[-0.1, -0.3, -0.2, -0.7]])
    min_neg_fcst = np.array([-0.7])
    max_pos_fcst = np.array([0])
    optimizer = Optimizer(
        name="opt", max_loss=0, gamma=2, sigma=2,
        initial_positions=initial_positions
    )
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    assert math.isclose(new_positions, -2, rel_tol=0.01)


def test_single_asset_large_pos_zero_max_loss():
    # reduce risk sigma, we should hit position limit (gamma)
    initial_positions = torch.tensor([0])
    returns_fcst = np.array([[0.1, 0.3, 0.2, 0.7]])
    min_neg_fcst = np.array([-0.7])
    max_pos_fcst = np.array([0.1])
    optimizer = Optimizer(
        name="opt", max_loss=0, gamma=2, sigma=0.2,
        initial_positions=initial_positions
    )
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    assert math.isclose(new_positions, 0, rel_tol=0.01)

def test_single_asset_large_neg_zero_max_loss():
    # reduce risk sigma, we should hit position limit (gamma)
    initial_positions = torch.tensor([0])
    returns_fcst = np.array([[-0.1, -0.3, -0.2, -0.7]])
    min_neg_fcst = np.array([-0.1])
    max_pos_fcst = np.array([0.7])
    optimizer = Optimizer(
        name="opt", max_loss=0, gamma=2, sigma=0.2,
        initial_positions=initial_positions
    )
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    assert math.isclose(new_positions, 0, rel_tol=0.01)

def test_single_asset_large_pos_some_max_loss():
    # some risk tolerance
    initial_positions = torch.tensor([0])
    returns_fcst = np.array([[0.1, 0.3, 0.2, 0.7]])
    min_neg_fcst = np.array([-0.1])
    max_pos_fcst = np.array([0.7])
    # increase gamma
    optimizer = Optimizer(
        name="opt", max_loss=0.5, gamma=4, sigma=20,
        initial_positions=initial_positions
    )
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    assert math.isclose(new_positions, 3.3, rel_tol=0.01)
    

def test_single_asset_large_neg_some_max_loss():
    # some risk tolerance
    initial_positions = torch.tensor([0])
    returns_fcst = np.array([[-0.1, -0.3, -0.2, -0.7]])
    min_neg_fcst = np.array([-0.7])
    max_pos_fcst = np.array([0.1])
    # increase gamma
    optimizer = Optimizer(
        name="opt", max_loss=0.5, gamma=4, sigma=20,
        initial_positions=initial_positions
    )
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    assert math.isclose(new_positions, -3.3, rel_tol=0.01)

def test_single_asset_large_neg_some_max_loss2():
    # some risk tolerance
    initial_positions = torch.tensor([0])
    returns_fcst = np.array([[-0.1, -0.3, -0.2, -0.7]])
    min_neg_fcst = np.array([-0.7])
    max_pos_fcst = np.array([0.4])
    # increase gamma
    optimizer = Optimizer(
        name="opt", max_loss=0.5, gamma=4, sigma=20,
        initial_positions=initial_positions
    )
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    assert math.isclose(new_positions, -0.2, rel_tol=0.01)

def test_single_asset_large_neg_drawback():
    initial_positions = torch.tensor([0])
    optimizer = Optimizer(
        name="opt", max_loss=0.5, gamma=2, sigma=2, initial_positions=initial_positions
    )
    returns_fcst = np.array([[0.1, -0.3, 0.2, 0.7]])
    min_neg_fcst = np.array([-0.7])
    max_pos_fcst = np.array([0.7])
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    # Even though return fcst is positive, there are large neg
    # fcst to hold it back
    assert math.isclose(new_positions, 0.4, rel_tol=0.01)


def test_single_asset_small_neg_drawback():
    initial_positions = torch.tensor([0])
    optimizer = Optimizer(
        name="opt",
        max_loss=0.2,
        gamma=2,
        sigma=2,
        pnl_risk=10,
        initial_positions=initial_positions,
    )
    returns_fcst = np.array([[0.1, -0.3, 0.2, 0.7]])
    min_neg_fcst = np.array([-0.5])
    max_pos_fcst = np.array([0.7])
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    # TODO: fix this test by adding back assert
    # assert math.isclose(new_positions, 2, rel_tol=0.01)


def test_two_assets_gamma_two():
    initial_positions = torch.tensor([0, 0])
    optimizer = Optimizer(
        name="opt", max_loss=0.2, gamma=4, initial_positions=initial_positions
    )
    returns_fcst = np.array([[0.1, 0.1, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0]])
    min_neg_fcst = np.array([-0.1, 0])
    max_pos_fcst = np.array([0.2, 0])
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    logging.info(f"new_positions:{new_positions}")
    assert math.isclose(new_positions[0], 4.0, rel_tol=0.01)
    assert math.isclose(new_positions[1], 0, rel_tol=0.01)


if __name__ == "__main__":
    logging_utils.init_logging()
