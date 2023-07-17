import unittest.mock as mock

from optimizer.position_utils import Optimizer

@mock.patch("optimizer.position_utils.Optimizer")
def test_valid_init(mock_optimizer):
    optimizer = mock_optimizer.return_value
    optimizer.name = "futures"

    assert optimizer.name == "futures"

