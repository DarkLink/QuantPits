import pytest

from quantpits.training.errors import TrainingRunnerContractError
from quantpits.training.runners import TrainingTargetRequest, _adapt_result


def test_runner_rejects_none_result():
    with pytest.raises(TrainingRunnerContractError):
        _adapt_result(TrainingTargetRequest(None, None), None)
