from xuance.engine.run_basic import RunnerBase
from trajectory_prediction.engine import write_json
from .agent import TrajectoryPredictionAgent


class RunnerTrajectoryPrediction(RunnerBase):
    """Native Xuance Runner for a dataset, rather than an online RL environment.

    RunnerBase.run(), dispatch, cleanup and ownership semantics are reused.
    Its online environment constructor is deliberately replaced for offline data.
    """
    def __init__(self, config):
        self.config = config
        self.rank, self.n_envs, self.envs = 0, 0, None
        self._own_envs, self._own_agent, self.manage_resources = False, True, True
        self.agent = TrajectoryPredictionAgent(config)

    def _run_train(self, **kwargs):
        return self.agent.train(**kwargs)

    def _run_test(self, **kwargs):
        return self.agent.test(**kwargs)

    def _run_benchmark(self, **kwargs):
        trained = self.agent.train(**kwargs)
        metrics = self.agent.test(output_path=self.agent.output / "test_metrics.json")
        result = {"training": trained, "test": metrics}
        write_json(self.agent.output / "benchmark_summary.json", result)
        return result
