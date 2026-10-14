from xuance.engine.uav_baseline_logging import RewardLossTensorBoard


class Writer:
    def __init__(self):
        self.points = []

    def add_scalar(self, name, value, step):
        self.points.append((name, value, step))


def episode(value, level=1):
    return {"Train-Returns/team_mean": value, "Train-Returns/uav_0": value,
            "Train-Rewards/team/r_purs": value + 1, "Train-Rewards/team/r_failure": -1,
            "Train-Rewards/team/r_form": 0, "Train-Curriculum/Level": level,
            "Train-Distance/Final-Mean": 300, "Train-Rewards-Per-Step/team/r_purs": 1}


def test_rewards_preserve_batch_means_and_sum_of_components():
    writer = Writer()
    log = RewardLossTensorBoard(writer)
    for i in range(1, 21):
        log.log(episode(i), i * 16)
    points = {name: (value, step) for name, value, step in writer.points}
    assert points == {"Train-Returns/team_mean": (10.5, 320), "Train-Returns/uav_0": (10.5, 320),
                      "Train-Rewards/team/r_purs": (11.5, 320), "Train-Rewards/team/r_failure": (-1, 320)}
    assert points["Train-Returns/team_mean"][0] == sum(points[k][0] for k in points if k.startswith("Train-Rewards/"))


def test_level_change_and_final_partial_batch_do_not_drop_or_mix_rewards():
    writer = Writer()
    log = RewardLossTensorBoard(writer)
    log.log(episode(4), 16)
    log.log(episode(8, level=2), 32)
    log.flush_rewards()
    log.flush_rewards()
    assert [p for p in writer.points if p[0] == "Train-Returns/team_mean"] == [
        ("Train-Returns/team_mean", 4, 16), ("Train-Returns/team_mean", 8, 32)]


def test_only_summarized_losses_and_evaluation_reward_are_written():
    writer = Writer()
    log = RewardLossTensorBoard(writer)
    log.log({"uav_0/loss_actor": -5, "uav_0/loss_critic": 8, "uav_0/predictQ": 4,
             "noise_scale": .1, "Baseline-Eval/current/Level-1/safe_success_rate": .7}, 100)
    assert writer.points == []
    log.log({"Train-Learning/loss_actor_mean": -3, "Train-Learning/loss_critic_mean": 6,
             "Train-Learning/predictQ_mean": 4, "Test-Results/Episode-Rewards-Level-0": 20,
             "Test-Results/Episode-Rewards-Std-Level-0": 8}, 200)
    assert writer.points == [("Loss/loss_actor", -3, 200), ("Loss/loss_critic", 6, 200),
                             ("Eval-Rewards/Level-1", 20, 200)]
