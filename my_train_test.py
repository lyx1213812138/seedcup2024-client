from std_env import Env
from team_algorithm import MyCustomAlgorithm

def main(logger, algorithm = MyCustomAlgorithm()):
    env = Env(is_senior=False,seed=10430,gui=False)
    done = False
    num_episodes = 15
    final_score = 0
    total_steps = 0
    total_distance = 0

    for i in range(num_episodes):
        score = 0
        done = False

        while not done:
            observation = env.get_observation()
            action = algorithm.get_action(observation)
            obs = env.step(action)
            score += env.success_reward

            # Check if the episode has ended
            done = env.terminated

        total_steps += env.step_num
        total_distance += env.get_dis()
        final_score += score
        # print("Episode", i, "completed. Score:", score)

    final_score /= num_episodes
    avg_distance = total_distance / num_episodes
    avg_steps = total_steps / num_episodes

    logger.record("Total steps", avg_steps)
    logger.record("Final distance", avg_distance)
    logger.record("Final score", final_score)
    # After exiting the loop, get the total steps and final distance
    # print("Test completed. Total steps:", avg_steps, "Final distance:", avg_distance, "Final score:", final_score)