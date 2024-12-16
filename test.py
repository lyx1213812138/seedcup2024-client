from std_env import Env
from zipp.team_algorithm import MyCustomAlgorithm
from datetime import datetime

def main(algorithm):
    now = datetime.now()
    env = Env(is_senior=False,seed=112,gui=True)
    done = False
    num_episodes = 100
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

        print(f"Test_{i} completed. steps:", env.step_num, "Distance:", env.get_dis(), "Score:", score)
        # if score < 70:
        #     from time import sleep
        #     sleep(10000)

    final_score /= num_episodes
    avg_distance = total_distance / num_episodes
    avg_steps = total_steps / num_episodes

    # After exiting the loop, get the total steps and final distance
    print("Test completed. Total steps:", avg_steps, "Final distance:", avg_distance, "Final score:", final_score)
    env.close()
    print("Time:", datetime.now()-now)

if __name__ == "__main__":
    algorithm = MyCustomAlgorithm()
    print(algorithm.__class__.__name__)
    # algorithm = TriangleAlgorithm()
    main(algorithm)