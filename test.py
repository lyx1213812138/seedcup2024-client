from env.env import Env
from team_algorithm import PPOAlgorithm, MyCustomAlgorithm, TriangleAlgorithm

def main(algorithm):
    env = Env(is_senior=False,seed=100,gui=True)
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

    final_score /= num_episodes
    avg_distance = total_distance / num_episodes
    avg_steps = total_steps / num_episodes

    # After exiting the loop, get the total steps and final distance
    print("Test completed. Total steps:", avg_steps, "Final distance:", avg_distance, "Final score:", final_score)
    env.close()

if __name__ == "__main__":
    algorithm = PPOAlgorithm()
    # algorithm = TriangleAlgorithm()
    main(algorithm)