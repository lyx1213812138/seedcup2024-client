from std_env import Env
from team_algorithm import MyCustomAlgorithm 
import numpy as np
from time import sleep, time
from pynput import keyboard
import threading
from ccalc import Calc

calc = Calc()

pause = False 

def main(algorithm):
    global pause, env
    env = Env(is_senior=False,seed=10430,gui=True, pos='left')
    done = False
    printed = True
    while True:
        if pause:
            # print('Paused')
            sleep(0.5)
            continue
        if not done:
            observation = env.get_observation()
                
            action = algorithm.get_action(observation)
            obs = env.step(action)
            sleep(0.05)

        # Check if the episode has ended
        done = env.terminated
        if done:
            print(f"Test completed. steps:", env.step_num, "Distance:", env.get_dis(), "Score:", env.success_reward)
            sleep(1)
            env.reset()
            done = False


def on_press(key):
    global pause
    try:
        # print(f'Key pressed: {key.char}')
        if key.char == 's':
            pause = True
        elif key.char == 'd':
            pause = False
        elif key.char == 'r':
            env.reset()
            pause = False

    except AttributeError:
        print(f'Special key pressed: {key}')

def on_release(key):
    if key == keyboard.Key.esc:
        return False  # Stop the listener

def dokeyboard():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == "__main__":
    algorithm = MyCustomAlgorithm()
    # algorithm = TriangleAlgorithm()
    t = threading.Thread(target=dokeyboard)
    t.start()
    main(algorithm)
    t.join()