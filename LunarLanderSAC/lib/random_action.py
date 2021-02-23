from . import core, rb, save
import datetime

def random_action(env, episodes=1000):
    # Prepare for interaction with environment
    run_folder = save.create_run_folder()
    run_start_time = datetime.datetime.now()
    t = 0

    for curr_episode in range(episodes):
        o, ep_ret, ep_len = env.reset(), 0, 0
        d = False

        while not d:
            a = env.action_space.sample()

            # Step the env
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            o = o2

            if d:
                print(f"episode: {curr_episode}, reward: {ep_ret}, length: {ep_len}")
                with open(f"{run_folder}/reward.csv", "a") as myfile:
                    myfile.write(f"{ep_ret}\n")

            t += 1

    run_stop_time = datetime.datetime.now()
    with open(f"{run_folder}/runtime", "a") as myfile:
        myfile.write(f"{run_stop_time - run_start_time}\n")