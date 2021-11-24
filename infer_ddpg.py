import torch, os, argparse, gym, torch, numpy as np
from model.actor import Actor_ddpg_v2


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-ckpt", type=str, help="ckpt path", required=True)
    parser.add_argument(
        "-steps", type=int, help="steps per episode", default=200, required=False
    )

    args = parser.parse_args()

    assert os.path.exists(args.ckpt)
    assert args.steps > 0

    ckpt = args.ckpt

    env = gym.make("Pendulum-v0")

    state_size = env.observation_space.shape[0]

    device = torch.device("cuda:0")

    actor = Actor_ddpg_v2(state_size)
    actor.load_state_dict(torch.load(ckpt))
    actor.to(device)
    actor.eval()

    max_step = args.steps
    epi_count = 0
    while True:

        print(f"epi: {epi_count}")

        state = env.reset()

        for _ in range(max_step):
            env.render()
            state = torch.FloatTensor([state.flatten()])
            state = state.to(device)

            with torch.no_grad():
                action = actor(state)

            action = action.cpu().detach().numpy()

            next_state, reward, done, _ = env.step(action)

            state = next_state

        epi_count += 1
