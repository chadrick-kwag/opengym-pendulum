""" 
run env with ckpt
"""


import argparse, os, gym, torch, time
from model.actor import Actor
from utils import normalize_state, denormalize_action

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('-actor_ckpt', type=str, help='actor ckpt file', required=True)


    args = parser.parse_args()

    assert os.path.exists(args.actor_ckpt)


    device = torch.device('cuda:0')

    env = gym.make('Pendulum-v0')

    state_size = env.observation_space.shape[0]
    actor = Actor(state_size)
    actor.load_state_dict(torch.load(args.actor_ckpt))
    actor.to(device)


    actor.eval()


    run_count = 10
    max_step = 200

    for _ in range(run_count):

        state = env.reset()
        state = normalize_state(state)
        state = torch.FloatTensor(state).to(device)

        for step in range(max_step):

            env.render()

            dist = actor(state)
            a = dist.sample()
            actor_val = denormalize_action(a.item())

            next_state, reward, done, _ = env.step([actor_val]) 

            next_state = torch.FloatTensor(normalize_state(next_state)).to(device)

            if done:
                break
             
        
        

        time.sleep(1)
