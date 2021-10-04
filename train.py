""" 
train actor/critic for pendulum

https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
"""


import gym, datetime, os, torch, argparse, shutil
from torch.utils.tensorboard import SummaryWriter

from model.actor import Actor
from model.critic import Critic


class PriorityQueue:

    def __init__(self):
        self.item_list = []

    def __len__(self):

        return len(self.item_list)
    
    def add(self, item):

        self.item_list.append(item)
        self.item_list.sort(key=lambda x: x)

    def getbyindex(self, idx):

        return self.item_list[idx]
    
    def remove_index(self, idx):

        del self.item_list[idx]

        


def normalize_state(state):
    # range: [-2,2] -> [-1,1]


    return state/2


def denormalize_action(action_val):

    return action_val * 2


if __name__ == '__main__':


    parser = argparse.ArgumentParser()



    # setup env

    env = gym.make('Pendulum-v0')

    # state_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

     
    # config
    actor_lr = 1e-4
    critic_lr = 1e-4
    device = torch.device('cuda:0')
    gamma = 0.99
    first_success_wait_episode_count = 10
    total_episode_count = 1000


    actor = Actor(state_size).to(device)
    critic = Critic(state_size).to(device)


    # setup optimizer
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr = critic_lr)

    episode_count = 0 
    first_success_reached = False
    patient_count = 0
    patient_epi_count = 5

    best_acc_reward = None

    # setup outputdir 
    timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    outputdir = f'ckpt/{timestamp}'
    os.makedirs(outputdir)

    # setup ckpt pq
    keep_size = 5
    pq = PriorityQueue()


    # setup writer
    logdir = os.path.join(outputdir, 'logs')
    os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    while True:


        # run episode 

        state = env.reset()

        state = normalize_state(state)
        state = torch.FloatTensor(state).to(device)

        max_step = 1000

        acc_reward = 0
        log_prob_list = []
        reward_list = []
        curr_value_list = []

        term_reached = False

        for step in range(max_step):

            # predict action and logprob
            dist = actor(state)

            a = dist.sample()
            log_prob = dist.log_prob(a).unsqueeze(0)

            log_prob_list.append(log_prob)

            action = denormalize_action( a.item() )

            next_state, reward, done, _ = env.step([action])

            acc_reward += reward
            reward_list.append(reward)
            
            next_state = normalize_state(next_state)
            next_state = torch.FloatTensor(next_state).to(device)

            curr_value = critic(next_state)
            curr_value_list.append(curr_value)


            state = next_state

            if done: 
                term_reached = True
                break

        # calculate advantages
        returns = []
        if term_reached:
            r = 0
        else:
            r = curr_value_list[-1].item()

        for reward in reversed(reward_list):
            r = reward + gamma * r 
            returns.insert(0, r)
        
        returns = torch.FloatTensor(returns).to(device).detach()

        curr_values = torch.FloatTensor(curr_value_list).to(device)

        advantages= returns - curr_values 

        log_probs = torch.cat(log_prob_list)

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = advantages.pow(2).mean()

        actor_optim.zero_grad()
        critic_optim.zero_grad()

        (actor_loss + critic_loss).backward()

        actor_optim.step()
        critic_optim.step()
        

        # save metric
        writer.add_scalar('train/steps', step, episode_count)        
        writer.add_scalar('train/actor_loss', actor_loss.item(), episode_count)
        writer.add_scalar('train/critic_loss', critic_loss.item(), episode_count)
        writer.add_scalar('train/acc_reward', acc_reward, episode_count)
        writer.flush()

        # print
        print(f'epi:{episode_count}> step={step}, actor_loss={actor_loss.item()}, critic_loss={critic_loss.item()}, acc reward={acc_reward}')
        
        
        if best_acc_reward is None or acc_reward > best_acc_reward:
            # save current ckpt

            savedir = os.path.join(outputdir, f'epi_{episode_count}_accreward={acc_reward:.2f}')
            os.makedirs(savedir)

            savepath = os.path.join(savedir, 'actor.pt')
            torch.save(actor.state_dict(), savepath)

            savepath = os.path.join(savedir, 'critic.pt')
            torch.save(critic.state_dict(), savepath)

            pq.add((-acc_reward, savedir))

            # if pq is full, remove worst
            if len(pq)> keep_size:
                
                _, d = pq.getbyindex(-1)
                shutil.rmtree(d)
                pq.remove_index(-1)

        episode_count +=1
