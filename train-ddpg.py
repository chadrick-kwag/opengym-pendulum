""" 
train using ddpg
"""


from model.actor import Actor_ddpg, Actor_ddpg_v2
from model.critic import Critic_ddpg, Critic_ddpg_v2
import gym, os, torch, datetime, json
from memorybuffer import Memory
from torch.utils.tensorboard import SummaryWriter
from ckptsaver import Saver
from tqdm import tqdm
from munch import Munch
from noise import OrnsteinUhlenbeckActionNoise


train_step_count = 0


def run_valid_episode(env: gym.Env, actor, device, max_step):

    state = env.reset()
    acc_reward = 0
    actor.eval()

    for i in range(max_step):

        state_tensor = torch.FloatTensor([state]).squeeze(-1).to(device)

        with torch.no_grad():
            action = actor(state_tensor).cpu().detach().numpy()

        next_state, reward, done, _ = env.step(action)

        reward = reward[0]

        acc_reward += reward

        if done:
            break

        state = next_state

    return acc_reward


def run_valid(
    epi_index,
    env: gym.Env,
    actor,
    writer: SummaryWriter,
    saver: Saver,
    device: torch.device,
    repeat_num=5,
    max_step=200,
):

    acc_reward_list = []

    for _ in range(repeat_num):

        acc_reward = run_valid_episode(env, actor, device, max_step)

        acc_reward_list.append(acc_reward)

    mean_acc_reward = sum(acc_reward_list) / len(acc_reward_list)

    # write to tensorboard
    writer.add_scalar("valid/mean acc reward", mean_acc_reward, epi_index)

    # push to saver
    saver.add_new_metric(epi_index, mean_acc_reward, actor)

    # print metric
    print(f"epi: {epi_index}, mean acc reward={mean_acc_reward}")


def copy_network_params(src_network: torch.nn.Module, target_network: torch.nn.Module):

    for src_param, target_param in zip(
        src_network.parameters(), target_network.parameters()
    ):
        target_param.data.copy_(src_param.data)

    return


def update_target_network(
    src_network: torch.nn.Module, target_network: torch.nn.Module, tau=0.001
):

    for src_param, target_param in zip(
        src_network.parameters(), target_network.parameters()
    ):

        new_param = src_param.data * tau + (1 - tau) * target_param.data

        target_param.data.copy_(new_param)


def convert_batch_sample_to_tensors(batch_samples: list):

    s1_list = []
    a_list = []
    s2_list = []
    r_list = []

    for s1, a, s2, r in batch_samples:
        s1_list.append(s1.flatten().tolist())
        a_list.append(a.flatten().tolist())
        s2_list.append(s2.flatten().tolist())
        r_list.append(r.flatten().tolist())

    return (
        torch.FloatTensor(s1_list),
        torch.FloatTensor(a_list),
        torch.FloatTensor(s2_list),
        torch.FloatTensor(r_list),
    )


def run_train_step(
    memory: Memory,
    actor,
    actor_target,
    critic,
    critic_target,
    actor_optim,
    critic_optim,
    device: torch.device,
    writer: SummaryWriter,
    batch_size=16,
    gamma=0.99,
    tau=0.001,
):

    # fetch samples. if memory size is less than batch size, then skip.

    if len(memory) < batch_size:
        return

    batch_samples = memory.get_random_steps(batch_size)

    s1_arr, a_arr, s2_arr, r_arr = convert_batch_sample_to_tensors(batch_samples)

    # move tensors to device
    s1_arr = s1_arr.to(device)
    a_arr = a_arr.to(device)
    s2_arr = s2_arr.to(device)
    r_arr = r_arr.to(device)

    critic_target.eval()
    actor_target.eval()
    actor.train()
    critic.train()

    # get critic loss
    with torch.no_grad():
        y_arr = r_arr + gamma * critic_target(s2_arr, actor_target(s2_arr).detach())

    critic_loss = (y_arr.detach() - critic(s1_arr, a_arr)).pow(2).mean()

    critic_optim.zero_grad()
    actor_optim.zero_grad()

    critic_loss.backward()
    critic_optim.step()

    # get actor loss
    pred_a_arr = actor(s1_arr)
    actor_loss = -(critic(s1_arr, pred_a_arr)).mean()

    actor_loss.backward()
    actor_optim.step()

    # update target network
    update_target_network(actor, actor_target, tau=tau)
    update_target_network(critic, critic_target, tau=tau)

    # writer to writer

    global train_step_count
    writer.add_scalar("train/critic_loss", critic_loss.item(), train_step_count)
    writer.add_scalar("train/actor_loss", actor_loss.item(), train_step_count)

    train_step_count += 1

    return


def run_episode(
    epi_index,
    env: gym.Env,
    actor,
    actor_target,
    critic,
    critic_target,
    actor_optim,
    critic_optim,
    writer: SummaryWriter,
    device: torch.device,
    memory: Memory,
    saver: Saver,
    gamma,
    tau,
    max_step,
    noise,
):

    state = env.reset()

    for step_i in range(max_step):

        # make step
        state_tensor = torch.FloatTensor([state]).squeeze(-1).to(device)

        actor.eval()
        with torch.no_grad():
            action = actor(state_tensor).cpu().detach().numpy()

            action += noise.sample() * 2
        next_state, reward, done, _ = env.step(action)

        memory.add_step([state, action, next_state, reward])

        if done:
            break

        # do train step

        run_train_step(
            memory,
            actor,
            actor_target,
            critic,
            critic_target,
            actor_optim,
            critic_optim,
            device,
            writer,
            gamma=gamma,
            tau=tau,
        )

        state = next_state

    return


def main():

    env = gym.make("Pendulum-v0")

    config = Munch()

    # print env state and action info
    print("action space")
    print(env.action_space)

    print("state space")
    print(env.observation_space)

    # state_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # config
    config.actor_lr = 1e-3
    config.critic_lr = 1e-3
    device = torch.device("cuda:0")

    noise = OrnsteinUhlenbeckActionNoise(action_size)

    actor = Actor_ddpg_v2(state_size).to(device)
    actor_target = Actor_ddpg_v2(state_size).to(device)
    critic = Critic_ddpg_v2(state_size, action_size).to(device)
    critic_target = Critic_ddpg_v2(state_size, action_size).to(device)

    # init copy original network weights to target networks
    copy_network_params(actor, actor_target)
    copy_network_params(critic, critic_target)

    # setup optimizer
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    # setup ckpt saver
    config.suffix = "working_case"
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    outputdir = f"ckpt/train_ddpg/{timestamp}_{config.suffix}"
    os.makedirs(outputdir)

    ckpt_savedir = os.path.join(outputdir, "valid_max_acc_reward")
    os.makedirs(ckpt_savedir)

    saver = Saver(ckpt_savedir)

    # setup replay buffer
    config.memory_size = 10000
    memory = Memory(config.memory_size)

    # setup tensorboard writer
    log_dir = os.path.join(outputdir, "logs")
    os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    config.epi_num = 300
    config.run_valid_interval = 10

    config.tau = 0.001
    config.gamma = 0.99
    config.max_step = 1000

    # save config as file
    savepath = os.path.join(outputdir, "config.json")
    with open(savepath, "w") as fd:
        json.dump(vars(config), fd, indent=4, ensure_ascii=False)

    for i in tqdm(range(config.epi_num)):

        run_episode(
            i,
            env,
            actor,
            actor_target,
            critic,
            critic_target,
            actor_optim,
            critic_optim,
            writer,
            device,
            memory,
            saver,
            config.gamma,
            config.tau,
            config.max_step,
            noise,
        )

        if (i + 1) % config.run_valid_interval == 0:
            run_valid(i, env, actor, writer, saver, device, max_step=config.max_step)

    return


if __name__ == "__main__":

    main()
