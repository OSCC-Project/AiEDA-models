from models.DDPG.DDPG import DDPG
from models.GNN.gcn import GCN_NET, GCN
from models.GNN.vae import VAE
from env.env import INTER_ENV, EnvTO
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
from torchviz import make_dot
from copy import deepcopy

from util import *
from sklearn.manifold import TSNE
import gym

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed(seed) #gpu
    torch.cuda.manual_seed_all(seed) #all gpu
    # os.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    env = gym.make('Pendulum-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = [-env.action_space.high[0], env.action_space.high[0]]

    set_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ''' 
        VAE model
    # '''
    # vae = VAE()
    # loss_list = []
    # best_acc = 0.0
    # path_vae = "vae.pt"
    # for i in range(300):
    #     loss = vae.train()
    #     acc = vae.accuracy_rate()
    #     if acc > best_acc:
    #         best_acc = acc
    #         torch.save(vae, path_vae)
    #     loss_list.append(loss)
    # plt.figure(4)
    # plt.plot(np.arange(len(loss_list)), loss_list)
    # plt.savefig("loss_list.png")
    # best_vae = torch.load(path_vae)
    # vae = best_vae
    # acc = vae.accuracy_rate()
    # print(acc)
    # print(vae)

    # RL Setting
    MAX_EPISODE = 100
    MAX_STEP = 50
    update_every = 50
    batch_size = 100

    gnn_layer = 2
    gnn_input_dim = 5
    gnn_hid_dim = 64
    gnn_output_dim = 32

    ''' 
        GCN model
    '''
    gcn = GCN(gnn_layer, gnn_input_dim, gnn_hid_dim, gnn_output_dim)
    next_state_transition = gcn.next_state_transition()
    embedding = gcn.embedding()
    t_SNE(embedding.cpu().detach().numpy())
    true_label = gcn.data.y

    reward_list = []
    step_reward_list = []
    ddpg = get_rl_model(gnn_output_dim)
    print(ddpg)

    # env
    # env = EnvTO(embedding, true_label)
    env = INTER_ENV(embedding)

    pbar_eps = tqdm(range(100), unit='batch')
    pbar_step = tqdm(range(10), unit='step')
    last_reward = 0.0

    best_reward = -100

    for episode in pbar_eps:
        # state = env.reset()
        state = embedding
        ep_reward = 0
        for n in pbar_step:
            action = ddpg.get_action(state, ddpg.act_noise)
            # MyConvNetVis = make_dot(action)
            # MyConvNetVis.format = "pdf"
            # MyConvNetVis.directory = "data11"
            # MyConvNetVis.view()
                    
            # next_state, reward, done = env.step(action)
            next_state, reward, done = env.step(action, next_state_transition)
            if reward > best_reward:
                best_reward = reward
                env.saveDef()
            reward2 = [reward for i in range(state.size()[0])]
            done2 = [done for i in range(state.size()[0])]
            trajectories = zip(state.detach().numpy(), action,
                               reward2, next_state, done2)
            # trajectories = zip(state, action, reward2, next_state, done2)
            for each in trajectories:
                (s, act, rew, next_s, d) = tuple(each)
                ddpg.replay_buffer.store(s, act, rew, next_s, d)

            # if episode >= 10 and j % update_every == 0:
            #     for _ in range(update_every):
            batch = ddpg.replay_buffer.sample_batch(state.size()[0])
            # batch = ddpg.replay_buffer.sample_batch(batch_size)
            ddpg.update(data=batch)

            step_reward_list.append(reward)
            with open("reward2.txt","a") as f:
                f.write(str(reward))
                f.write(" ")

            ep_reward += reward
            pbar_step.set_description('step: %d' % (n))
         # report
        pbar_eps.set_description('epoch: %d' % (episode))
        if ep_reward > last_reward:  # check whether gain improvement on validation set
            best_policy = deepcopy(ddpg)  # save the best policy
        last_reward = ep_reward

        # print('Episode:', episode, 'Reward:%i' % int(ep_reward))
        reward_list.append(ep_reward)
        with open("reward2.txt","a") as f:
            f.write("\n")

    # for i_episode in range(1, 1000):
    #     embd = gcn.embedding()
    #     pred = best_policy.get_action(embd, 0)
    #     gcn.train(pred)

    plt.figure(1)
    plt.plot(np.arange(len(reward_list)), reward_list)
    plt.savefig("reward_list.png")
    plt.figure(2)
    plt.plot(np.arange(len(step_reward_list)), step_reward_list)
    plt.savefig("step_reward_list.png")
    # plt.show()

def t_SNE(output):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(output)
    
    for i in range(output.shape[0]):
        if i < 122 :
            plt.scatter(result[i, 0], result[i, 1], cmap=plt.cm.Spectral, color = "r")
        else:
            plt.scatter(result[i, 0], result[i, 1], cmap=plt.cm.Spectral, color = "b")
    plt.savefig("tsne.png")
    plt.show()


def run_gcn(dataset="Cora"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # edges, features = init_data()
    edges, features, labels, number_classes, idx_train, idx_val, idx_test = load_data()

    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    # dataset = Planetoid(path, dataset)
    # number_classes = dataset.num_classes

    # data1 = dataset[0].to(device)

    # self define data
    data = Data(x=features, edge_index=edges.T, y=labels)

    # GNN model
    max_layer = 3
    input_dim = data.num_node_features
    hid_dim = 16
    ouput_dim = number_classes
    model = GCN_NET(max_layer, input_dim, hid_dim, ouput_dim).to(device)
    print(model)
    adj = to_dense_adj(data.edge_index).numpy()[0]
    norm = np.array([np.sum(row) for row in adj])
    adj = (adj / norm).T

    # data = Data(x=features, edge_index=edges.T)
    # graph_showing(data)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    embedding = model(data)

    # MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    MyConvNetVis = make_dot(embedding)
    MyConvNetVis.format = "png"
    MyConvNetVis.directory = "data"
    MyConvNetVis.view()

    true_label = data.y
    return embedding, true_label


def get_rl_model(input_dim):  # Try different models
    act_bound = [-float(10), float(10)]

    ddpg = DDPG(input_dim, 1, act_bound)
    return ddpg


if __name__ == '__main__':
    main()
