import os
import torch
import torch.nn.functional as F
from utils import *
from model import *
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class DrQV2:
    def __init__(self, action_space, args, stddev_schedule, img_size, feature_dim, encoder_layers, encoder_filters, frame_stack=3):

        self.gamma = args.gamma
        self.tau = args.tau

        self.feature_dim = feature_dim
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.frame_stack = frame_stack
        self.agent_image_size = img_size
        self.encoder_layers = encoder_layers
        self.encoder_filters = encoder_filters
        self.device = torch.device("cuda")
        self.stddev_schedule = stddev_schedule
        agent_obs_shape = (self.frame_stack, self.agent_image_size, self.agent_image_size)
        self.encoder = Encoder(agent_obs_shape).to(self.device)
        self.actor = Actor(self.encoder.repr_dim, action_space.shape, feature_dim,
                           hidden_dim = args.hidden_size).to(self.device)

        self.critic = Critic(self.encoder.repr_dim, action_space.shape, feature_dim,
                             hidden_dim = args.hidden_size).to(self.device)
        self.critic_target = Critic(self.encoder.repr_dim, action_space.shape,
                                    feature_dim, hidden_dim = args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

        # data augmentation
        self.centercrop = center_crop_image(pad=8)
        self.random_pad_resize = random_pad_resize()

    def select_action(self, state, step, evaluate=False):
        obs = self.encoder(state.unsqueeze(0))
        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)

        with torch.no_grad():
            if evaluate is False:             
                action = dist.sample(clip=None)        
            else:
                action = dist.mean
        return action.cpu().data.numpy()[0] 

    def update_critic(self, state_batch_aug, action_batch, reward_batch, gamma, next_state_batch_aug, step):
        with torch.no_grad():
            next_state_batch_aug = self.encoder(next_state_batch_aug)
            stddev = schedule(self.stddev_schedule, step)
            dist = self.actor(next_state_batch_aug, stddev)
            next_action = dist.sample(clip=0.3)
            target_Q1, target_Q2 = self.critic_target(next_state_batch_aug, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + (gamma * target_V)
        
        Q1, Q2 = self.critic(state_batch_aug, action_batch)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        return Q1, Q2, target_Q
    
    def update_actor(self, state_batch_aug, step):
        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(state_batch_aug, stddev)
        action = dist.sample(clip=0.3)
        Q1, Q2 = self.critic(state_batch_aug, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        return actor_loss
    
    def update_parameters_gpu(self, memory, batch_size, updates, step):
        state_batch, action_batch, reward_batch, next_state_batch, _ = memory.sample_gpu(batch_size=batch_size)
        state_batch = state_batch.squeeze(dim=1)
        next_state_batch = next_state_batch.squeeze(dim=1)
        for _ in range(1):
            state_batch_aug = self.centercrop(state_batch)
            next_state_batch_aug = self.centercrop(next_state_batch)            
            state_batch_aug = self.encoder(state_batch_aug)
            Q1, Q2, target_Q = self.update_critic(state_batch_aug, action_batch, reward_batch, self.gamma, next_state_batch_aug, step)
            actor_loss = self.update_actor(state_batch_aug.detach(), step)
            if updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)
        return F.mse_loss(Q1,target_Q).item(), F.mse_loss(Q2,target_Q).item(), actor_loss.item()  
    
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/drqv2_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_opt.state_dict(),
                    'policy_optimizer_state_dict': self.actor_opt.state_dict(),
                    'encoder_state_dict': self.encoder.state_dict(),
                    'encoder_optimizer_state_dict': self.encoder_opt.state_dict()}, ckpt_path)

    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.actor.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_opt.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor_opt.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.encoder_opt.load_state_dict(checkpoint['encoder_optimizer_state_dict'])

            if evaluate:
                self.actor.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.actor.train()
                self.critic.train()
                self.critic_target.train()
