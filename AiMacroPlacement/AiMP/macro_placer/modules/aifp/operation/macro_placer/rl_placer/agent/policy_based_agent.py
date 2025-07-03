import torch
import logging
from torch.distributions import Categorical

class PolicyBasedAgent:
    def __init__(
        self,
        model,
        continues_action=False):

        self._model = model
        self._continous_action = continues_action

    @property
    def is_shared(self):
        return self._model.is_shared

    def sample(self, *obs):
        """ Define the sampling process. This function returns the action according to action distribution.
        
        Args:
            obs: observations, same as params in self._model.forward
        Returns:
            value (numpy ndarray): value, shape([batch_size, 1])
            action (numpy ndarray): action, shape([batch_size] + action_shape)
            action_log_probs (numpy ndarray): action log probs, shape([batch_size])
            action_entropy (numpy ndarray): action entropy, shape([batch_size])
        """
        if self._continous_action:
            raise NotImplementedError
        else:
            with torch.no_grad():
                if self._model.is_shared: # actor and critic share network
                    action_logits, value = self._model.policy_value(*obs)
                else:
                    action_logits = self._model.policy(*obs)
                    value = self._model.value(*obs)
                
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                action_log_probs = dist.log_prob(action)
                action_entropy = dist.entropy()

                value = value.squeeze(-1).cpu().detach().numpy() # squeeze last dim
                action = action.cpu().detach().numpy()
                action_log_probs = action_log_probs.cpu().detach().numpy()
                action_entropy = action_entropy.cpu().detach().numpy()
        return value, action, action_log_probs, action_entropy

    def to_device(self, device):
        self._model.to(device)
        self._model.set_device(device)
        self._model._device = device
        print('setting device')

    def predict(self, *obs):
        """ use the model to predict action deterministicly
        Args:
            obs: observations, same as params in self._model.forward
        Returns:
            action (torch tensor): action, shape([batch_size] + action_shape),
            noted that in the discrete case we take the argmax along the last axis as action
        """
        if self._continous_action:
            raise NotImplementedError
        else:
            with torch.no_grad():
                if self._model.is_shared:
                    action_logits, _ = self._model.policy_value(*obs)
                else:
                    action_logits = self._model.policy(*obs)

                dist = Categorical(logits=action_logits)
                action = dist.probs.argmax(dim=-1, keepdim=True)
                action = action.squeeze().cpu().detach().numpy()
        return action

    def value(self, *obs):
        with torch.no_grad():
            if self._model.is_shared:
                _, value = self._model.policy_value(*obs)
            else:
                value = self._model.value(*obs)
            value = value.squeeze(-1).cpu().detach().numpy()
        return value
    
    def policy_value_forward(self, *obs):
        """ used in training process, with grad
        Args:
            obs: observations, same as params in self._model.forward
        Returns:
            action_logits:
            value:
        """
        if self._model.is_shared:
            return self._model.policy_value(*obs)
        else:
            return self._model.policy(*obs), self._model.value(*obs)
        
    def save_model(self, save_path, last_iteration):
        torch.save({'last_iteration': last_iteration, 'model_state_dict': self._model.state_dict()}, save_path)
        logging.info('================== model saved to {} ====================='.format(save_path))

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        last_iteration = checkpoint['last_iteration']
        logging.info('================== model loaded from {} ====================='.format(load_path))
        logging.info('================== last iteration: {} ===================='.format(last_iteration))
        return last_iteration

    def get_params(self):
        if self._model.is_shared:
            return self._model.get_params()
        else:
            return self._model.get_policy_params(), self._model.get_value_params()