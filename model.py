import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import orthogonal


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, x, states = self(inputs, states, masks)
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
        return value, action, action_log_probs, states

    def evaluate_option(self, inputs, states, masks, action, option):
        value, x, states = self.intraOption[option].forward(inputs, states, masks)
        action_log_probs, dist_entropy = self.intraOption[option].dist.logprobs_and_entropy(x, actions)
        return value, action_log_probs, dist_entropy, states

    def evaluate_selection(self, inputs, states, masks, option):
        value, x, states = self.optionSelection.forward(inputs, states, masks)
        option_log_probs, dist_entropy = self.optionSelection.dist.logprobs_and_entropy(x, option)
        return option_log_probs

    def evaluate_termination(self, inputs, states, masks, termination, option):
        value, x, states = self.terminationOption[option].forward(inputs, states, masks)
        termination_log_probs, dist_entropy = self.terminationOption[option].dist.logprobs_and_entropy(x, termination)
        return termination_log_probs

class OptionCritic(FFPolicy):
    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1
    def __init__(self, num_options, num_inputs, action_space, use_gru):
        super(OptionCritic, self).__init__()
        self.num_options = num_options
        self.featureNN = CNNPolicy(num_inputs, use_gru)
        self.optionSelection = OptionPolicy(64, num_options, None, use_gru)
        self.intraOption = [OptionPolicy(64, None, action_space, use_gru) for i in range(num_options)]
        self.terminationOption = [OptionPolicy(64, 2, None, use_gru) for i in range(num_options)]
        # self.state_values = torch.zeros(num_states)

    def get_output(self, options, inputs, states, masks, deterministic= False):
        features = self.featureNN(inputs, states, masks)
        value_list, logits_list, states_list = [], [], []
        for j in range(len(options)):
                option = options[j].data[0]
                option_nn = self.intraOption[option]
                print(features)
                print(states)
                print(masks)
                value_j, x_j, states_j = option_nn.forward(features, states, masks)
                value_list.append(value_j)
                logits_list.append(logits_j)
                states_list.append(states_j)
        value = torch.cat(value_list)
        dist_inputs = torch.cat(logits_list)
        states = torch.cat(states_list)
        dist = self.intraOption[0].dist
        action = dist.sample(dist_inputs, deterministic = deterministic)
        action_log_probs, dist_entropy = dist.logprobs_and_entropy(dist_inputs, action)
        return value, action, action_log_probs, states 

    def get_option(self, inputs, states, masks, deterministic = False):
        features = self.featureNN(inputs, states, masks)
        value, x, states = self.optionSelection.forward(features, states, masks)
        option = self.optionSelection.dist.sample(x, deterministic = deterministic)
        option_log_probs, dist_entropy = self.optionSelection.dist.logprobs_and_entropy(x, option)
        return value, option, option_log_probs, states 

    def get_termination(self, options, inputs, states, masks, deterministic = False):
        features = self.featureNN(inputs, states, masks)

        value_list, logits_list, states_list = [], [], []
        for j in range(len(options)):
            option = options[j].data[0]
            term_nn = self.terminationOption[option]
            value_j, logits_j, states_j, = term_nn.forward(features[j:j + 1], states[j:j + 1], masks[j:j + 1])
            value_list.append(value_j)
            logits_list.append(logits_j)
            states_list.append(states_j)
        value = torch.cat(value_list)
        dist_inputs = torch.cat(logits_list)
        states = torch.cat(states_list)

        dist = self.terminationOption[0].dist
        actions = dist.sample(dist_inputs, deterministic = deterministic)
        action_log_probs, dist_entropy = dist.logprobs_and_entropy(dist_inputs, actions)
        return value, actions, action_log_probs, states

class OptionPolicy(FFPolicy):
    def __init__(self, num_inputs, num_outputs, action_space, use_gru):
        super(OptionPolicy, self).__init__()
        if num_outputs == None:
            if action_space.__class__.__name__ == "Discrete":
                num_outputs = action_space.n
                self.dist = Categorical(512, num_outputs)
            elif action_space.__class__.__name__ == "Box":
                num_outputs = action_space.shape[0]
                self.dist = DiagGaussian(512, num_outputs)
            else:
                raise NotImplementedError
        else:
            self.dist = Categorical(512, num_outputs)
        self.conv1 = nn.Conv2d(64, 64, 3, stride = 1)
        self.linear1 = nn.Linear(1600, 512)
        self.linear_critic = nn.Linear(512, 1)
        self.train()
        self.reset_parameters()

    
    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)
        x = x.view(-1, 1600)
        x = self.linear1(x)
        x = F.relu(x)
        value = self.linear_critic(x)

        return value, x, states

class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, use_gru):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        
    def forward(self, inputs, states, masks):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        return x


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        self.v_fc1 = nn.Linear(num_inputs, 64)
        self.v_fc2 = nn.Linear(64, 64)
        self.v_fc3 = nn.Linear(64, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(64, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(64, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.v_fc1(inputs)
        x = F.tanh(x)

        x = self.v_fc2(x)
        x = F.tanh(x)

        x = self.v_fc3(x)
        value = x

        x = self.a_fc1(inputs)
        x = F.tanh(x)

        x = self.a_fc2(x)
        x = F.tanh(x)

        return value, x, states
