import torch
import torch.nn as nn
import torch.nn.functional as F


class DCNNet(nn.Module):
    """Deep Convolutional Neural Network for Blokus"""

    def __init__(self, obs_shape, n_actions, board_size=8):
        super().__init__()
        # game params
        self.action_size = n_actions  # 30433
        self.num_players = 2  # 4
        self.board_size = board_size  # 8

        # Neural Net
        # Input have shape (batch_size, players*2, board_x, board_y)
        # Output have shape (batch_size, action_size)
        self.conv1 = nn.Conv2d(obs_shape[2], 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        
        
        self.fc1 = nn.Linear(
            128 * board_size * board_size,
            128,
        )
        self.fc_bn1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 64)
        self.fc_bn2 = nn.LayerNorm(64)

        self.fc3 = nn.Linear(64, self.action_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # print(f'x shape: {x.shape}')
        x = F.relu(
            self.bn1(self.conv1(x))
        ) 
        x = F.relu(
            self.bn2(self.conv2(x))
        )  # batch_size x num_channels x board_x x board_y
        x = F.relu(
            self.bn3(self.conv3(x))
        )  # batch_size x num_channels x (board_x-2) x (board_y-2)
        x = F.relu(
            self.bn4(self.conv4(x))
        )  # batch_size x num_channels x (board_x-4) x (board_y-4)
        
        # print(f'x after conv: {x.shape}')
        
        x = x.reshape(-1, 128 * (self.board_size) * (self.board_size))

        # print(f'x after reshape: {x.shape}')
        
        x = F.dropout(
            F.relu(self.fc_bn1(self.fc1(x))),
            p=0.1,
            training=self.training,
        )  
        
        x = F.dropout(
            F.relu(self.fc_bn2(self.fc2(x))),
            p=0.1,
            training=self.training,
        )  # batch_size x 512
        # print(f'x after fc: {x.shape}')
        out = self.fc3(x)  # batch_size x action_size

        return out


class ResNet(nn.Module):
    """Residual Neural Network for Blokus"""

    def __init__(self, obs_shape, n_actions, board_size=8):
        super().__init__()

        # game params
        
        self.action_size = n_actions  
        self.num_players = 2  
        self.board_size = board_size 

        self.conv1 = nn.Conv2d(obs_shape[2], 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # A series of residual blocks
        self.res_blocks = nn.Sequential(
            *[self._build_res_block(64) for _ in range(5)]
        )

        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_out = nn.Linear(2 * self.board_size * self.board_size, self.action_size)


    def _build_res_block(self, channel_in):
        block = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_in),
            nn.ReLU(),
            nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_in),
        )
        return block

    def _residual(self, x, residual_function):
        return F.relu(x + residual_function(x))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self._residual(x, self.res_blocks)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(policy.size(0), -1)

        policy = self.policy_out(policy)

        return policy


class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(DQN, self).__init__()
        # observations : (10, 10, 5)
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),              
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),             
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),              
            nn.ReLU(),
            nn.Flatten()
        )
        
        dummy = torch.zeros(1, obs_shape[2], obs_shape[0], obs_shape[1])
        conv_out_size = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_actions)
        )
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  
        x = self.conv(x)
        x = self.fc(x)
        return x

