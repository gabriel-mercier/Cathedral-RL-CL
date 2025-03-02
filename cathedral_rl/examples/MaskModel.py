from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Dict

class SimpleTorchActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        assert isinstance(obs_space, Dict), "Observation space must be a dictionary"
        
        # Create the base model for the actual observation part (not the mask)
        self.base_model = FullyConnectedNetwork(
            obs_space["observation"], 
            action_space,
            num_outputs,
            model_config,
            name + "_base"
        )
        
        # Whether to mask invalid actions
        self.no_masking = model_config.get("custom_model_config", {}).get("no_masking", False)
    
    def forward(self, input_dict, state, seq_lens):
        # Extract action mask from observation
        action_mask = input_dict["obs"]["action_mask"]
        
        # Forward pass through base model using the observation component
        wrapped_obs = {"obs": input_dict["obs"]["observation"]}
        model_out, self._value_out = self.base_model(wrapped_obs, state, seq_lens)
        
        # Apply action mask by setting logits of invalid actions to a large negative value
        if not self.no_masking:
            inf_mask = torch.clamp(torch.log(action_mask), min=-1e38)
            model_out = model_out + inf_mask
            
        return model_out, state
    
    def value_function(self):
        return self.base_model.value_function()