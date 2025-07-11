import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import difflg_model
import pandas as pd

class DiffLogicGateController():
    def load_model(self, norm_vec_path='normalization_vec_a.csv'):
        # Load the model from the file
        self.model = difflg_model.Model(
            difflg_model.SEED,
            difflg_model.GATE_ARCHITECTURE,
            difflg_model.INTERCONNECT_ARCHITECTURE,
            difflg_model.NUMBER_OF_CATEGORIES,
            difflg_model.INPUT_SIZE
        )
        path_to_model = os.path.join(os.path.dirname(__file__), 'difflg_cartpole.pth')
        self.model.load_state_dict(torch.load(path_to_model))
        norm_vec_path = os.path.join(os.path.dirname(__file__), norm_vec_path)
        if os.path.exists(norm_vec_path):
            self.norm_vec = pd.read_csv(norm_vec_path, header=None).values.flatten()
            self.norm_vec = np.array(self.norm_vec, dtype=np.float32)
        else:
            print(f"File {norm_vec_path} does not exist.")
        self.model.eval()
    
    def predict(self, input_data):
        # Ensure input has 7 elements
        if len(input_data) != 7:
            raise ValueError("Input data must have exactly 7 elements.")
        # Ensure input is a numpy array
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data, dtype=np.float32)
            input_data = input_data * self.norm_vec
        
        x_np = input_data.view(np.uint32)
        bits_list = []
        for val in x_np:
            bits = np.unpackbits(
                np.array([val], dtype=np.uint32).view(np.uint8)
            )
            bits_list.append(bits)

        # Concatenate to a single array of shape [224]
        bits_concat = np.concatenate(bits_list, axis=0)

        # Convert back to torch tensor if needed
        bits_tensor = torch.from_numpy(bits_concat)
        x = bits_tensor.type(torch.float32)
        # Ensure input match the model's expected input size
        if x.shape[0] != difflg_model.INPUT_SIZE:
            raise ValueError(f"Input data must have {difflg_model.INPUT_SIZE} elements, got {x.shape[0]}.")
        # Forward pass through the model
        with torch.no_grad():
            output = self.model(x.unsqueeze(0))  # Add batch dimension
            output = torch.clamp(output, -1.0, 1.0)  # Clamp output to [-1, 1]
        return output.item()


#Example usage:

controller = DiffLogicGateController()
controller.load_model()
test_input = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)  # Example input
output = controller.predict(test_input)
print(f"Predicted output: {output}")
