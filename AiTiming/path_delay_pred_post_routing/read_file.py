import yaml
import torch
import hashlib


class CircuitParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.capacitance_list = []
        self.slew_list = []
        self.resistance_list = []
        self.incr_list = []
        self.point_list = []

    def load_data(self):
        """Load YAML data from the file."""
        try:
            with open(self.file_path, 'r') as f:
                self.data = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading file: {e}")

    def parse_data(self):
        """Parse nodes, net arcs, and instance arcs in order."""
        for key, value in self.data.items():
            if key.startswith("node_"):
                # Parse node
                self.capacitance_list.append(value.get("Capacitance", 0))
                self.slew_list.append(value.get("slew", 0))
                self.resistance_list.append(0)  # Default R value for nodes
                point = value.get("Point", "")
                if point:
                    self.point_list.append(point.split("(")[0].strip())  # Extract portion before '('
            elif key.startswith("net_arc_"):
                self.incr_list.append(value.get("Incr", 0))
                for edge_key, edge_value in value.items():
                    if edge_key.startswith("edge_"):
                        self.capacitance_list.append(edge_value.get("wire_C", 0))
                        self.slew_list.append(edge_value.get("to_slew", 0))
                        self.resistance_list.append(edge_value.get("wire_R", 0))
                        wire_from_node = edge_value.get("wire_from_node", "")
                        if wire_from_node:
                            self.point_list.append(wire_from_node.replace(" ", ""))  # Remove spaces
            elif key.startswith("inst_arc_"):
                self.incr_list.append(value.get("Incr", 0))
        self.point_list = list(dict.fromkeys(self.point_list))

    def get_combined_tensor(self):
        """Combine all lists into a single 2D tensor with each list as a row."""
        combined_data = [
            self.capacitance_list,
            self.slew_list,
            self.resistance_list
        ]
        tensor = torch.tensor(combined_data, dtype=torch.float32)
        return tensor

    def get_incr_tensor(self):
        """Get the tensor of Incr values and calculate the sum."""
        incr_tensor = torch.tensor(self.incr_list, dtype=torch.float32)
        incr_sum = incr_tensor.sum().item()
        return incr_tensor, incr_sum

    @staticmethod
    def pad_tensors(tensor_list, max_length):
        """Pad all tensors in the list to the max length."""
        padded_tensors = []
        for tensor in tensor_list:
            padded = torch.nn.functional.pad(tensor, (0, max_length - tensor.size(1)), "constant", 0)
            padded_tensors.append(padded)
        return padded_tensors


    def generate_hash(self):
        """Generate a hash for the concatenated unique strings."""
        concatenated = "".join(self.point_list)
        hash_object = hashlib.md5(concatenated.encode())
        return hash_object.hexdigest()

