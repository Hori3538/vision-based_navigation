import torch

def onehot_encoding(decoded_abst_pose: torch.Tensor) -> torch.Tensor:
    encoded_abst_pose: torch.Tensor = torch.zeros(len(decoded_abst_pose), 9, dtype=torch.float32)
    for i in range(len(decoded_abst_pose)):
        for j in range(decoded_abst_pose.shape[1]):
            encoded_abst_pose[i, j*3 + 1 + decoded_abst_pose[i, j].to(torch.long)] = 1

    return encoded_abst_pose

def onehot_decoding(encoded_abst_pose: torch.Tensor) -> torch.Tensor:
    decoded_abst_pose: torch.Tensor = torch.zeros(len(encoded_abst_pose), 3, dtype=torch.int8)
    for i in range(len(encoded_abst_pose)):
        for j in range(encoded_abst_pose.shape[1]):
            if encoded_abst_pose[i, j] == 1:
                decoded_abst_pose[i, j // 3] = j % 3 - 1

    return decoded_abst_pose

def create_onehot_from_output(outputs: torch.Tensor) -> torch.Tensor:
    onehot_outputs = torch.zeros(outputs.shape, dtype=torch.int8)
    for i in range(len(outputs)):
        onehot_outputs[i, [outputs[i, :3].max(0).indices,
            outputs[i, 3:6].max(0).indices+3, outputs[i, 6:].max(0).indices+6]] = 1

    return onehot_outputs
# def create_onehot_from_output(outputs: torch.Tensor) -> torch.Tensor:
#     onehot_outputs = torch.zeros(outputs.shape, dtype=torch.int8)
#     onehot_outputs[:, [outputs[:, :4].max(0).indices,
#         outputs[:, 4:].max(0).indices+4]] = 1
#
#     return onehot_outputs

