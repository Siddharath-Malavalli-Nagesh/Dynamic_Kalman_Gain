import torch
ckpt = torch.load('best_student_nclt.pt', map_location='cpu', weights_only=False)
print(ckpt['H.weight'])
print(ckpt['H.bias'])