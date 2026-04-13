import torch
import torch.nn as nn

# s = torch.tensor(0,dtype=torch.float32)
# for i in range(1000):
#     s += torch.tensor(0.01,dtype=torch.float32)
# print(s)

# s = torch.tensor(0,dtype=torch.float16)
# for i in range(1000):
#     s += torch.tensor(0.01,dtype=torch.float16)
# print(s)

# s = torch.tensor(0,dtype=torch.float32)
# for i in range(1000):
#     s += torch.tensor(0.01,dtype=torch.float16)
# print(s)

# s = torch.tensor(0,dtype=torch.float32)
# for i in range(1000):
#     x = torch.tensor(0.01,dtype=torch.float16)
#     s += x.type(torch.float32)
# print(s)


class ToyModel(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("After fc1 ", x.dtype)
        x = self.ln(x)
        print("After layer norm: ", x.dtype)
        x = self.fc2(x)
        print("After fc2:", x.dtype)
        return x

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ToyModel(8, 4).to(device)
    x = torch.randn(2, 8, device=device)
    print(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(x)
    
    print("final logits dtype", y.dtype)
    loss = y.sum()
    print("loss dtype", loss.dtype)
    loss.backward()
    for name, param in model.named_parameters():
        print(name, "grad dtype:", param.grad.dtype)

if __name__ == "__main__":
    main()