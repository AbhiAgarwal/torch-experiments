local y = torch.Tensor({5, 25, 22, 18})
local X = torch.Tensor({{1, 100, 2}, {1, 50, 42}, {1, 45, 31}, {1, 60, 35}})
local theta = torch.Tensor({{1}, {0}, {0.5}})

local y_hat = X * theta

print(y_hat)