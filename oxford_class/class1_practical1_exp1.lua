local t = torch.Tensor(10,10)
local t2 = torch.Tensor(10,10)
t3 = t + t2

local t = torch.Tensor(10,10)
local t2 = torch.Tensor(10,10)
t:add(t2)

-- extract the middle col from t
-- should print the 1-d tensor: 2,5,8
local t = torch.Tensor({{1,2,3},{4,5,6},{7,8,9}})
local col = t:narrow(2,2,1)

-- col:resize(1,3)

print(col)