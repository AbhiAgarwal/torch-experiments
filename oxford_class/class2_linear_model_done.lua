require 'torch'
require 'optim'
require 'nn'

-- {corn, fertilizer, insecticide}
data = torch.Tensor{
    {40,  6,  4},
    {44, 10,  4},
    {46, 12,  5},
    {48, 14,  7},
    {52, 16,  9},
    {58, 18, 12},
    {60, 22, 14},
    {68, 24, 20},
    {74, 26, 21},
    {80, 32, 24}
}

ninputs = 2;
noutputs = 1;

model = nn.Sequential()
model:add(nn.Linear(ninputs, noutputs))

loss_criterion = nn.MSECriterion()
x, dl_dx = model:getParameters()

feval = function(x_new)
    if x ~= x_new then
        x:copy(x_new)
    end

    _nidx_ = (_nidx_ or 0) + 1
    if _nidx_ > (#data)[1] then _nidx_ = 1 end

    local sample = data[_nidx_]
    local inputs = sample[{{2, 3}}]
    local target = sample[{{1}}]

    dl_dz.zero()

    local loss_x = loss_criterion:forward(model:forward(inputs), target)
    model:backwards(inputs, loss_criterion:backward(model.output, target))
end
