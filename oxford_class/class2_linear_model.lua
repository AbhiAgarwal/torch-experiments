-- neural network package linear regression

require 'torch'
require 'optim'
require 'nn'

-- Predict the amount of corn produced, given the amount of fertilizer
-- and insecticide used. In other words: fertilizer & insecticide are our
-- two input variables, and corn is our target value.

-- Training data
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

-- The input to our model will be fertilizer and insecticide, and the
-- output will be amount of corn.

ninputs = 2;
noutputs = 1;

-- Defining model
-- The linear model must be held in a container. A sequential container
-- is appropriate since the outputs of each module become the inputs of 
-- the subsequent module in the model. In this case, there is only one
-- module. In more complex cases, multiple modules can be stacked using
-- the sequential container.

model = nn.Sequential()

model:add(nn.Linear(ninputs, noutputs))

-- Loss function
-- We use Mean Square Error here.

loss_criterion = nn.MSECriterion()

-- Training the model

-- To minimize the loss defined above, using the linear model defined
-- in 'model', we follow a stochastic gradient descent procedure (SGD).

x, dl_dx = model:getParameters()

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. x is the vector of trainable weights,
-- which, in this example, are all the weights of the linear matrix of
-- our model, plus one bias.

feval = function(x_new)
    -- set x to x_new, if different
    -- x_new will typically always point to x
    if x ~= x_new then
        x:copy(x_new)
    end

    -- select new training set
    -- # preceding a variable returns the length of that variable
    _nidx_ = (_nidx_ or 0) + 1
    if _nidx_ > (#data)[1] then _nidx_ = 1 end

    local sample = data[_nidx_]
    local inputs = sample[{{2, 3}}]
    local target = sample[{{1}}]

    dl_dz.zero()

    -- evaluate the loss function and its derivative wrt x, for that sample
    local loss_x = loss_criterion:forward(model:forward(inputs), target)
    model:backwards(inputs, loss_criterion:backward(model.output, target))
end


