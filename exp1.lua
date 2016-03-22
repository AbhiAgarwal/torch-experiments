require 'torch'

-- Basic example 1 from:
-- http://torch.ch/docs/five-simple-examples.html

torch.manualSeed(1234)

-- choose a dimension
N = 5

-- create a random NxN matrix
A = torch.rand(N, N)

-- make it symmetric positive
A = A * A:t()

-- make it definite
A:add(0.001, torch.eye(N))

-- add a linear term
b = torch.rand(N)

-- create the quadratic form
function J(x)
    return 0.5 * x:dot(A * x) - b:dot(x)
end

print(J(torch.rand(N)))