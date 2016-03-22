Lua basics

[Lua cheatsheet](http://tylerneylon.com/a/learn-lua/)

1. Why is the local keyword important? (hint: default variable scope is not local)
2. What is the difference between a.f() and a:f()? (hint: one implicitly adds self as the first argument)
3. What does require do, and why do we sometimes capture its return value but sometimes not? (this is a way to isolate things into namespaces to prevent naming conflicts, related to the answer to why we need to use local)
4. What is the Lua equivalent of a list object? of a dictionary? How do you iterate over each
of these?

Tensor: generalize vectors (1-d arrays) and matrices (2-d arrays) to n-dimensions. Torch has a flexible and efficient class for storing and manipulating these objects.

```lua
local t = torch.Tensor(10,10)
local t2 = torch.Tensor(10,10)
t3 = t + t2
```

vs

```lua
local t = torch.Tensor(10,10)
local t2 = torch.Tensor(10,10)
t:add(t2)
```

The second one saves us memory as we don't have to create a new variable t3. This becomes important when the size of the tensors grow.
