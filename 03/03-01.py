# Implementing the attention mechanism

import torch

# Basic implementation of the softmax function for normalizing the attention scores
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

# Input
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# Compute attention weights and the context vector for input 2

query = inputs[1]                            #1
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

res = 0.
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
print(res)
print(torch.dot(inputs[0], query))

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())


# Computed the normalized attention weights,
attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())


# Final step
query = inputs[1]         #1
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2_naive[i]*x_i
print(context_vec_2)


# Now calculate attention weights and context vectors for all inputs.
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

# The same, but using matrix multiplication instead of a for loop
attn_scores = inputs @ inputs.T
print(attn_scores)

# Step 2, normalize weights
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# Veryfy. All rows should sum 1
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# Step 3, use the attention weights to compute all context vectors via matrix multiplication:
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)


# Double-check that the code is correct by comparing the second row with the context vector
print("Previous 2nd context vector:", context_vec_2)

print()