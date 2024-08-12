from reg_20 import generate_reg_20, policy_L, policy_R

reg_20 = generate_reg_20(1.0)

left, right = [], []

# record the non-zero index of the policy
policy_L = [i for i in range(12) if policy_L[i] == 1]
policy_R = [i for i in range(12) if policy_R[i] == 1]

for i in range(6):
    left.append(reg_20[policy_L[i]])
    right.append(reg_20[policy_R[i]])
    
# given i, calculate inner product r_{ij} and r_{ik} for all j, k

i = 0
import numpy as np

left_inner_product = []
for j in range(6):
    for k in range(6):
        ri = np.array(left[i])
        rj = np.array(left[j])
        rk = np.array(left[k])
        
        rij = rj - ri
        rik = rk - ri
        
        left_inner_product.append(np.dot(rij, rik))
        
right_inner_product = []
for j in range(6):
    for k in range(6):
        ri = np.array(right[i])
        rj = np.array(right[j])
        rk = np.array(right[k])
        
        rij = rj - ri
        rik = rk - ri
        
        right_inner_product.append(np.dot(rij, rik))
    
# compare whether the inner products are the same, as set

left_set = set(left_inner_product)
right_set = set(right_inner_product)



print(left_set == right_set)

print(sum(left_inner_product), sum(right_inner_product))

