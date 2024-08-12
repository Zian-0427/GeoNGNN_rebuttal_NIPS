train_time = []
train_mem = []
val_time = []
val_mem = []


def cal_mean(lst):
    import numpy as np
    ary = np.array(lst)
    
    mean = ary.mean()
    std = ary.std()
    ary = ary[(ary < mean + 1 * std) & (ary > mean - 1 * std)]
    
    mean = ary.mean()
    std = ary.std()
    ary = ary[(ary < mean + 1 * std) & (ary > mean - 1 * std)]
    
    mean = ary.mean()
    std = ary.std()
    ary = ary[(ary < mean + 1 * std) & (ary > mean - 1 * std)]
    
    print("neat ary: ", ary)
    
    return ary.mean()


def print_eff():

    if len(train_time) != 0:
        print('Train time: ', train_time)
    if len(train_mem) != 0:
        print('Train mem: ', train_mem)
    if len(val_time) != 0:
        print('Val time: ', val_time)
    if len(val_mem) != 0:
        print('Val mem: ', val_mem)
        
    print("\n\n")
    
    if len(train_time) != 0:
        print("Mean train time: ", cal_mean(train_time))
    if len(train_mem) != 0:
        print("Mean train mem: ", max(train_mem))
    if len(val_time) != 0:
        print("Mean val time: ", cal_mean(val_time))
    if len(val_mem) != 0:
        print("Mean val mem: ", max(val_mem))