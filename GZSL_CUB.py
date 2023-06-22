import numpy as np
import scipy
from scipy import spatial
import load_data as ld

X_tr_ip, X_te_ip, S_tr_ip, S_te_ip, Y_te_ip, test_cls_ip, Y_tr_ip, S_te_all_ip = ld.cub(int_proj=True)
