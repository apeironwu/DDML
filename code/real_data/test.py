import numpy as np
import pickle

np.random.seed(2024)

vec1 = np.random.randn(3)
vec2 = np.random.randn(3)

print(vec1, vec2)

pickle.dump([vec1, vec2], open("code/real_data/nui_par_est.pydata", "wb"))

vec3, vec4 = pickle.load(open("code/real_data/nui_par_est.pydata", "rb"))

print(vec3, vec4)



