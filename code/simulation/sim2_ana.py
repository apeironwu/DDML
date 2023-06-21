import pandas as pd
import numpy as np


## `sim2_org_ini` ----------------------------------------------------------

# out = pd.read_csv('output/out_sim2_ora_ini_K5_n100_iter100.csv')

out = pd.read_csv('output/out_sim2_ora_ini_K5_n500_iter100.csv')

print(out.describe())







## `sim2_org_ini_rds` ----------------------------------------------------------

# out = pd.read_csv(
#     'output/out_sim2_ora_ini_K5_n100_rnp500_rds10.csv', 
#     sep=', ', engine='python'
# )

# print(out[out['rnd_ds'] == 130].describe())


# print(out[(out['M1'] > 6) | (out['M1'] < -2)])

# out_gp = out.dropna().groupby('rnd_np').mean().reset_index().describe()
# out_gp = out.groupby('rnd_np').median().reset_index().describe()

# print(out_gp)


## `sim2_org_ini_rdsu` ----------------------------------------------------------

#### 100 rep with 20 iter ----

# out = pd.read_csv(
#     'output/out_sim2_ora_ini_K5_n100_rnp100_rdsu20.csv', 
#     sep=', ', engine='python'
# )

# print(out.describe())

#### 200 rep with 50 iter ----

# out = pd.read_csv(
#     'output/out_sim2_ora_ini_K5_n100_rnp200_rdsu50.csv', 
#     sep=', ', engine='python'
# )

# filter = (out['M50'] < -2) | (out['M50'] > 6) | (out["NaN"] == True)

# print(out[-filter].describe())

