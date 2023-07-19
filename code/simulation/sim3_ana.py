import pandas as pd
import numpy as np

print("=" * 20, "[ psi_d = 0.1 ]", "=" * 20)

out = pd.read_csv('output/out_sim3_K5_n100_iter100_psid01.csv')

print(out.describe(), "\n")


print("=" * 20, "[ psi_d = 9 ]", "=" * 20)

out = pd.read_csv('output/out_sim3_K5_n100_iter100_psid9.csv')

print(out.describe(), "\n")

out = pd.read_csv('output/out_sim3_K5_n100_iter500_psid9.csv')

print(out.describe(), "\n")


print("=" * 20, "[ psi_d = 36 ]", "=" * 20)

out = pd.read_csv('output/out_sim3_K5_n100_iter100_psid36.csv')

print(out.describe(), "\n")


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

