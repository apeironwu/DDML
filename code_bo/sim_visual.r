
# loading packages --------------------------------------------------------

library(tidyverse)
library(patchwork)



tib_v0 <- read_csv("output/out_sim7_v0.csv")

tib_v0 %>% 
  rename(
    Avg = avg, 
    IVWAvg = var_avg, 
    M1s = m1_single
  ) %>% 
  pivot_longer(
    cols = Avg:M2,
    names_to = "method", 
    values_to = "est"
  ) %>% 
  ggplot() +
  geom_boxplot(
    mapping = aes(x = method, y = est)
  )


tib_v1 <- read_csv("output/out_sim7_v1.csv")

tib_v1 %>% 
  rename(
    Avg = avg, 
    IVWAvg = var_avg, 
    M1s = m1_single
  ) %>% 
  pivot_longer(
    cols = Avg:M2,
    names_to = "method", 
    values_to = "est"
  ) %>% 
  ggplot() +
  geom_boxplot(
    mapping = aes(x = method, y = est)
  )




tib_v2 <- read_csv("output/out_sim7_v2.csv")


plt_traj_v2 <- tib_v2 %>% 
  rename(
    Avg = avg, 
    IVWAvg = var_avg, 
    M1s = m1_single
  ) %>% 
  pivot_longer(
    cols = Avg:M2,
    names_to = "method", 
    values_to = "est"
  ) %>% 
  group_by(method) %>% 
  mutate(
    idx = factor(rnd) %>% as.integer()
  ) %>% 
  mutate(
    cum_avg = cumsum(est) / idx
  ) %>% 
  ggplot() + 
  geom_line(
    mapping = aes(
      x = idx, y = cum_avg, 
      group = method, color = method
    )
  ) + 
  theme_bw()
  

plt_box_v2 <- tib_v2 %>% 
  rename(
    Avg = avg, 
    IVWAvg = var_avg, 
    M1s = m1_single
  ) %>% 
  pivot_longer(
    cols = Avg:M2,
    names_to = "method", 
    values_to = "est"
  ) %>% 
  ggplot() +
  geom_boxplot(
    mapping = aes(x = method, y = est, fill = method), 
    show.legend = FALSE
  ) + 
  theme_bw()

plt_traj_v2 / plt_box_v2











# binary treatment --------------------------------------------------------
tib_bt_v0 <- read_csv("output/sim7_binTrt_ver0.csv")


plt_traj_bt_v0 <- tib_bt_v0 %>% 
  pivot_longer(
    cols = Avg:M2,
    names_to = "method", 
    values_to = "est"
  ) %>% 
  group_by(method) %>% 
  mutate(
    idx = factor(rnd) %>% as.integer()
  ) %>% 
  mutate(
    cum_avg = cumsum(est) / idx
  ) %>% 
  ggplot() + 
  geom_line(
    mapping = aes(
      x = idx, y = cum_avg, 
      group = method, color = method
    )
  ) + 
  theme_bw()
  

plt_box_bt_v0 <- tib_bt_v0 %>% 
  pivot_longer(
    cols = Avg:M2,
    names_to = "method", 
    values_to = "est"
  ) %>% 
  ggplot() +
  geom_boxplot(
    mapping = aes(x = method, y = est, fill = method), 
    show.legend = FALSE
  ) + 
  geom_hline(yintercept = -2, color = "red", lty = "dashed") +
  theme_bw()

plt_box_bt_v0



plt_traj_bt_v0 / plt_box_bt_v0









