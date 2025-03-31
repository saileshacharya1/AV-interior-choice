### Load packages and import prepared data #####################################
################################################################################

# load packages
library(tidyverse)
library(ggplot2)

# import prepared data
df <- read.csv("../data/prepared_data.csv")

# keep first observation for each individual, for plots 
df <- df %>%
  group_by(id) %>%
  slice(1) %>%
  ungroup()

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#


### Plot of indicators of latent variables #####################################
################################################################################

# create a df consisting required columns
df_p1 <- df[, c(
  "av_benefit_1", "av_benefit_2", "av_benefit_3",
  "av_benefit_4", "av_benefit_5", "av_benefit_6",
  "av_concern_1", "av_concern_4", "av_concern_5",
  "av_concern_2", "av_concern_3", "av_concern_6",
  "av_concern_7", "tech_savvy_1", "tech_savvy_3",
  "enjoy_driving_1", "enjoy_driving_3", "enjoy_driving_4",
  "polychronicity_1", "polychronicity_2", "polychronicity_3",
  "envt_concern_1", "envt_concern_2", "envt_concern_3"
)]

# convert the df to long format
df_p1 <- df_p1 %>%
  pivot_longer(
    cols = c(
      "av_benefit_1", "av_benefit_2", "av_benefit_3",
      "av_benefit_4", "av_benefit_5", "av_benefit_6",
      "av_concern_1", "av_concern_4", "av_concern_5",
      "av_concern_2", "av_concern_3", "av_concern_6",
      "av_concern_7", "tech_savvy_1", "tech_savvy_3",
      "enjoy_driving_1", "enjoy_driving_3", "enjoy_driving_4",
      "polychronicity_1", "polychronicity_2", "polychronicity_3",
      "envt_concern_1", "envt_concern_2", "envt_concern_3"
    ),
    names_to = c("item"),
    values_to = "outcome"
  )

# summarize the items outcomes
df_p1 <- df_p1 %>%
  group_by(item, outcome) %>%
  summarize(
    count = n(),
    percent = count / nrow(df) * 100
  )

# add name of latent variable
df_p1$variable <- "AV usefulness"
df_p1$variable <- ifelse(str_detect(df_p1$item, c("concern_2")),
  "AV concern", df_p1$variable
)
df_p1$variable <- ifelse(str_detect(df_p1$item, c("concern_3")),
  "AV concern", df_p1$variable
)
df_p1$variable <- ifelse(str_detect(df_p1$item, c("concern_6")),
  "AV concern", df_p1$variable
)
df_p1$variable <- ifelse(str_detect(df_p1$item, c("concern_7")),
  "AV concern", df_p1$variable
)
df_p1$variable <- ifelse(str_detect(df_p1$item, "tech"),
  "Technology \n savviness", df_p1$variable
)
df_p1$variable <- ifelse(str_detect(df_p1$item, "enjoy"),
  "Driving \n enjoyment", df_p1$variable
)
df_p1$variable <- ifelse(str_detect(df_p1$item, "polychronicity"),
  "Polychronicity", df_p1$variable
)
df_p1$variable <- ifelse(str_detect(df_p1$item, "envt"),
  "Environmental \n awareness", df_p1$variable
)

# factor levels of variables
df_p1$variable <- factor(df_p1$variable, levels = c(
  "AV usefulness",
  "AV concern",
  "Polychronicity",
  "Driving \n enjoyment",
  "Technology \n savviness",
  "Environmental \n awareness"
))

# factor levels of outcomes
df_p1$outcome <- factor(df_p1$outcome,
  levels = c(5:1),
  labels = c(
    "Strongly agree (5)", "Somewhat agree (4)",
    "Neutral (3)",
    "Somewhat disagree (2)", "Strongly disagree (1)"
  )
)

# rename the items
items <-
  c(
    av_benefit_1 = "AVs will drive me safely to wherever I want. (AU-1)",
    av_benefit_2 = "Using an AV will improve my (and others') driving
    efficiency. (AU-2)",
    av_benefit_3 = "I could multitask while traveling in an AV
    (e.g., work, sleep, surf the internet). (AU-3)",
    av_benefit_4 = " Using an AV will reduce my driving burden/stress. (AU-4)",
    av_benefit_5 = "AVs will improve the mobility of overall
    transportation. (AU-5)",
    av_benefit_6 = "AVs will offer economic and social benefits in
    overall. (AU-6)",
    av_concern_1 = "I would feel comfortable having an AV pickup/drop off
    children without adult supervision. (AU-7)",
    av_concern_4 = "AVs would make me feel safer on the streets as a
    pedestrian or as a bicyclist. (AU-8)",
    av_concern_5 = "AVs would perform well even in poor weather or other
    unexpected conditions. (AU-9)",
    av_concern_2 = "I am concerned about the potential failure of AV sensors,
    equipment, technology, and system safety. (AC-1)",
    av_concern_3 = "I am concerned about the legal liability for drivers
    or owners of AVs in accidents/crashes. (AC-2)",
    av_concern_6 = "I am concerned about the data privacy and security
    breaches/hacking in AVs. (AC-3)",
    av_concern_7 = "I am worried about the higher purchase, maintenance,
    and insurance costs associated with AVs. (AC-4)",
    tech_savvy_1 = "I like to be among the first to have the latest
    technology. (TS-1)",
    tech_savvy_3 = "Having internet connectivity everywhere I go is
    important to me. (TS-2)",
    enjoy_driving_1 = "I enjoy driving myself. (DE-1)",
    enjoy_driving_3 = "I prefer not to have the responsibility of
    driving. (DE-2)",
    enjoy_driving_4 = "I feel stressed or nervous when driving. (DE-3)",
    polychronicity_1 = "I like to be engaged in two or more activities
    simultaneously. (PC-1)",
    polychronicity_2 = "I believe people should aim at performing multiple
    tasks simultaneously. (PC-2)",
    polychronicity_3 = "It makes me feel good to be involved in multiple
    activities simultaneously. (PC-3)",
    envt_concern_1 = "I am concerned about current environmental pollution
    and its impact on health. (EA-1)",
    envt_concern_2 = "I don't change my behavior based solely on concern
    for the environment. (EA-2)",
    envt_concern_3 = "I rarely worry about the effects of pollution on
    myself and my family. (EA-3)"
  )
df_p1$item <- as.character(items[df_p1$item])
df_p1$item <- factor(df_p1$item, levels = rev(as.character(items)))
rm(items)

# plot of items of latent variables

p1 <- ggplot(
  data = df_p1,
  aes(x = item, y = percent, fill = outcome)
) +
  geom_bar(
    stat = "identity",
    position = "fill",
    width = 0.7,
    color = "black",
    size = 0.1
  ) +
  scale_fill_brewer(palette = "RdBu", direction = -1) +
  geom_text(aes(label = ifelse(percent > 5, count, "")),
    position = position_fill(vjust = 0.5),
    colour = "black",
    size = 3
  ) +
  scale_y_continuous(labels = scales::percent, expand = c(0.01, 0.01)) +
  coord_flip() +
  theme(
    text = element_text(family = "Times New Roman", color = "black", size = 10),
    panel.spacing.x = unit(0, "lines"),
    panel.border = element_rect(colour = "black", fill = NA, size = 0.5),
    panel.grid.major.x = element_line(color = "grey80", size = 0.25),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    axis.text = element_text(color = "black"),
    axis.ticks.x = element_line(size = 0.25),
    axis.ticks.y = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.justification = c(1, 0),
    legend.background = element_blank(),
    legend.key.size = unit(0.08, "in"),
    strip.background = element_rect(color = "black", fill = "white")
  ) +
  guides(fill = guide_legend(reverse = TRUE)) +
  facet_grid(variable ~ ., scales = "free_y", space = "free_y")

# print the plot
p1

# save the plot
ggsave(
  filename = "../outputs/plots/p1_latent_variables.jpeg",
  plot = p1, width = 6.5, height = 8.5, unit = "in", dpi = 1000
)
rm(p1)
rm(df_p1)
dev.off()
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#


### Plot of travel-based activities ############################################
################################################################################


df_p2 <- df %>%
  dplyr::select(contains(c("tba_hv", "tba_av")))


# convert the df to long format
df_p2 <- df_p2 %>%
  pivot_longer(
    cols = colnames(dplyr::select(df, contains(c("tba_hv", "tba_av")))),
    names_to = c("activity"),
    values_to = "outcome"
  )

# summarize the activities outcomes
df_p2 <- df_p2 %>%
  group_by(activity, outcome) %>%
  summarize(
    count = n(),
    percent = count / nrow(df) * 100
  )

# add variable describing HV or AV scenario
df_p2$variable <- ifelse(str_detect(df_p2$activity, "hv"),
                         "HV-TBA",
                         "AV-TBA"
)

# remove observations with "NO" outcomes
df_p2 <- df_p2[df_p2$outcome == 1, ]
df_p2$outcome <- NULL

# remove "hv" and "av" from activities
df_p2$activity <- gsub("hv_|av_", "", df_p2$activity)





# create a df consisting required columns
df_p2 <- df %>%
  dplyr::select(contains(c("tba_g")))%>%
  dplyr::select(contains(c("_cat")))

# convert the df to long format
df_p2 <- df_p2 %>%
  pivot_longer(
    cols = colnames(dplyr::select(df_p2, contains(c("hv", "av")))),
    names_to = c("activity"),
    values_to = "outcome"
  )

# summarize the activities outcomes
df_p2 <- df_p2 %>%
  group_by(activity, outcome) %>%
  summarize(
    count = n(),
    percent = count / nrow(df) * 100
  )

# add variable describing HV or AV scenario
df_p2$variable <- ifelse(str_detect(df_p2$activity, "hv"),
  "HV-TBA",
  "AV-TBA"
)

# remove observations with "NO" outcomes
df_p2 <- df_p2[df_p2$outcome == 1, ]
df_p2$outcome <- NULL

# remove "hv" and "av" from activities
df_p2$activity <- gsub("_hv_cat|_av_cat", "", df_p2$activity)

# rename the activities
activities <-
  c(
    tba_g1 = "Use social media",
    tba_g2 = "Work/study/read",
    tba_g3 = "Interact",
    tba_g4 = "Entertain",
    tba_g5 = "Eat/care",
    tba_g6 = "Relax",
    tba_g7 = "Watch road"
  )
df_p2$activity <- as.factor(as.character(activities[df_p2$activity]))
rm(activities)

# reorder the levels of activities
df_p2$activity <- fct_reorder2(df_p2$activity, df_p2$variable, df_p2$count)

# some column manipulations
df_p2$percent <- ifelse(df_p2$variable == "AV-TBA",
  -df_p2$percent, df_p2$percent
)
df_p2$x_max <- ifelse(df_p2$variable == "AV-TBA", 0, 100)
df_p2$x_min <- ifelse(df_p2$variable == "AV-TBA", -100, 0)


# plot of travel-based activities
p2 <- df_p2 %>% ggplot(aes(x = percent, y = activity, fill = variable)) +
  geom_bar(stat = "identity", width = 0.9, color = "black", size = 0.1) +
  facet_wrap(~variable, scales = "free_x") +
  geom_blank(aes(x = x_min)) +
  geom_blank(aes(x = x_max)) +
  geom_text(aes(label = count, 
                hjust = ifelse(df_p2$variable == "AV-TBA", 1.5, -0.5), ),
    position = position_dodge(width = 0.6),
    colour = "black",
    size = 3
  ) +
  scale_fill_manual(values = c("#f4a582", "#92c5de")) +
  scale_x_continuous(labels = function(x) paste0(abs(x), "%")) +
  theme(
    text = element_text(family = "Times New Roman", color = "black", size = 10),
    panel.spacing.x = unit(0, "lines"),
    panel.border = element_rect(colour = "black", fill = NA, size = 0.5),
    panel.grid.major.x = element_line(color = "grey80", size = 0.25),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    axis.text = element_text(color = "black"),
    axis.ticks.x = element_line(size = 0.25),
    axis.ticks.y = element_line(size = 0.25),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    strip.background = element_rect(color = "black", fill = "white"),
    legend.position = "none"
  )


# print the plot
p2

# save the plot
ggsave(
  filename = "../outputs/plots/p2_tba.jpeg",
  plot = p2, width = 6.5, height = 3.5, unit = "in", dpi = 1000
)
rm(p2, df_p2)
dev.off()
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#


### Plot of travel time usefulness #############################################
################################################################################

# create a df consisting required columns
df_p3 <- df[, c("tu_hv", "tu_av")]

# convert the df to long format
df_p3 <- df_p3 %>%
  pivot_longer(
    cols = c("tu_hv", "tu_av"),
    names_to = c("tu"),
    values_to = "level"
  )

# summarize travel usefulness levels
df_p3 <- df_p3 %>%
  group_by(tu, level) %>%
  summarize(
    count = n(),
    percent = count / nrow(df) * 100
  )

# factor levels of travel usefulness
df_p3$level <- factor(df_p3$level,
  levels = c(5:1),
  labels = c(
    "Mostly useful (5)", "Somewhat useful (4)",
    "Neutral (3)",
    "Somewhat wasted (2)", "Mostly wasted (1)"
  )
)

# make travel usefulness column as factor
df_p3$tu <- factor(df_p3$tu,
  levels = c("tu_hv", "tu_av"),
  labels = c("HV-TTU", "AV-TTU")
)

# plot of travel usefulness
p3 <- ggplot(
  data = df_p3,
  aes(x = tu, y = percent, fill = level)
) +
  geom_bar(
    stat = "identity",
    position = "fill",
    width = 0.7,
    color = "black",
    size = 0.1
  ) +
  scale_fill_brewer(palette = "RdBu", direction = -1) +
  geom_text(aes(label = ifelse(percent > 5, count, "")),
    position = position_fill(vjust = 0.5),
    colour = "black",
    size = 3
  ) +
  scale_y_continuous(labels = scales::percent, expand = c(0.01, 0.01)) +
  coord_flip() +
  theme(
    text = element_text(family = "Times New Roman", color = "black", size = 10),
    panel.border = element_rect(colour = "black", fill = NA, size = 0.5),
    panel.grid.major.x = element_line(color = "grey80", size = 0.25),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    axis.text = element_text(color = "black"),
    axis.ticks.x = element_line(size = 0.25),
    axis.ticks.y = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.key.size = unit(0.08, "in"),
    legend.background = element_blank()
  ) +
  guides(fill = guide_legend(reverse = TRUE))

# print the plot
p3

# save the plot
ggsave(
  filename = "../outputs/plots/p3_ttu.jpeg",
  plot = p3, width = 6.5, height = 1.8, unit = "in", dpi = 1000
)
rm(p3)
rm(df_p3)
dev.off()

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#


### Summary statistics #########################################################
################################################################################

# av_usefulness
summary(df$av_usefulness_std)
summary(df$av_usefulness)


# envt_concern
summary(df$envt_concern_std)

# polychronicity
summary(df$polychronicity_std)

# av familiarity
summary(df$av_fam)
sd(df$av_fam)

# ttu_hv
summary(df$tu_hv)
sd(df$tu_hv)

# ttu_av
summary(df$tu_av)
sd(df$tu_av)

# ttu_diff
summary(df$ttu_diff)
sd(df$ttu_diff)

# tba_diff
summary(df$tba_tot_diff)
sd(df$tba_tot_diff)

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

