# =====================================================================
# 01_channel_mix.R
# Figures for the channel mix results of sql/03_channel_mix.sql
#   Monthly-Channel-Share: monthly channel share, 2020-01 to 2026-09
#   Phone-Share-by-Complaint-Family: phone share by complaint family, 2020 vs 2025
#   Hourly-Profile-by-Channel: hour-of-day profile per channel
#   Borough-Channel-Mix: channel mix by borough, 2025
# Input: mart.monthly_channel, mart.segment_channel
# =====================================================================

source("R/00_setup.R")

# Last point of each line, with label positions pushed apart so they never overlap
end_labels <- function(df, x, y, min_gap) {
  df |>
    group_by(channel) |> slice_max({{ x }}, n = 1, with_ties = FALSE) |> ungroup() |>
    arrange({{ y }}) |>
    mutate(label_y = {{ y }},
           label_y = purrr::accumulate(label_y, ~ max(.y, .x + min_gap)))
}

# ---- figure 1 -----------------------------------------------------------
monthly <- q("SELECT month_start, channel, share_pct::float8 AS share_pct FROM mart.monthly_channel")

p1 <- ggplot(monthly, aes(month_start, share_pct, colour = channel)) +
  annotate("rect", xmin = as.Date("2022-07-01"), xmax = as.Date("2023-12-31"),
           ymin = -Inf, ymax = Inf, fill = pal$grid, alpha = 0.6) +
  annotate("text", x = as.Date("2023-04-01"), y = 63, label = "Phone rebound\n2022-H2 to 2023",
           size = 3.2, colour = pal$text_secondary) +
  geom_line(linewidth = 0.7) +
  geom_point(data = end_labels(monthly, month_start, share_pct, 3), size = 2.2) +
  geom_text(data = end_labels(monthly, month_start, share_pct, 3),
            aes(y = label_y, label = sprintf("%s %.0f%%", channel, share_pct)),
            hjust = -0.12, size = 3.3, colour = pal$text_primary) +
  scale_colour_manual(values = channel_colors) +
  scale_x_date(expand = expansion(mult = c(0.01, 0.13))) +
  scale_y_continuous(labels = label_percent(scale = 1)) +
  labs(title = "Online overtook phone as the main 311 channel",
       subtitle = "Monthly share of requests with a known channel (UNKNOWN/OTHER excluded)",
       x = NULL, y = "Share of requests", colour = NULL)
save_fig(p1, "Monthly-Channel-Share")

# ---- figure 2 -----------------------------------------------------------
seg <- q("SELECT yr, borough, complaint_family, hr, is_weekend, channel, n::float8 AS n
          FROM mart.segment_channel")

fam <- seg |>
  filter(yr %in% c(2020, 2025)) |>
  group_by(complaint_family, yr) |>
  summarise(phone_pct = 100 * sum(n[channel == "PHONE"]) / sum(n), .groups = "drop") |>
  pivot_wider(names_from = yr, names_prefix = "phone_", values_from = phone_pct) |>
  mutate(change_pp = phone_2025 - phone_2020,
         complaint_family = fct_reorder(complaint_family, phone_2025))
print(arrange(fam, desc(phone_2025)) |> mutate(across(where(is.numeric), ~ round(.x, 1))))

p2 <- ggplot(fam, aes(y = complaint_family)) +
  geom_segment(aes(x = phone_2020, xend = phone_2025, yend = complaint_family),
               colour = pal$reference, linewidth = 0.6) +
  geom_point(aes(x = phone_2020), colour = pal$reference, size = 2.6) +
  geom_point(aes(x = phone_2025), colour = channel_colors[["PHONE"]], size = 3) +
  geom_text(aes(x = phone_2025, label = sprintf("%.0f%%", phone_2025)),
            nudge_y = 0.35, size = 3.1, colour = pal$text_primary) +
  scale_x_continuous(labels = label_percent(scale = 1), limits = c(0, 100)) +
  labs(title = "Housing complaints still depend on the phone; sanitation moved online",
       subtitle = "Share of requests made by phone: grey = 2020, orange = 2025",
       x = "Phone share", y = NULL)
save_fig(p2, "Phone-Share-by-Complaint-Family", height = 6)

# ---- figure 3 -----------------------------------------------------------
hourly <- seg |>
  group_by(channel, hr) |>
  summarise(n = sum(n), .groups = "drop_last") |>
  mutate(pct = 100 * n / sum(n)) |>
  ungroup()

p3 <- ggplot(hourly, aes(hr, pct, colour = channel)) +
  annotate("rect", xmin = 9, xmax = 17, ymin = -Inf, ymax = Inf, fill = pal$grid, alpha = 0.6) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.6) +
  geom_text(data = end_labels(hourly, hr, pct, 0.3), aes(y = label_y, label = channel),
            hjust = -0.25, size = 3.3, colour = pal$text_primary) +
  scale_colour_manual(values = channel_colors) +
  scale_x_continuous(breaks = seq(0, 23, 3), expand = expansion(mult = c(0.02, 0.1))) +
  scale_y_continuous(labels = label_percent(scale = 1)) +
  labs(title = "Phone peaks in office hours, mobile peaks late in the evening",
       subtitle = "Distribution of each channel's volume by hour, 2020-01 to 2026-09 (band = 09-17)",
       x = "Hour of day", y = "Share of the channel's volume", colour = NULL)
save_fig(p3, "Hourly-Profile-by-Channel")

# ---- figure 4 -----------------------------------------------------------
boro <- seg |>
  filter(yr == 2025) |>
  group_by(borough, channel) |>
  summarise(n = sum(n), .groups = "drop_last") |>
  mutate(pct = 100 * n / sum(n)) |>
  ungroup() |>
  mutate(channel = factor(channel, levels = c("MOBILE", "ONLINE", "PHONE")),
         borough = fct_reorder(borough, ifelse(channel == "MOBILE", pct, 0), .fun = max))

# reverse = TRUE stacks the first level on the left, so the bars read in legend order
p4 <- ggplot(boro, aes(pct, borough, fill = channel)) +
  geom_col(width = 0.7, colour = pal$surface, linewidth = 0.6, position = position_stack(reverse = TRUE)) +
  geom_text(aes(label = sprintf("%.0f%%", pct)), position = position_stack(vjust = 0.5, reverse = TRUE),
            size = 3.3, colour = pal$text_primary) +
  scale_fill_manual(values = channel_colors) +
  scale_x_continuous(labels = label_percent(scale = 1)) +
  labs(title = "Staten Island leans phone; the Bronx mobile share is inflated by heavy requesters",
       subtitle = paste0("Channel mix by borough, 2025\nWithout addresses filing >1000 requests, ",
                         "the Bronx mobile share falls from 33% to 21% (R/02 T5)"),
       x = NULL, y = NULL, fill = NULL)
save_fig(p4, "Borough-Channel-Mix", height = 4.5)

dbDisconnect(con)
