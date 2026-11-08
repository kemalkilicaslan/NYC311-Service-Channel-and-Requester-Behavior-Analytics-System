# =====================================================================
# 04_survival.R
# Time to closure for HEAT/HOT WATER requests, 2024-2025.
#
# Dropping open requests, or counting them as zero, makes closure look faster
# than it is. Kaplan-Meier and Cox treat open requests as censored instead.
#
#   S1: Kaplan-Meier curves by channel + log-rank test
#   S2: Cox proportional hazards: channel + borough + weekend + time band + year
#       (HR > 1 = closes faster)
#   S3: proportional hazards check (cox.zph, Schoenfeld residuals)
#   S4: robustness: stratified Cox for the covariates that break proportionality
# Input: mart.heat_survival (580K requests)
# =====================================================================

source("R/00_setup.R")
suppressPackageStartupMessages(library(survival))

hs <- q("SELECT channel, borough, hr, is_weekend, yr, time_h::float8 AS time_h, event
         FROM mart.heat_survival ORDER BY unique_key") |>
  mutate(time_d  = time_h / 24,
         channel = relevel(factor(channel), ref = "PHONE"),
         borough = relevel(factor(borough), ref = "MANHATTAN"),
         hr_band = factor(ifelse(hr >= 9 & hr <= 16, "Office", "Off hours")),
         yr      = factor(yr))
cat(sprintf("Requests: %s | Closed: %s | Censored: %s\n",
            format(nrow(hs), big.mark = ","), format(sum(hs$event), big.mark = ","),
            format(sum(hs$event == 0), big.mark = ",")))

# ---- S1 --------------------------------------------------------------
km <- survfit(Surv(time_d, event) ~ channel, data = hs)
cat("\n=== S1. Kaplan-Meier median time to closure (days) with 95% CI ===\n")
print(summary(km)$table[, c("records", "events", "median", "0.95LCL", "0.95UCL")])

km_pts <- summary(km, times = c(1, 2, 3, 7, 14, 30))
km_tab <- tibble(channel = sub("channel=", "", km_pts$strata), day = km_pts$time,
                 still_open_pct = round(100 * km_pts$surv, 1))
cat("\n=== S1. Share still open after N days (%) ===\n")
print(pivot_wider(km_tab, names_from = day, names_prefix = "day_", values_from = still_open_pct))
save_tab(km_tab, "Kaplan-Meier-Share-Still-Open")

lr <- survdiff(Surv(time_d, event) ~ channel, data = hs)
cat(sprintf("\nLog-rank: chi2 = %.0f, df = %d, p = %.3g\n",
            lr$chisq, length(lr$n) - 1, pchisq(lr$chisq, length(lr$n) - 1, lower.tail = FALSE)))

km_df <- tibble(time = km$time, surv = km$surv,
                channel = rep(sub("channel=", "", names(km$strata)), km$strata)) |>
  filter(time <= 14)
km_med <- tibble(channel = sub("channel=", "", rownames(summary(km)$table)),
                 median  = summary(km)$table[, "median"])
p6 <- ggplot(km_df, aes(time, surv, colour = channel)) +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = pal$reference) +
  geom_step(linewidth = 0.7) +
  geom_point(data = km_med, aes(x = median, y = 0.5), size = 2.4) +
  geom_text(data = km_med |> arrange(median) |> mutate(y = 0.5 + 0.06 * row_number()),
            aes(x = median, y = y, label = sprintf("%s %.2f d", channel, median)),
            hjust = -0.1, size = 3.2, colour = pal$text_primary) +
  scale_colour_manual(values = channel_colors) +
  scale_y_continuous(labels = label_percent()) +
  scale_x_continuous(breaks = c(0, 1, 2, 3, 5, 7, 10, 14)) +
  labs(title = "Heat and hot water requests close at almost the same pace on every channel",
       subtitle = "Kaplan-Meier share still open, HEAT/HOT WATER 2024-2025 (open requests censored); labels = median days",
       x = "Days since the request", y = "Share still open", colour = NULL)
save_fig(p6, "Kaplan-Meier-Heat-Hot-Water")

# ---- S2 --------------------------------------------------------------
cox <- coxph(Surv(time_d, event) ~ channel + borough + is_weekend + hr_band + yr, data = hs)
hr_tab <- tidy(cox, exponentiate = TRUE, conf.int = TRUE) |>
  transmute(term, hazard_ratio = round(estimate, 3), conf.low = round(conf.low, 3),
            conf.high = round(conf.high, 3), p.value = signif(p.value, 3))
cat("\n=== S2. Cox hazard ratios (HR > 1 = closes faster; ref: PHONE, MANHATTAN) ===\n")
print(hr_tab, n = Inf)
cat(sprintf("Concordance = %.3f\n", cox$concordance["concordance"]))
save_tab(hr_tab, "Cox-Hazard-Ratios")

# ---- S3 --------------------------------------------------------------
# cox.zph on 580K rows is slow; refit on a 50K random subsample for the check.
# The subsample must be a named object: cox.zph re-evaluates the model's `data`
# argument, so an inline slice_sample() call would draw a second, different sample.
set.seed(42)
hs_sub <- slice_sample(hs, n = 50000)
cox_s  <- coxph(Surv(time_d, event) ~ channel + borough + is_weekend + hr_band + yr, data = hs_sub)
cat("\n=== S3. Proportional hazards check (cox.zph, 50K subsample) ===\n")
print(cox.zph(cox_s))

# ---- S4 --------------------------------------------------------------
# Weekend and year break proportionality, so they become strata (each stratum
# gets its own baseline hazard). If the channel HR holds, the main result is robust.
cox_strat <- coxph(Surv(time_d, event) ~ channel + borough + hr_band + strata(is_weekend, yr), data = hs)
cat("\n=== S4. Channel HR: main model vs stratified (weekend x year) ===\n")
print(bind_rows(
  tidy(cox,       exponentiate = TRUE, conf.int = TRUE) |> mutate(model = "S2 main"),
  tidy(cox_strat, exponentiate = TRUE, conf.int = TRUE) |> mutate(model = "S4 stratified")
) |> filter(startsWith(term, "channel")) |>
  transmute(model, term, hazard_ratio = round(estimate, 3), conf.low = round(conf.low, 3),
            conf.high = round(conf.high, 3)))

dbDisconnect(con)
