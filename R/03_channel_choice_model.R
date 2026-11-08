# =====================================================================
# 03_channel_choice_model.R
# Which factors explain the choice of a digital channel (ONLINE/MOBILE) over phone?
#
# Logistic regression: P(digital) ~ complaint family + borough + time band
#                      + weekend + year (+ address history in M2)
#
# Grouped binomial GLM: cbind(digital, phone) ~ X gives exactly the same
# coefficients as fitting one row per request with the same covariates, but
# it runs on ~17K cells instead of ~17M rows.
#
# Universe: 2020-01 to 2026-09, known channel, heavy requesters (>1000 requests)
#           excluded, because R/02 T5 showed they distort the borough effect.
# =====================================================================

source("R/00_setup.R")

cells <- q("
  SELECT yr, borough, complaint_family, is_weekend,
         CASE WHEN hr BETWEEN 9 AND 16  THEN 'Office 09-17'
              WHEN hr BETWEEN 17 AND 21 THEN 'Evening 17-22'
              WHEN hr BETWEEN 6 AND 8   THEN 'Morning 06-09'
              ELSE 'Night 22-06' END                         AS hr_band,
         CASE WHEN prior_n = 0              THEN '0 first request'
              WHEN prior_digital_rate = 0   THEN '1 always phone'
              WHEN prior_digital_rate < 0.5 THEN '2 mostly phone'
              WHEN prior_digital_rate < 1   THEN '3 mostly digital'
              ELSE '4 always digital' END                    AS history,
         SUM(is_digital)::float8                             AS digital,
         SUM(1 - is_digital)::float8                         AS phone
  FROM mart.request_history
  WHERE req_total <= 1000 AND borough IS NOT NULL
  GROUP BY 1, 2, 3, 4, 5, 6") |>
  mutate(yr               = factor(yr),
         borough          = relevel(factor(borough), ref = "MANHATTAN"),
         complaint_family = relevel(factor(complaint_family), ref = "Parking & Vehicles"),
         hr_band          = relevel(factor(hr_band), ref = "Office 09-17"),
         history          = factor(history),
         is_weekend       = factor(is_weekend, labels = c("Weekday", "Weekend")))

cat(sprintf("Cells: %s | Requests represented: %s\n",
            format(nrow(cells), big.mark = ","), format(sum(cells$digital + cells$phone), big.mark = ",")))

m0 <- glm(cbind(digital, phone) ~ 1, family = binomial, data = cells)
m1 <- glm(cbind(digital, phone) ~ complaint_family + borough + hr_band + is_weekend + yr,
          family = binomial, data = cells)
m2 <- update(m1, . ~ . + history)

# Request-level log-likelihood (without the grouped binomial constant), used for
# McFadden pseudo-R2, and a weighted AUC computed from the cell-level fit
unit_loglik <- function(m) {
  p <- fitted(m)
  sum(cells$digital * log(p) + cells$phone * log(1 - p))
}
weighted_auc <- function(m) {
  d <- tibble(p = fitted(m), pos = cells$digital, neg = cells$phone) |>
    group_by(p) |> summarise(pos = sum(pos), neg = sum(neg), .groups = "drop") |>
    arrange(p)
  # Mann-Whitney: for every positive, the share of negatives scored below it (ties count 1/2)
  neg_below <- cumsum(d$neg) - d$neg
  sum(d$pos * (neg_below + 0.5 * d$neg)) / (sum(d$pos) * sum(d$neg))
}
ll0 <- unit_loglik(m0)
fit_tab <- tibble(
  model       = c("M1: segment + time", "M2: M1 + address history"),
  n_params    = c(length(coef(m1)), length(coef(m2))),
  mcfadden_r2 = round(1 - c(unit_loglik(m1), unit_loglik(m2)) / ll0, 4),
  auc         = round(c(weighted_auc(m1), weighted_auc(m2)), 4),
  aic         = round(c(AIC(m1), AIC(m2))))
cat("\n=== Model fit ===\n"); print(fit_tab)
save_tab(fit_tab, "Logistic-Regression-Model-Fit")

cat("\n=== Likelihood ratio test: does address history add information? ===\n")
print(anova(m1, m2, test = "LRT"))

or_tab <- tidy(m2, conf.int = TRUE, conf.method = "Wald", exponentiate = TRUE) |>
  filter(term != "(Intercept)") |>
  mutate(group = case_when(
           startsWith(term, "complaint_family") ~ "Complaint family (ref: Parking & Vehicles)",
           startsWith(term, "borough")          ~ "Borough (ref: Manhattan)",
           startsWith(term, "hr_band")          ~ "Time band (ref: Office 09-17)",
           startsWith(term, "is_weekend")       ~ "Weekend (ref: Weekday)",
           startsWith(term, "yr")               ~ "Year (ref: 2020)",
           startsWith(term, "history")          ~ "Address history (ref: first request)"),
         label = sub("^(complaint_family|borough|hr_band|is_weekend|yr|history)", "", term)) |>
  select(group, label, odds_ratio = estimate, conf.low, conf.high, p.value)

cat("\n=== M2 odds ratios (95% CI) ===\n")
print(or_tab |> mutate(across(c(odds_ratio, conf.low, conf.high), ~ round(.x, 3)),
                       p.value = signif(p.value, 2)), n = Inf, width = Inf)
save_tab(or_tab, "Logistic-Regression-Odds-Ratios")

# Points above 1 lean digital (ONLINE blue), below 1 lean phone (PHONE orange);
# a CI that contains 1 is drawn in the neutral reference grey
p5 <- or_tab |>
  filter(!startsWith(group, "Year")) |>
  mutate(label = fct_reorder(label, odds_ratio),
         side  = case_when(conf.low > 1  ~ "Leans digital",
                           conf.high < 1 ~ "Leans phone",
                           TRUE          ~ "No clear effect")) |>
  ggplot(aes(odds_ratio, label, colour = side)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = pal$reference) +
  geom_pointrange(aes(xmin = conf.low, xmax = conf.high), size = 0.35, linewidth = 0.6) +
  scale_colour_manual(values = c(`Leans digital` = channel_colors[["ONLINE"]],
                                 `Leans phone`   = channel_colors[["PHONE"]],
                                 `No clear effect` = pal$reference)) +
  scale_x_log10(breaks = c(0.1, 0.25, 0.5, 1, 2, 4)) +
  facet_grid(group ~ ., scales = "free_y", space = "free_y", labeller = label_wrap_gen(20)) +
  labs(title = "Address history and complaint type drive the choice of a digital channel",
       subtitle = "Logistic regression (M2), odds ratios with 95% CI on a log scale; year effects not shown",
       x = "Odds ratio of digital vs phone", y = NULL, colour = NULL) +
  theme(strip.text.y = element_text(angle = 0, hjust = 0, size = 9, colour = pal$text_secondary))
save_fig(p5, "Logistic-Regression-Odds-Ratios", width = 10, height = 8.5)

dbDisconnect(con)
