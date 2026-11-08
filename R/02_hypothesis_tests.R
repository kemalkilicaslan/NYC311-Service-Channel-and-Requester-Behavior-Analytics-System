# =====================================================================
# 02_hypothesis_tests.R
# Hypothesis tests with effect sizes.
#
# With millions of rows almost every difference is "significant" (p ~ 0), so
# every test reports an effect size next to the p-value (Cramér's V,
# epsilon squared, a difference in percentage points with its CI).
#
#   T1: channel x borough            -> chi-square + Cramér's V
#   T2: channel x complaint family   -> chi-square + Cramér's V (which one explains channel better?)
#   T3: digital share 2020 vs 2025   -> two-proportion test + 95% CI
#   T4: resolution time ~ channel within one complaint type -> Kruskal-Wallis + epsilon squared,
#       pairwise Wilcoxon with Holm correction
#   T5: sensitivity: does the high mobile share of a borough come from heavy requesters?
# =====================================================================

source("R/00_setup.R")
set.seed(42)

cramers_v <- function(tab) {
  chi <- suppressWarnings(chisq.test(tab, correct = FALSE))
  list(chi = chi, v = sqrt(unname(chi$statistic) / (sum(tab) * (min(dim(tab)) - 1))))
}

seg25 <- q("SELECT borough, complaint_family, channel, SUM(n)::float8 AS n
            FROM mart.segment_channel WHERE yr = 2025 GROUP BY 1, 2, 3")

# ---- T1 and T2 -------------------------------------------------------
t1 <- cramers_v(xtabs(n ~ borough + channel, data = seg25))
t2 <- cramers_v(xtabs(n ~ complaint_family + channel, data = seg25))

cat("\n=== T1. Channel x borough (2025) ===\n")
cat(sprintf("chi2 = %.0f, df = %d, p = %.3g, Cramér's V = %.3f\n",
            t1$chi$statistic, t1$chi$parameter, t1$chi$p.value, t1$v))
cat("\n=== T2. Channel x complaint family (2025) ===\n")
cat(sprintf("chi2 = %.0f, df = %d, p = %.3g, Cramér's V = %.3f\n",
            t2$chi$statistic, t2$chi$parameter, t2$chi$p.value, t2$v))
cat("Standardized residuals for PHONE (large |z| = far from independence):\n")
print(sort(round(t2$chi$stdres[, "PHONE"], 0), decreasing = TRUE))
save_tab(tibble(test = c("channel x borough", "channel x complaint_family"),
                chi2 = c(unname(t1$chi$statistic), unname(t2$chi$statistic)),
                df   = c(unname(t1$chi$parameter), unname(t2$chi$parameter)),
                cramers_v = c(t1$v, t2$v)),
         "Chi-Square-and-Cramers-V")

# ---- T3 --------------------------------------------------------------
dig <- q("SELECT yr, SUM(n * is_digital)::float8 AS digital, SUM(n)::float8 AS total
          FROM mart.segment_channel WHERE yr IN (2020, 2025) GROUP BY yr ORDER BY yr")
t3 <- prop.test(x = dig$digital[c(2, 1)], n = dig$total[c(2, 1)])
cat("\n=== T3. Digital share, 2025 vs 2020 ===\n")
cat(sprintf("2020: %.2f%%  2025: %.2f%%  difference: %.2f pp  95%% CI: [%.2f, %.2f] pp, p = %.3g\n",
            100 * dig$digital[1] / dig$total[1], 100 * dig$digital[2] / dig$total[2],
            100 * diff(t3$estimate[2:1]), 100 * t3$conf.int[1], 100 * t3$conf.int[2], t3$p.value))

# ---- T4 --------------------------------------------------------------
kw_by_type <- function(ctype, n_per_channel = 20000) {
  d <- q(sprintf("
      SELECT unique_key::float8 AS unique_key, channel, resolution_hours::float8 AS hours
      FROM core.service_requests
      WHERE complaint_type = '%s' AND is_closed_valid
        AND created_at >= '2025-01-01' AND created_at < '2026-01-01'
        AND channel IN ('ONLINE', 'MOBILE', 'PHONE')", ctype))
  # Equal-size random sample per channel keeps the test balanced and fast. Rows are
  # sorted by unique_key first (in R: ORDER BY in SQL makes PostgreSQL walk the primary
  # key index), so set.seed(42) draws the same sample on every run.
  d  <- d |> arrange(unique_key) |> group_by(channel) |> slice_sample(n = n_per_channel) |> ungroup()
  kw <- kruskal.test(hours ~ channel, data = d)
  pw <- pairwise.wilcox.test(d$hours, d$channel, p.adjust.method = "holm")
  med <- d |> group_by(channel) |> summarise(m = median(hours), .groups = "drop")
  list(type = ctype, kw = kw, eps = unname(kw$statistic) / (nrow(d) - 1), pw = pw,
       med = setNames(med$m, med$channel))
}

cat("\n=== T4. Resolution time ~ channel inside one complaint type (2025, 20K per channel) ===\n")
t4 <- lapply(c("Illegal Parking", "Noise - Residential", "HEAT/HOT WATER"), kw_by_type)
t4_tab <- bind_rows(lapply(t4, function(r) tibble(
  complaint_type = r$type,
  kw_chi2        = round(unname(r$kw$statistic), 1),
  p_value        = signif(r$kw$p.value, 3),
  epsilon_sq     = round(r$eps, 4),
  median_mobile  = round(r$med[["MOBILE"]], 1),
  median_online  = round(r$med[["ONLINE"]], 1),
  median_phone   = round(r$med[["PHONE"]], 1))))
print(t4_tab, width = Inf)
for (r in t4) { cat("\n", r$type, ": Holm-adjusted pairwise Wilcoxon p-values\n"); print(signif(r$pw$p.value, 3)) }
save_tab(t4_tab, "Kruskal-Wallis-Resolution-Time-by-Channel")

# ---- T5 --------------------------------------------------------------
t5 <- q("
  SELECT borough,
         ROUND(100.0 * AVG((channel = 'MOBILE')::int), 1)                                  AS mobile_pct_all,
         ROUND(100.0 * AVG((channel = 'MOBILE')::int) FILTER (WHERE req_total <= 1000), 1) AS mobile_pct_excl_heavy,
         ROUND(100.0 * AVG((req_total > 1000)::int), 1)                                    AS heavy_share_pct
  FROM mart.request_history
  WHERE yr = 2025 AND borough IS NOT NULL
  GROUP BY borough ORDER BY mobile_pct_all DESC")
cat("\n=== T5. Mobile share (2025) with and without heavy requesters (>1000 requests) ===\n")
print(t5)
save_tab(t5, "Heavy-Requester-Sensitivity")

dbDisconnect(con)
