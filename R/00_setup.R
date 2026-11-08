# =====================================================================
# 00_setup.R
# Shared setup: packages, PostgreSQL connection, palette, ggplot theme, save helpers.
# Every other R script sources this file.
#
# Open NYC311-Service-Channel-and-Requester-Behavior-Analytics-System.Rproj in RStudio so the working directory is
# the project root. The password is read from ~/.pgpass by libpq.
# =====================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(RPostgres)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
  library(forcats)
  library(broom)
})

con <- dbConnect(
  RPostgres::Postgres(),
  host   = Sys.getenv("PGHOST", "localhost"),
  port   = 5432,
  user   = Sys.getenv("PGUSER", "postgres"),
  dbname = Sys.getenv("PGDATABASE", "nyc311")
)

# Figures go to the project root with the project name as prefix, like the README
# images of the other repositories (same prefix as python/common.py)
fig_prefix <- "NYC311-Service-Channel-and-Requester-Behavior-Analytics-"
tab_dir <- "outputs/tables"
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

# One palette file for R and Python, so a channel has the same color everywhere
pal <- jsonlite::fromJSON("config/palette.json")
channel_colors <- unlist(pal$channel)

theme_set(
  theme_minimal(base_size = 12) +
    theme(plot.background    = element_rect(fill = pal$surface, colour = NA),
          panel.background   = element_rect(fill = pal$surface, colour = NA),
          panel.grid.major   = element_line(colour = pal$grid, linewidth = 0.3),
          panel.grid.minor   = element_blank(),
          text               = element_text(colour = pal$text_primary),
          axis.text          = element_text(colour = pal$text_secondary),
          plot.title         = element_text(face = "bold"),
          plot.subtitle      = element_text(colour = pal$text_secondary),
          plot.title.position = "plot",
          legend.position    = "top",
          legend.justification = "left")
)

save_fig <- function(plot, name, width = 9, height = 5.5) {
  path <- paste0(fig_prefix, name, ".png")
  ggsave(path, plot, width = width, height = height, dpi = 150, bg = pal$surface)
  message("Saved: ", path)
}

save_tab <- function(df, name) {
  path <- file.path(tab_dir, paste0(name, ".csv"))
  readr::write_csv(df, path)
  message("Saved: ", path)
}

q <- function(sql) as_tibble(dbGetQuery(con, sql))
