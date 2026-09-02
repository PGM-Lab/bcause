# Friedman test (+ Shaffer all-pairs post-hoc) comparing the four samplers on
# RMSE against the exact bounds, run once per analysis group of one dataset.
#
# Stage 2 of the pipeline: friedman_analysis.py prepares
# output/<dataset>/<group>/friedman_input.csv for every group, then invokes
# this script. Each group gets its own console summary, CSV tables, PNG plots
# and LaTeX snippets; a cross-group summary_all_groups.csv is written alongside
# them for the headline comparison.
#
# Usage:  Rscript run_friedman_test.R <dataset>

suppressMessages({
  library(exreport)
  library(xtable)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript run_friedman_test.R <dataset>")
dataset <- args[1]

dataset_dir <- file.path("output", dataset)
if (!dir.exists(dataset_dir)) stop(sprintf("no prepared inputs at %s", dataset_dir))

# "all" first, then the nparents groups in order.
groups <- list.dirs(dataset_dir, full.names = FALSE, recursive = FALSE)
groups <- c(intersect("all", groups), sort(setdiff(groups, "all")))

summary_rows <- list()

for (group in groups) {
  group_dir <- file.path(dataset_dir, group)
  df <- read.csv(file.path(group_dir, "friedman_input.csv"))

  # One row per (method, problem) already -- no expReduce/expInstantiate
  # needed, e$parameters is empty by construction.
  e <- expCreate(df, methods = "method", problems = "problem",
                 name = sprintf("%s / %s (RMSE @ final checkpoint)", dataset, group))
  stopifnot(length(e$parameters) == 0)

  # rankOrder = "min": RMSE is an error metric, lower is better. The warning
  # exreport raises when the omnibus test is not rejected is suppressed here
  # because the outcome is reported explicitly below instead.
  test <- suppressWarnings(
    testMultiplePairwise(e, "rmse", rankOrder = "min", alpha = 0.05))

  tags <- test$friedman$tags
  mean_ranks <- sort(rowMeans(test$friedman$ranks))

  cat(sprintf("\n===== %s / %s =====\n", dataset, group))
  cat(sprintf("problems: %d | chi2 = %.4f (df = %d) | p = %.4e | %s\n",
              nlevels(factor(df$problem)), tags$statistic, tags$distribution,
              tags$pvalue, tags$outcome))
  print(round(mean_ranks, 4))

  # --- Per-group text summary ---
  sink(file.path(group_dir, "friedman_result.txt"))
  summary(test)
  cat("\nMean rank per method (lower = better):\n")
  print(mean_ranks)
  sink()

  # --- CSV tables. tabularTestSummary/tabularTestPairwise return an exTabular
  # object; the actual data.frame lives at $tables[[1]] (one entry per output
  # variable -- there is exactly one here, "rmse"). "rank" is not a valid
  # column for a pairwise test (rank is per-method, pairwise rows are method
  # pairs), so the per-method mean rank is reported separately above. ---
  summary_tab <- tabularTestSummary(test, c("pvalue", "wtl"))$tables[[1]]
  pairwise_tab <- tabularTestPairwise(test, "pvalue")$tables[[1]]
  write.csv(summary_tab, file.path(group_dir, "friedman_summary.csv"), row.names = FALSE)
  write.csv(pairwise_tab, file.path(group_dir, "friedman_pairwise_pvalues.csv"), row.names = FALSE)

  # --- PNG plots ---
  png(file.path(group_dir, "rank_distribution.png"), width = 1000, height = 700)
  print(plotRankDistribution(test))
  invisible(dev.off())

  # --- LaTeX snippets, via xtable (no LaTeX install needed to generate them,
  # only to compile them into a document later). ---
  print(xtable(summary_tab), file = file.path(group_dir, "friedman_summary.tex"),
        include.rownames = FALSE)
  print(xtable(pairwise_tab), file = file.path(group_dir, "friedman_pairwise_pvalues.tex"),
        include.rownames = FALSE)

  summary_rows[[group]] <- data.frame(
    dataset = dataset,
    group = group,
    n_problems = nlevels(factor(df$problem)),
    chi_squared = round(tags$statistic, 4),
    df = tags$distribution,
    p_value = signif(tags$pvalue, 5),
    outcome = tags$outcome,
    best_method = names(mean_ranks)[1],
    Gibbs_Sampling = round(mean_ranks[["Gibbs_Sampling"]], 4),
    Zanella = round(mean_ranks[["Zanella"]], 4),
    Metropolis_Hastings = round(mean_ranks[["Metropolis_Hastings"]], 4),
    Parallel_Tempering = round(mean_ranks[["Parallel_Tempering"]], 4),
    stringsAsFactors = FALSE)
}

overview <- do.call(rbind, summary_rows)
write.csv(overview, file.path(dataset_dir, "summary_all_groups.csv"), row.names = FALSE)
print(xtable(overview), file = file.path(dataset_dir, "summary_all_groups.tex"),
      include.rownames = FALSE)

cat(sprintf("\n----- %s: cross-group overview -----\n", dataset))
print(overview, row.names = FALSE)
cat(sprintf("\nWrote results to %s/\n", dataset_dir))
