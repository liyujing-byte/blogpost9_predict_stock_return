library(tidyverse)
library(tidymodels)
library(slider)
library(xgboost)

set.seed(123)
snapshot <- readRDS("data/raw/prices_snapshot.rds")
prices <- snapshot$prices
spy <- snapshot$spy %>% arrange(date)

# Reuse the feature engineering from the classroom source.
original <- parse(file = "code/classroom_original.R")
feature_step <- which(vapply(original, function(expr) {
  is.call(expr) && identical(expr[[1]], quote(`<-`)) &&
    identical(expr[[2]], quote(df))
}, logical(1)))[1]
stopifnot(!is.na(feature_step))
eval(original[[feature_step]])

panel <- df %>%
  group_by(symbol) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(entry = lead(date), exit = lead(date, 20),
         target = lead(adjusted, 20) / lead(adjusted) - 1) %>%
  ungroup() %>%
  filter(is.finite(target))

validation_start <- as.Date("2023-07-01")
test_start <- as.Date("2024-01-01")
training <- filter(panel, exit < test_start)
validation_fit <- filter(training, exit < validation_start)
validation <- filter(training, date >= validation_start)
test <- filter(panel, date >= test_start)
stopifnot(nrow(validation_fit) > 0, nrow(validation) > 0, nrow(test) > 0,
          max(training$exit) < min(test$date),
          max(validation_fit$exit) < min(validation$date))

fit_model <- function(data, depth, min_n) {
  rec <- recipe(target ~ ., data = data) %>%
    update_role(date, symbol, entry, exit, new_role = "id")
  spec <- boost_tree(trees = 300, tree_depth = depth, learn_rate = 0.05,
                     loss_reduction = 0, min_n = min_n) %>%
    set_engine("xgboost", nthread = 2) %>% set_mode("regression")
  workflow() %>% add_recipe(rec) %>% add_model(spec) %>% fit(data = data)
}

candidates <- crossing(depth = 1:3, min_n = c(10L, 30L, 60L)) %>%
  bind_rows(tibble(depth = 5L, min_n = 10L)) %>%
  mutate(validation_rmse = map2_dbl(depth, min_n, function(depth, min_n) {
    model <- fit_model(validation_fit, depth, min_n)
    rmse_vec(validation$target, predict(model, validation)$.pred)
  })) %>% arrange(validation_rmse)
best <- filter(candidates, depth < 5) %>% slice_head(n = 1)

baseline <- fit_model(training, 5L, 10L)
shallower <- fit_model(training, best$depth, best$min_n)
forecasts <- test %>% select(date, symbol, entry, exit, target) %>%
  mutate(`Zero-return benchmark` = 0,
         `Classroom XGBoost` = predict(baseline, test)$.pred,
         `Tuned XGBoost` = predict(shallower, test)$.pred) %>%
  pivot_longer(cols = 6:8, names_to = "model", values_to = "prediction")
stopifnot(all(is.finite(forecasts$prediction)))

metrics <- forecasts %>% group_by(model) %>%
  summarise(RMSE = rmse_vec(target, prediction), MAE = mae_vec(target, prediction),
            observations = n(), .groups = "drop")

# Use every 20th trading session and align both endpoints with SPY.
calendar <- spy %>%
  mutate(entry = lead(date), exit = lead(date, 20),
         spy_return = lead(spy_price, 20) / lead(spy_price) - 1) %>%
  filter(date >= test_start, date <= max(test$date), !is.na(exit) & exit <= max(test$exit))
calendar <- calendar %>% slice(seq.int(1L, n(), by = 20L))
weights <- function(prediction) {
  positive <- pmax(prediction, 0)
  if (sum(positive) == 0) rep(1 / length(positive), length(positive)) else positive / sum(positive)
}
portfolios <- forecasts %>%
  filter(model != "Zero-return benchmark") %>%
  inner_join(select(calendar, date, entry, exit), by = c("date", "entry", "exit")) %>%
  group_by(model, date, exit) %>%
  arrange(desc(prediction), symbol, .by_group = TRUE) %>%
  slice_head(n = 10) %>%
  mutate(weight = weights(prediction)) %>%
  summarise(gross_return = sum(weight * target), holdings = n(),
            total_weight = sum(weight), .groups = "drop")
stopifnot(nrow(portfolios) == 2 * nrow(calendar), all(portfolios$holdings == 10),
          all(abs(portfolios$total_weight - 1) < 1e-10))

# Sensitivity case assumes a full purchase and sale at 10 bps per side.
returns <- portfolios %>%
  mutate(net_return = (1 + gross_return) * (1 - 0.001)^2 - 1) %>%
  select(model, date, exit, gross_return, net_return) %>%
  bind_rows(transmute(calendar, model = "SPY", date, exit,
                      gross_return = spy_return, net_return = spy_return)) %>%
  arrange(model, date) %>% group_by(model) %>%
  mutate(gross_wealth = cumprod(1 + gross_return),
         net_wealth = cumprod(1 + net_return)) %>% ungroup()
investment <- returns %>% group_by(model) %>%
  summarise(periods = n(), gross_return = last(gross_wealth) - 1,
            cost_scenario_return = last(net_wealth) - 1,
            max_drawdown = min(gross_wealth / cummax(c(1, gross_wealth))[-1] - 1),
            .groups = "drop")

plot_data <- returns %>% select(model, date = exit, wealth = gross_wealth) %>%
  bind_rows(tibble(model = unique(returns$model), date = min(calendar$entry), wealth = 1))
chart <- ggplot(plot_data, aes(date, wealth, color = model)) +
  geom_hline(yintercept = 1, color = "grey80") + geom_line(linewidth = 0.9) +
  scale_color_manual(values = c("Classroom XGBoost" = "#35649a",
                               "Tuned XGBoost" = "#007f6c", "SPY" = "#666666")) +
  labs(x = NULL, y = "Growth of $1", color = NULL) +
  theme_minimal(base_size = 12) + theme(legend.position = "bottom")

dir.create("data/results", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/post", recursive = TRUE, showWarnings = FALSE)
write_csv(candidates, "data/results/validation.csv")
write_csv(metrics, "data/results/prediction_metrics.csv")
write_csv(investment, "data/results/investment_metrics.csv")
write_csv(returns, "data/results/backtest_returns.csv")
ggsave("figures/post/cumulative_returns.png", chart, width = 8, height = 4.4, dpi = 160)
print(best)
print(metrics)
print(investment)
cat("Training observations:", nrow(training), "Validation observations:", nrow(validation), "\n")
cat("Test signal dates:", format(range(test$date)), "\n")
cat("Backtest entry and exit:", format(min(calendar$entry)), format(max(calendar$exit)), "\n")

writeLines(capture.output(sessionInfo()), "data/results/session_info.txt")
