# =============================================================================
# 00 — PACKAGES
# Project: RPE Prediction in Competitive Runners
# =============================================================================

library(tidyverse)      # data wrangling and ggplot2
library(readr)          # fast CSV reading
library(rpart)          # Decision Tree (CART)
library(rpart.plot)     # Tree visualisation — required by assignment
library(randomForest)   # Random Forest ensemble
library(xgboost)        # Gradient Boosting
library(caret)          # chronological cross-validation (timeslice)
library(Metrics)        # rmse(), mae()
