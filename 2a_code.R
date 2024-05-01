setwd("C:/ann/fall 2023/stat 4360/project 5")

# Question 2a

# Install and load required libraries
library(ISLR)

#Hitters dataset
data("Hitters")
Hitters <- na.omit(Hitters)

model <- lm(log(Hitters$Salary) ~ ., data = Hitters)
model.summary()

#no need to split data into test and training sets, since "all data is taken as training data"
predictions <- predict(model, newdata = Hitters)

# Compute the test MSE
test_mse_linear <- mean((log(Hitters$Salary) - predictions)^2)

# Print the results
cat("Test MSE (Linear model):", test_mse_linear, "\n")

#Question 2b

library(pls)

# Separate predictors and response
X <- model.matrix(Salary ~ ., data = Hitters)[, -1]
y <- log(Hitters$Salary)

# Perform Principal Component Regression with LOOCV
pcr_model <- pcr(y ~ X, scale = TRUE, validation = "LOO")
summary(pcr_model)

optimal_components <- 16 #this is when the CV error is the smallest (0.6406)

pcr_pred <- predict(pcr_model, ncomp=16)

#test mse
test_mse_pcr <- mean((pcr_pred-predictions)^2)

# Print the results
cat("Test MSE (PCR):", test_mse_pcr, "\n")

#Question 2c

pls_model = plsr(y ~ X, data = Hitters, scale = TRUE, validation = "LOO")
summary(pls_model)

#M=12

pls_pred <- predict(pls_model, ncomp=12)

#test mse
test_mse_pls <- mean((pls_pred-predictions)^2)

# Print the results
cat("Test MSE (PLS):", test_mse_pls, "\n")

# Question 2d

library(glmnet)

# ridge regression model with LOOCV
ridge_model <- cv.glmnet(x = X, y = y, alpha = 0)  # alpha = 0 for ridge regression

# Optimal lambda (penalty parameter) from LOOCV
optimal_lambda <- ridge_model$lambda.min

# Fit Ridge Regression model with the optimal penalty parameter
final_ridge_model <- glmnet(x = X, y = y, alpha = 0, lambda = optimal_lambda)

# Make predictions on the test data
X_test <- model.matrix(Salary ~ ., data = Hitters)[, -1]
y_test <- log(Hitters$Salary)
predictions_ridge <- predict(final_ridge_model, s = optimal_lambda, newx = X_test)

# Compute the test MSE
test_mse_ridge <- mean((y_test - predictions_ridge)^2)
cat("Test MSE (Ridge Regression):", test_mse_ridge, "\n")

# Question 3a

# Separate predictors and response
X <- model.matrix(Salary ~ ., data = Hitters)[, -1]
y <- log(Hitters$Salary)

# Fit a linear model
model <- lm(y ~ X)
model

# Extract coefficients and their names
coefficients <- coef(model)
predictor_names <- names(coefficients)

# Identify the most important predictor (variable with the largest absolute coefficient)
most_important_predictor <- predictor_names[which.max(abs(coefficients[-1]))]

# Print the results
cat("Most important predictor:", most_important_predictor, "\n")

# Question 3b
# Load necessary packages
install.packages("splines")
library(splines)

# Fit natural cubic spline regression model with LOOCV
spline_model <- lm(y ~ ns(Hitters$CWalks, df = 7), data = Hitters)
spline_model

# Make predictions on the test data
predictions_spline <- predict(spline_model, newdata = Hitters)

# Compute the test MSE
test_mse_spline <- mean((y - predictions_spline)^2)

cat("Estimated test MSE (spline):", test_mse_spline, "\n")

# Question 3c

install.packages("gam")
library(gam)

# Fit a Generalized Additive Model (GAM)
gam_model <- gam(y ~ s(AtBat, 3) + s(Hits,4) + s(HmRun, 5) 
                 + s(Runs, 5) + s(RBI, 5) + s(Walks,4) 
                 + s(Years, 4) + s(CAtBat, 4) + s(CHits, 4)
                 + s(CHmRun,4) + s(CRuns, 5) + s(CRBI, 5)
                 + s(CWalks, 3) + League + Division + s(PutOuts, 5)
                 + s(Assists, 5) + s(Errors, 5) + s(Salary, 5)
                 + NewLeague, data = Hitters)

# Summarize the model results
summary(gam_model)

#find the most important predictor
name <- names(coef(gam_model))
most_important_gam_predictor <- name[which.max(abs(coefficients[-1]))]
most_important_gam_predictor
