setwd("C:/ann/fall 2023/stat 4360/project 5")

# Question 1b

# Install and load required libraries
install.packages("ISLR")
library(ISLR)

#Hitters dataset
data("Hitters")

# Remove the response variable (Salary) and select only the predictor variables
hitters_data <- Hitters[, -which(names(Hitters) == "Salary")]

# Standardize the variables
#convert League, Division, and NewLeague to numeric 
hitters_data$League <- as.numeric(as.factor(hitters_data$League))
hitters_data$Division <- as.numeric(as.factor(hitters_data$Division))
hitters_data$NewLeague <- as.numeric(as.factor(hitters_data$NewLeague))

scaled_hitters <- scale(hitters_data)

# Perform PCA
pca_result <- prcomp(scaled_hitters, scale. = TRUE)

# Summary of PCA
summary(pca_result)

# Scree plot - to determine where the elbow falls
plot(pca_result, type = "l", main = "Scree Plot")


# how many variables to retain, with at least 70% of variance
cumulative_variance <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
num_components <- which(cumulative_variance >= 0.70)[1]
num_components # 3 components


#Question 1c

# Extract the scores for the first 2 PCs
pc_scores <- as.data.frame(pca_result$x[, 1:2])
pc_scores

# Extract the loadings for the first first 2 PCs
pc_loadings <- as.data.frame(pca_result$rotation[, 1:2])
pc_loadings

# Correlations of standardized quantitative variables with the first two PCs
cor_with_pcs <- cor(scaled_hitters, pc_scores)

# Create a table
cor_table <- as.table(cor_with_pcs)
cor_table 

# install.packages("remotes")
# remotes::install_github("vqv/ggbiplot")
# ggbiplot(fit, labels =  rownames(hitters_data))
require(ggbiplot)
ggbiplot(pca_result)

# Question 2a

# Load required libraries
library(MASS)  # For the 'lm.ridge' function
library(boot)  # For the 'cv.glm' function


# get standardized attributes of Hitters and keep Salary
Hitters$League <- as.numeric(as.factor(Hitters$League))
Hitters$Division <- as.numeric(as.factor(Hitters$Division))
Hitters$NewLeague <- as.numeric(as.factor(Hitters$NewLeague))

#omit rows with NA
na.omit(Hitters)

# Fit a linear regression model with log(Salary) as the response
model <- lm(log(Salary) ~ ., data = Hitters)

# Compute LOOCV for the linear regression model
loocv_result <- cv.glm(data = Hitters, glmfit = model)

# Extract the estimated test MSE
test_mse <- loocv_result$delta[1]

# Print the test MSE
cat("Test MSE for Linear Regression Model:", test_mse, "\n")

# Question 2b
# Assuming 'scaled_hitters' contains the standardized predictor variables
# Assuming 'Salary' is the response variable in the Hitters dataset

# Load required libraries
library(pls)   # For the 'pcr' function
library(boot)  # For the 'cv.plsR' function

# Combine response variable and predictors
data_matrix <- cbind(log(Hitters$Salary), scaled_hitters)

# Perform LOOCV for PCR to find the optimal number of components (M)
loocv_pcr <- cv.plsR(data_matrix[, -1], data_matrix[, 1], ncomp = 1:20, method = "simpls")

# Optimal number of components
optimal_components <- loocv_pcr$min

# Fit PCR model with the optimal number of components
pcr_model <- pcr(log(Salary) ~ ., data = data_matrix, scale = TRUE, ncomp = optimal_components)

# Compute LOOCV for the PCR model
loocv_result_pcr <- cv.plsR(data_matrix[, -1], data_matrix[, 1], ncomp = optimal_components, method = "simpls")

# Extract the estimated test MSE
test_mse_pcr <- loocv_result_pcr$MSEP

# Print the optimal number of components and test MSE for PCR
cat("Optimal Number of Components for PCR:", optimal_components, "\n")
cat("Test MSE for PCR Model:", test_mse_pcr, "\n")


