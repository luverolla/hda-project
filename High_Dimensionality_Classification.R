# >>> installation ---
#install.packages("glmnet") # general linear models
#install.packages("caret") # train/test split
#install.packages("magrittr") # for the pipe operator
#install.packages("dplyr") # for data manipulation
#install.packages("ggplot2") # for data visualization
#install.packages("gridExtra") # for grid layout

library(glmnet) # general linear models
library(caret) # train/test split
library(magrittr) # for the pipe operator
library(dplyr) # for data manipulation
library(ggplot2) # for data visualization
library(gridExtra) # for grid layout

options(max.print=100000000)

# Load the data
data <- read.csv('data/leukemia.csv')

# Separate features and response variable
X <- as.matrix(data[, -1])
Y <- as.factor(data$Y)

# drop columns with missing values or zero values from data
data = data[, colSums(is.na(data)) == 0]
data = data[, colSums(data == 0) == 0]

# Set seed for reproducibility
set.seed(123)

# Create train/test split
trainIndex <- createDataPartition(Y, p = .8, 
                                  list = FALSE, 
                                  times = 1)

X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
Y_train <- Y[trainIndex]
Y_test <- Y[-trainIndex]

#Function for create histograms 
create_histogram <- function(variable) {
  if (!(is.factor(data[[variable]]) || is.character(data[[variable]]))) {  # Exclude discrete variables
    ggplot(data, aes(x = !!sym(variable), fill = Y)) +
      geom_histogram(aes(y = after_stat(density)), color = 'black', alpha = 0.5, bins = 30) +
      geom_density(color = 'black') +
      labs(x = variable, y = 'Density', fill = 'Leukemia', title = paste('Distribution of', variable)) +
      theme_bw() +
      theme(plot.title = element_text(hjust = 0.5),
            legend.position = 'top',
            legend.justification = 'right')
  } else {
    return(NULL)  # Return NULL for discrete variables
  }
}

# Function to plot histograms
plot_histograms <- function(data) {
  histograms <- lapply(names(data), create_histogram)
  histograms <- histograms[!sapply(histograms, is.null)]
  dev.new()
  grid.arrange(grobs = histograms, ncol = 6)
}

# >>> Plain linear regression ---

linear_model <- glmnet(X_train, Y_train, family = "binomial", alpha=0, lambda=0);

lin_predictions <- predict(linear_model, X_test, type = "response")
lin_predicted_classes <- ifelse(lin_predictions > 0.5, 1, 0)
lin_predicted_classes <- as.factor(lin_predicted_classes)

# Confusion matrix
lin_conf_matrix <- confusionMatrix(lin_predicted_classes, Y_test)
print(lin_conf_matrix)
dev.new()
plot(lin_conf_matrix$table, col = c("red", "#0b872c"), main = "Confusion Matrix", xlab = "Predicted classes", ylab = "True classes")

# ROC Curve
library(pROC)
lin_modelROC <- roc(as.numeric(Y_test),as.numeric(lin_predicted_classes))
dev.new()
plot(lin_modelROC, legacy.axes = TRUE)




# >>> Lasso Logistic regressions ---- 

lasso_cv_fit <- cv.glmnet(X_train, Y_train, family = "binomial", alpha = 1)
lasso_best_lambda <- lasso_cv_fit$lambda.min

# Train the final model using the best lambda
lasso_model <- glmnet(X_train, Y_train, family = "binomial", alpha = 1, lambda = lasso_best_lambda)
lasso_beta = lasso_model$beta

# Predict on the test data
lasso_predictions <- predict(lasso_model, X_test, type = "response")
lasso_predicted_classes <- ifelse(lasso_predictions > 0.5, 1, 0)
lasso_predicted_classes <- as.factor(lasso_predicted_classes)

data_lasso = data[, which(lasso_beta != 0)]
X_lasso = X_train[, which(lasso_beta != 0)]
plot_histograms(data_lasso)

# scatterplot between feature 1 and 2
dev.new()
plot(X_lasso[,1], X_lasso[,2], col=c("red","#0b872c")[Y_train], pch=16, xlab = "Feature 1", ylab = "Feature 2")

# scatterplot between feature 2 and 3
dev.new()
plot(X_lasso[,2], X_lasso[,3], col=c("red","#0b872c")[Y_train], pch=16, xlab = "Feature 2", ylab = "Feature 3")

# scatterplot between feature 1 and 10
dev.new()
plot(X_lasso[,1], X_lasso[,10], col=c("red","#0b872c")[Y_train], pch=16, xlab = "Feature 1", ylab = "Feature 10")

# pair plots of first 10 features
dev.new()
pairs(X_lasso[,1:5], col=c("red","#0b872c")[Y_train])

# Confusion matrix
conf_matrix <- confusionMatrix(lasso_predicted_classes, Y_test)
print(conf_matrix)
dev.new()
plot(conf_matrix$table, col = c("red", "#0b872c"), main = "Confusion Matrix", xlab = "Predicted classes", ylab = "True classes")

# ROC Curve
library(pROC)
modelROC <- roc(as.numeric(Y_test),as.numeric(lasso_predicted_classes))
dev.new()
plot(modelROC, legacy.axes = TRUE)


# >>> Ridge Logistic regressions ---- 

ridge_cv_fit <- cv.glmnet(X_train, Y_train, family = "binomial", alpha = 0)
ridge_best_lambda <- ridge_cv_fit$lambda.min

# Train the final model using the best lambda
ridge_model <- glmnet(X_train, Y_train, family = "binomial", alpha = 0, lambda = ridge_best_lambda)
ridge_beta = ridge_model$beta

# Predict on the test data
ridge_predictions <- predict(ridge_model, X_test, type = "response")
ridge_predicted_classes <- ifelse(ridge_predictions > 0.5, 1, 0)
ridge_predicted_classes <- as.factor(ridge_predicted_classes)

# Confusion matrix
ridge_conf_matrix <- confusionMatrix(ridge_predicted_classes, Y_test)
print(ridge_conf_matrix)
dev.new()
plot(ridge_conf_matrix$table, col = c("red", "#0b872c"), main = "Confusion Matrix", xlab = "Predicted classes", ylab = "True classes")

# ROC Curve
library(pROC)
ridge_modelROC <- roc(as.numeric(Y_test),as.numeric(ridge_predicted_classes))
dev.new()
plot(ridge_modelROC, legacy.axes = TRUE, xlim=c(1,0), ylim=c(0,1))


# >>> PCA ----

library(FactoMineR) # multivariate analysis
library(factoextra) # complement of the above

# Perform PCA
pca <- prcomp(X, center = TRUE, scale. = TRUE)

cev = cumsum(pca$sdev^2 / sum(pca$sdev^2))
dev.new()
plot(cev, xlab = "Principal Component", ylab = "Cumulative Explained Variance")

# find number of components that explains 95% of the variance
n_components = which(cev > 0.95)[1]

# Perform PCA with the number of components that explains 95% of the variance
pca <- prcomp(X_train, center = TRUE, scale. = TRUE, rank. = n_components)

# apply PCA also to the test set
X_train_pca = predict(pca, X_train)
X_test_pca = predict(pca, X_test)

cv_fit_pca <- cv.glmnet(X_train_pca, Y_train, family = "binomial", alpha = 1)

# Get the best lambda value
best_lambda_pca <- cv_fit_pca$lambda.min
lasso_model_pca <- glmnet(X_train_pca, Y_train, family = "binomial", alpha = 1, lambda = best_lambda_pca)

predictions_pca <- predict(lasso_model_pca, X_test_pca, type = "response")
predicted_classes_pca <- ifelse(predictions_pca > 0.5, 1, 0)
predicted_classes_pca <- as.factor(predicted_classes_pca)

# Confusion matrix
conf_matrix_pca <- confusionMatrix(predicted_classes_pca, Y_test)
plot(conf_matrix_pca$table, col = c("red", "green"), main = "Confusion Matrix", xlab = "Predicted classes", ylab = "True classes")

# ROC Curve
library(pROC)
modelROC_pca <- roc(as.numeric(Y_test),as.numeric(predicted_classes_pca))
dev.new()
plot(modelROC_pca, legacy.axes = TRUE)

# 3D PCA scatterplot
library("scatterplot3d")
dev.new()
scatterplot3d(X_train_pca[,1], y=X_train_pca[,2], z=X_train_pca[,3], color=c("red","green")[Y_train], pch=16, main="PCA scatter", xlab="PC1", ylab="PC2", zlab="PC3")



