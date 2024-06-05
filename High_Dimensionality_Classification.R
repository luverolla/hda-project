library(glmnet) # general linear models
library(caret) # train/test split
library(magrittr) # for the pipe operator
library(dplyr) # for data manipulation

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

# >>> Plain linear regression ---
linear_model <- glmnet(X_train, Y_train, family = "binomial", alpha=0, lambda=0);

predictions <- predict(linear_model, X_test, type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
predicted_classes <- as.factor(predicted_classes)

# Confusion matrix
conf_matrix <- confusionMatrix(predicted_classes, Y_test)
print(conf_matrix)
plot(conf_matrix$table, col = c("red", "#0b872c"), main = "Confusion Matrix", xlab = "Predicted classes", ylab = "True classes")

# ROC Curve
modelROC <- roc(as.numeric(Y_test),as.numeric(predicted_classes))
modelROC
plot(modelROC, legacy.axes = TRUE)




# >>> Lasso Logistic regressions ---- 

cv_fit <- cv.glmnet(X_train, Y_train, family = "binomial", alpha = 1)
best_lambda <- cv_fit$lambda.min

# Train the final model using the best lambda
lasso_model <- glmnet(X_train, Y_train, family = "binomial", alpha = 1, lambda = best_lambda)

beta = lasso_model$beta

# print only beta that are not zero
#print(beta)

# Predict on the test data
predictions <- predict(lasso_model, X_test, type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
predicted_classes <- as.factor(predicted_classes)

# Confusion matrix
conf_matrix <- confusionMatrix(predicted_classes, Y_test)
print(conf_matrix)
plot(conf_matrix$table, col = c("red", "#0b872c"), main = "Confusion Matrix", xlab = "Predicted classes", ylab = "True classes")

# ROC Curve
library(pROC)
modelROC <- roc(as.numeric(Y_test),as.numeric(predicted_classes))
modelROC
plot(modelROC, legacy.axes = TRUE)


# >>> Ridge Logistic regressions ---- 

cv_fit <- cv.glmnet(X_train, Y_train, family = "binomial", alpha = 0)
best_lambda <- cv_fit$lambda.min

# Train the final model using the best lambda
lasso_model <- glmnet(X_train, Y_train, family = "binomial", alpha = 0, lambda = best_lambda)

beta = lasso_model$beta

# print only beta that are not zero
#print(beta)

# Predict on the test data
predictions <- predict(lasso_model, X_test, type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
predicted_classes <- as.factor(predicted_classes)

# Confusion matrix
conf_matrix <- confusionMatrix(predicted_classes, Y_test)
print(conf_matrix)
plot(conf_matrix$table, col = c("red", "#0b872c"), main = "Confusion Matrix", xlab = "Predicted classes", ylab = "True classes")

# ROC Curve
library(pROC)
modelROC <- roc(as.numeric(Y_test),as.numeric(predicted_classes))
modelROC
plot(modelROC, legacy.axes = TRUE, xlim=c(1,0), ylim=c(0,1))


# >>> Feature analysis for Lasso ---
# get only features whose index correspond to a non-zero beta
X_train_lasso = X_train[, which(beta != 0)]

# scatterplot between feature 1 and 2
plot(X_train_lasso[,1], X_train_lasso[,2], col=c("red","#0b872c")[Y_train], pch=16, xlab = "Feature 1", ylab = "Feature 2")

# scatterplot between feature 2 and 3
plot(X_train_lasso[,2], X_train_lasso[,3], col=c("red","#0b872c")[Y_train], pch=16, xlab = "Feature 2", ylab = "Feature 3")

# scatterplot between feature 1 and 10
plot(X_train_lasso[,1], X_train_lasso[,10], col=c("red","#0b872c")[Y_train], pch=16, xlab = "Feature 1", ylab = "Feature 10")

# pair plots of first 10 features
dev.new()
pairs(X_train_lasso[,1:5], col=c("red","#0b872c")[Y_train])


# >>> PCA ----
library(FactoMineR) # multivariate analysis
library(factoextra) # complement of the above

# Perform PCA
pca <- prcomp(X, center = TRUE, scale. = TRUE)

cev = cumsum(pca$sdev^2 / sum(pca$sdev^2))
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
modelROC <- roc(as.numeric(Y_test),as.numeric(predicted_classes_pca))
modelROC
plot(modelROC, legacy.axes = TRUE)

# 3D PCA scatterplot
library("scatterplot3d")
scatterplot3d(X_train_pca[,1], y=X_train_pca[,2], z=X_train_pca[,3], color=c("red","green")[Y_train], pch=16, main="PCA scatter", xlab="PC1", ylab="PC2", zlab="PC3")



