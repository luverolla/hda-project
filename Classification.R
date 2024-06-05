##################################
###  DIABETES CLASSIFICATION   ###
##################################

#Packages installation
install.packages("tidyverse")
install.packages("e1071")
install.packages("caTools")
install.packages("caret")
install.packages("klaR")
install.packages("mvtnorm")
install.packages("class")
install.packages("gridExtra")

#Packages loading
library(tidyverse)
library(ggplot2)
library(e1071)
library(caTools)
library(caret)
# library(klaR)
library(mvtnorm)
library(class)
library(gridExtra)
library(pROC)

#Set working directory
#getwd()
#setwd('/Users/americoliguori/Desktop/Ame/University/Laurea Magistrale/Second Year/Second Semester/HDA/HdaClassification')
#dir()

#### Loading Data & Dataset Details #### 
Diabetes=read.csv("data/DiabetesClass.csv",header=T,na.strings='')
head(Diabetes)
str(Diabetes)
dim(Diabetes)

summary(Diabetes)

#### Defining the Response Variable ####
count <- table(Diabetes$DiabDiagnosis)
withDiabetes <- count[2]
withoutDiabetes <- count[1]
cat("DiabDiagnosis Count:", withDiabetes, "\nNon-DiabDiagnosis Count:", withoutDiabetes, "\n")

#Removal of rows with NAs values in the DiabDiagnosis column.
Diabetes <- Diabetes[complete.cases(Diabetes$DiabDiagnosis), ]
dim(Diabetes)

#Pairwise Plots
numeric_data <- sapply(Diabetes, is.numeric)
dev.new()
pairs(Diabetes[,numeric_data], pch = 19, cex = 0.1, main = "Pair Plot of Numeric Data")
#Notes about Pairwise Plot:
#some notable relationships include weight and waist, bp.1s and bp.1d, waist and hip, weight and hip.


##########################
#### DATASET CLEANING ####
##########################

#We drop bp.2s and bp.2d because there are too many missing values (262 NAs each)
Diabetes <- Diabetes %>% select(-"bp.2s", -"bp.2d")

#Sobstitute the NAs of chol, hdl e ratio with median computed values
medians_by_DiabDiagnosis <- Diabetes %>%
  group_by(DiabDiagnosis) %>%
  summarise(across(c(chol, hdl, ratio), ~ median(.x, na.rm = TRUE)))

Diabetes <- left_join(Diabetes, medians_by_DiabDiagnosis, by = "DiabDiagnosis", suffix = c("", "_median"))

Diabetes <- Diabetes %>%
  mutate(
    chol = ifelse(is.na(chol), chol_median, chol),
    hdl = ifelse(is.na(hdl), hdl_median, hdl),
    ratio = ifelse(is.na(ratio), chol_median / hdl_median, ratio)
  ) %>%
  select(-ends_with("_median")) 

#################################
#### HISTOGRAM VISUALIZATION ####
#################################

#Analyze the distribution of each variable according to the presence or not of Diabete

#Function for create histograms 
create_histogram <- function(variable) {
  if (!(is.factor(Diabetes[[variable]]) || is.character(Diabetes[[variable]]))) {  # Exclude discrete variables
    ggplot(Diabetes, aes(x = !!sym(variable), fill = DiabDiagnosis)) +
      geom_histogram(aes(y = after_stat(density)), color = 'black', alpha = 0.5, bins = 30) +
      geom_density(color = 'black') +
      labs(x = variable, y = 'Density', fill = 'DiabDiagnosis', title = paste('Distribution of', variable)) +
      theme_bw() +
      theme(plot.title = element_text(hjust = 0.5),
            legend.position = 'top',
            legend.justification = 'right')
  } else {
    return(NULL)  # Return NULL for discrete variables
  }
}

#Creation of histograms for each variable
histograms <- lapply(names(Diabetes)[-1], create_histogram)

# Filter out NULL values (for discrete variables)
histograms <- histograms[!sapply(histograms, is.null)]

# Visualization of histograms in a grid layout
dev.new()
grid.arrange(grobs = histograms, ncol = 3)


##################################
####      CLASSIFICATION      ####
##################################

set.seed(2024)

Diabetes$frame <- as.numeric(factor(Diabetes$frame, levels = c('small', 'medium', 'large')))
Diabetes <- Diabetes[, !(names(Diabetes) %in% c('id', 'location', 'height', 'bp.1d', 'time.ppn', 'gender'))]
Diabetes$DiabDiagnosis=factor(Diabetes$DiabDiagnosis)

# Split the data into training and testing sets
splitIndex <- createDataPartition(Diabetes$DiabDiagnosis, p = 0.7, list = FALSE)
train_data <- Diabetes[splitIndex, ]
test_data <- Diabetes[-splitIndex, ]

# Check for and handle missing values
train_data <- train_data %>% na.omit()
test_data <- test_data %>% na.omit()

# Define the models
models <- list(
  'Gaussian Naive Bayes' = naiveBayes(DiabDiagnosis ~ ., data = train_data),
  'Logistic Regression' = glm(DiabDiagnosis ~., data = train_data, family = binomial),
  'k-NN Classification' = knn(train = train_data[, -ncol(train_data)], test = test_data[, -ncol(test_data)], cl = train_data$DiabDiagnosis, k = 5)
)

# Create a data frame to store the results
results_Diabetes <- data.frame(Model = character(),
                               Accuracy = numeric(),
                               Precision = numeric(),
                               Recall = numeric(),
                               F1_Score = numeric(),
                               stringsAsFactors = FALSE)

# Train and evaluate models
for (model_name in names(models)) {
  model <- models[[model_name]]
  
  if (model_name == 'Gaussian Naive Bayes') {
    # Make predictions on the test set
    y_pred <- as.factor(predict(model, newdata = test_data, type = 'class'))
  } else if (model_name == 'Logistic Regression') {
    y_pred <- factor(predict(model, newdata = test_data, type = "response") > 0.5, levels = c(FALSE, TRUE), labels = c("No", "Yes"))
  } else if (model_name == 'k-NN Classification') {
    y_pred <- model
  }
  
  # Calculate evaluation metrics
  cm <- confusionMatrix(y_pred, test_data$DiabDiagnosis)
  accuracy <- cm$overall['Accuracy'] * 100
  precision <- cm$byClass['Pos Pred Value'] * 100
  recall <- cm$byClass['Sensitivity'] * 100
  f1 <- cm$byClass['F1'] * 100
  
  # Print results
  cat(paste0(model_name, ":\n"))
  cat(paste0("  Accuracy: ", accuracy, "\n"))
  cat(paste0("  Precision: ", precision, "\n"))
  cat(paste0("  Recall: ", recall, "\n"))
  cat(paste0("  F1-Score: ", f1, "\n"))
  
  # Add results to the data frame
  results_Diabetes <- rbind(results_Diabetes, data.frame(Model = model_name,
                                                         Accuracy = accuracy,
                                                         Precision = precision,
                                                         Recall = recall,
                                                         F1_Score = f1))
  
  # Print Confusion Matrix
  cat("  Confusion Matrix:\n")
  print(table(y_pred, test_data$DiabDiagnosis))
  cat("\n")
}

# Print final results
print(results_Diabetes)
