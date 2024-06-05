##################################
###  DIABETES CLASSIFICATION   ###
##################################

#Packages installation
#install.packages("tidyverse")
#install.packages("e1071")
#install.packages("caTools")
#install.packages("caret")
#install.packages("klaR")
#install.packages("mvtnorm")
#install.packages("class")
#install.packages("gridExtra")

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
cat("Diabetic Count:", withDiabetes, "\nNon-Diabetic Count:", withoutDiabetes, "\n")

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
      labs(x = variable, y = 'Density', fill = 'Diabetic', title = paste('Distribution of', variable)) +
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

# Remove id, location, height, bp.1d, time.ppn, gender columns
Diabetes <- Diabetes[, !(names(Diabetes) %in% c('id', 'location', 'height', 'bp.1d', 'time.ppn', 'gender'))]

# Splitting data into train and test data
split <- sample.split(Diabetes, SplitRatio = 0.7)
train_cl <- subset(Diabetes, split == "TRUE")
test_cl <- subset(Diabetes, split == "FALSE")

Diabetes$DiabDiagnosis=factor(Diabetes$DiabDiagnosis)

##################################
####  NAIVE BAYES CLASSIFIER  ####
##################################

# Fitting Naive Bayes model to training data set
classifier_cl <- naiveBayes(DiabDiagnosis ~ ., data = train_cl); classifier_cl

# Predicting on test data
y_pred <- predict(classifier_cl, newdata = test_cl)

# Confusion Matrix
ct <- table(y_pred, test_cl$DiabDiagnosis); ct
confusionMatrix(ct)

# ROC curve
# replace "yes" with 1 and "no" with 0 in the test_cl$DiabDiagnosis
test_lvls <- as.factor(ifelse(test_cl$DiabDiagnosis == "Yes", 1, 0))
naiveROC <- roc(as.numeric(test_lvls),as.numeric(y_pred))
naiveROC
dev.new()
plot(naiveROC, legacy.axes = TRUE, main = "ROC curve - Naive Bayes Classifier", col = "blue", lwd = 2)


######################################
####   LOGISTIC CLASSIFICATION    ####
######################################

# Splitting data into train and test data
split <- sample.split(Diabetes, SplitRatio = 0.7)
train_cl <- subset(Diabetes, split == "TRUE")
test_cl <- subset(Diabetes, split == "FALSE")

Diabetes$DiabDiagnosis=factor(Diabetes$DiabDiagnosis)

# logistic classifier
classifier_logi <- glm(DiabDiagnosis ~., data = train_cl, family=binomial); 
summary(classifier_logi)

# Predicting on test data
y_pred_logi = predict(classifier_logi, type="response", newdata = test_cl)
contrasts(train_cl$DiabDiagnosis) 

# Confusion Matrix
out_logi <- rep("No",nrow(test_cl))
out_logi[y_pred_logi > .5]="Yes"
ct_logi = table(out_logi, test_cl$DiabDiagnosis); ct_logi
mean(out_logi == test_cl$DiabDiagnosis)
confusionMatrix(data=factor(out_logi),reference = test_cl$DiabDiagnosis)

# ROC curve
logiROC <- roc(as.numeric(test_cl$DiabDiagnosis),as.numeric(y_pred_logi))
logiROC
dev.new()
plot(logiROC, legacy.axes = TRUE, main = "ROC curve - Logistic Classifier", col = "blue", lwd = 2)