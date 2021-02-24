############ Version 1 ####################
########## Date: Feb 17 ###################
#### Author: Keun Young (Jennifer) Ko ##### 

# Set working directory
setwd("/Volumes/external/wids")

# Reading packages
library(tidyverse)
library(base)
library(class)
library(data.table)
library(datasets)
library(geojsonio)
library(leaflet)
library(writexl)
library(knitr)
library(lubridate)
library(tidytext)
library(devtools)
library(fmsb)
library(ggplot2)
library(quanteda)
library(stm)
library(scales)
library(DT)
library(xml2)
library(rvest)
library(dplyr)
library(caret)
library(ipred)
library(rvest)
library(stringr)
library(ggrepel)
library(dendextend)
library(binaryLogic)
library(factoextra)
library(rpart)
library(rattle)
library(fastAdaboost)
library(randomForest)
library(ISLR)
library(xtable)
library(MASS)
library(pastecs)
library(extraTrees)
library(caretEnsemble)
library(class)
library(corrplot)
library(Matrix)
# install.packages("mltools")
library(mltools)
# install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
# install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
# install.packages("xgboost")
library(xgboost)
require(xgboost)

### PRELIMINARY STAGE ###
# Reading files
training.data <- read.csv("TrainingWiDS2021.csv")
unlabeled.data <- read.csv("UnlabeledWiDS2021.csv")

### DATA MANIPULATION ###
na.columns <- as.data.frame(sapply(training.data, function(x) sum(is.na(x)))) #counts NA frequency
colnames(na.columns) <- "frequency"
na.columns <- na.columns %>%
  filter(frequency < (130157/2))
colnames(t(na.columns)) # retrieve column names

# Extracting data of columns with < 50% of NA values
training.data.spec <- training.data[c("X","encounter_id","hospital_id","age","bmi","elective_surgery",
                "ethnicity","gender","height","hospital_admit_source","icu_admit_source",
                "icu_id","icu_stay_type","icu_type","pre_icu_los_days",
                "readmission_status","weight","apache_2_diagnosis",       
                "apache_3j_diagnosis","apache_post_operative","arf_apache",                 
                "bun_apache","creatinine_apache","gcs_eyes_apache",            
                "gcs_motor_apache","gcs_unable_apache","gcs_verbal_apache",     
                "glucose_apache","heart_rate_apache","hematocrit_apache",          
                "intubated_apache","map_apache","resprate_apache",            
                "sodium_apache","temp_apache","urineoutput_apache",         
                "ventilated_apache","wbc_apache","d1_diasbp_max",              
                "d1_diasbp_min","d1_diasbp_noninvasive_max","d1_diasbp_noninvasive_min",  
                "d1_heartrate_max","d1_heartrate_min","d1_mbp_max",                 
                "d1_mbp_min","d1_mbp_noninvasive_max","d1_mbp_noninvasive_min",     
                "d1_resprate_max","d1_resprate_min","d1_spo2_max",                
                "d1_spo2_min","d1_sysbp_max","d1_sysbp_min",               
                "d1_sysbp_noninvasive_max","d1_sysbp_noninvasive_min","d1_temp_max",                
                "d1_temp_min","h1_diasbp_max","h1_diasbp_min",              
                "h1_diasbp_noninvasive_max","h1_diasbp_noninvasive_min","h1_heartrate_max",           
                "h1_heartrate_min","h1_mbp_max","h1_mbp_min",                 
                "h1_mbp_noninvasive_max","h1_mbp_noninvasive_min","h1_resprate_max",           
                "h1_resprate_min","h1_spo2_max","h1_spo2_min",                
                "h1_sysbp_max","h1_sysbp_min","h1_sysbp_noninvasive_max",   
                "h1_sysbp_noninvasive_min","h1_temp_max","h1_temp_min",                
                "d1_bun_max","d1_bun_min","d1_calcium_max",             
                "d1_calcium_min","d1_creatinine_max","d1_creatinine_min",          
                "d1_glucose_max","d1_glucose_min","d1_hco3_max",                
                "d1_hco3_min","d1_hemaglobin_max","d1_hemaglobin_min",          
                "d1_hematocrit_max","d1_hematocrit_min","d1_platelets_max",           
                "d1_platelets_min","d1_potassium_max","d1_potassium_min",           
                "d1_sodium_max","d1_sodium_min","d1_wbc_max",                 
                "d1_wbc_min","aids","cirrhosis",                  
                "hepatic_failure","immunosuppression","leukemia",                   
                "lymphoma","solid_tumor_with_metastasis", "diabetes_mellitus")]

# Drop any rows that have NA values for accuracy purposes               
training.data.spec <- training.data.spec %>%
  drop_na() 
exclude_cols <- c("X", "encounter_id", "hospital_id", "icu_id")
training.data.spec2 = training.data.spec[!names(training.data.spec) %in% exclude_cols]
training.data.spec2$diabetes_mellitus = as.binary(training.data.spec2$diabetes_mellitus)

### CORRELATIONS ###
# corr <- as.data.frame(cor(training.data.spec2[sapply(training.data.spec2, is.numeric)]))
corrplot(cor(training.data.spec2[sapply(training.data.spec2, is.numeric)]))
plot.new()
corr <- cor(training.data.spec2[sapply(training.data.spec2, is.numeric)], method = "pearson", use = "complete.obs")

### TRAINING & TESTING ###
# Creating training and testing data
indexes <- sample(nrow(training.data.spec2),
                  size = nrow(training.data.spec2) * 0.7)

train.subset <- training.data.spec2[indexes, ] #grabbing rows from the indexes
test.subset <- training.data.spec2[-indexes, ]

## METHODS ##
# 1. Random forest
model.trial <- train(factor(diabetes_mellitus) ~ ., 
               data = training.data.spec2,
               method = "ranger", #ranger for randomforest
               tuneLength = 5) 
model$bestTune 
plot(model) #extratrees renders on average higher accuracy (boostrap) than gini. the smaller the number of randomly selected predictors, the more accurate the prediction.

tune.grid <- expand.grid(mtry = 1:2,
                         splitrule = c("gini", "extratrees"),
                         min.node.size = 1)

model2 <- train(factor(diabetes_mellitus) ~ d1_glucose_max + glucose_apache + d1_glucose_min + bmi + weight + age + d1_creatinine_max + d1_creatinine_min + creatinine_apache,
                data = training.data.spec2,
                method = "ranger",
                tuneGrid = tune.grid)
model2

# 1a. Finding key variables that best determine diabetes mellitus diagnosis
rf <- randomForest(factor(diabetes_mellitus) ~ .,
                   data = training.data.spec2,
                   ntree=1000, keep.forest=FALSE,
                   importance=TRUE)
varImpPlot(rf)
# Key variables in random forest include:
# d1_glucose_max, glucose_apache, d1_glucose_min, bmi, weight, age, d1_creatinine_max, d1_creatinine_min, and creatinine_apache

# 1b. Rerunning random forests, now using different train and test data sets
ind <- sample(2, nrow(training.data.spec2), replace = TRUE, prob=c(0.7, 0.3))
rf.true.vals <- training.data.spec2[ind == 2,]$diabetes_mellitus
rf.tree.train <- randomForest(factor(diabetes_mellitus) ~ d1_glucose_max + glucose_apache + d1_glucose_min + bmi + weight + age + d1_creatinine_max + d1_creatinine_min + creatinine_apache,
                              data=as.data.frame(lapply(training.data.spec2[ind == 1,], unlist))) #train data
rf.tree.preds <- predict(rf.tree.train, training.data.spec2[ind == 2,]) #test data
table(observed = training.data.spec2[ind == 2, "diabetes_mellitus"], predicted = rf.tree.preds)
rf.tree.preds <- predict(rf.tree.train, training.data.spec2[ind == 2,], predict.all=TRUE) 
mean(rf.tree.preds == rf.true.vals) # accuracy = 0.815827

# 2. Extratrees
# Basic parameter tuning to control for cross-validation // resampling
cv5 <- trainControl(method = "repeatedcv",
                    number = 5,
                    repeats = 5)
extratrees.model <- train(factor(diabetes_mellitus) ~ d1_glucose_max + glucose_apache + d1_glucose_min + bmi + weight + age + d1_creatinine_max + d1_creatinine_min + creatinine_apache,
                data = as.data.frame(lapply(training.data.spec2, unlist)),
                method = "ranger",
                tuneGrid = data.frame(mtry = 2,
                                      min.node.size = 1,
                                      splitrule = "extratrees"),
                trControl = cv5)
extratrees.model # accuracy = 0.8135571

# 3. Bagging
true.vals <- test.subset$diabetes_mellitus
bagging.model.train <- bagging(factor(diabetes_mellitus) ~ d1_glucose_max + glucose_apache + d1_glucose_min + bmi + weight + age + d1_creatinine_max + d1_creatinine_min + creatinine_apache,
                               data = train.subset)
bagging.preds <- predict(bagging.model.train, test.subset)
mean(bagging.preds == true.vals) # accuracy = 0.7992341

# 4. rpart
rpart.tree.model.train <- rpart(factor(diabetes_mellitus) ~ d1_glucose_max + glucose_apache + d1_glucose_min + bmi + weight + age + d1_creatinine_max + d1_creatinine_min + creatinine_apache,
                          data = train.subset)
rpart.tree.preds <- ifelse(predict(tree.model.train, test.subset)[,2] > 0.5, 1, 0)

mean(bagging.preds == true.vals)
mean(rpart.tree.preds == true.vals) # accuracy = 0.8122265

# 5. k-Nearest Neighbors (kNN): not a quantitatively viable method due to its abnormally high classification error rate; Reference: https://daviddalpiaz.github.io/r4sl/knn-class.html 
# Training data
train.subset.knn <- train.subset %>%
  dplyr::select(diabetes_mellitus, d1_glucose_max, glucose_apache, d1_glucose_min, bmi, weight, age, d1_creatinine_max, d1_creatinine_min, creatinine_apache)
x_knn_trn = train.subset.knn[,-1]
y_knn_trn = unlist(train.subset.knn[,10]) # note: y must be a vector
# Testing data
test.subset.knn <- test.subset %>%
  dplyr::select(diabetes_mellitus, d1_glucose_max, glucose_apache, d1_glucose_min, bmi, weight, age, d1_creatinine_max, d1_creatinine_min, creatinine_apache)

x_knn_tst = test.subset.knn[,-1]
y_knn_tst = unlist(test.subset.knn[,10])

head(knn(train = x_knn_trn,
         test = x_knn_tst,
         cl = y_knn_trn,
         k = 3))
typeof(y_knn_trn)

calc_class_err = function(actual, predicted) {
  mean(actual != predicted)
}

calc_class_err(actual    = y_knn_tst,
               predicted = knn(train = x_knn_trn,
                               test  = x_knn_tst,
                               cl    = y_knn_trn,
                               k     = 2))
set.seed(42)
k_to_try = 1:100
err_k = rep(x = 0, times = length(k_to_try))

for (i in seq_along(k_to_try)) {
  pred = knn(train = x_knn_trn,
             test  = x_knn_tst,
             cl    = y_knn_trn,
             k     = k_to_try[i])
  err_k[i] = calc_class_err(y_knn_tst, pred)
}

# plot error vs choice of k
plot(err_k, type = "b", col = "dodgerblue", cex = 1, pch = 20, 
     xlab = "k, number of neighbors", ylab = "classification error",
     main = "(Test) Error Rate vs Neighbors")
# add line for min error seen
abline(h = min(err_k), col = "darkorange", lty = 3)
# add line for minority prevalence in test set
abline(h = mean(y_knn_tst == "Yes"), col = "grey", lty = 2)

# 6. LDA
lda.data <- as.data.frame(lapply(training.data.spec2, unlist))
lda.model <- lda(factor(diabetes_mellitus) ~ d1_glucose_max + glucose_apache + d1_glucose_min + bmi + weight + age + d1_creatinine_max + d1_creatinine_min + creatinine_apache,
                   data = as.data.frame(lapply(train.subset, unlist)))
lda.preds <- lda.model %>%
  predict(test.subset)
mean(lda.preds$class == test.subset$diabetes_mellitus) # accuracy = 0.8068927

# 7. XGBoost
sparse.matrix <- sparsify(training.data.spec2, with = FALSE)

## CONCLUSION ##
# Best methods in the order of accuracy: 
# 1. Random Forest 0.815827
# 2. Extratrees 0.8135571
# 3. rpart 0.8122265
# 4. LDA 0.8068927
# 4. Bagging 0.7992341

### IMPLEMENTATION ### 
# Using both random forest and extratrees to maximize the accuracy rate
rf.final <- randomForest(factor(diabetes_mellitus) ~ d1_glucose_max + glucose_apache + d1_glucose_min + bmi + weight + age + d1_creatinine_max + d1_creatinine_min + creatinine_apache,
                         data=as.data.frame(lapply(training.data.spec2, unlist)),
                         ntree = 1000, keep.forest = FALSE, importance = TRUE)
rf.tree.preds <- predict(rf.final, unlabeled.data) 

predict(extratrees.model, unlabeled.data)
