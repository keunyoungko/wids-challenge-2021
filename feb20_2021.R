############ Version 1 ####################
########## Date: Feb 17 ###################
#### Author: Keun Young (Jennifer) Ko ##### 

# Set working directory
setwd("/Volumes/external/wids")

# Reading packages
library(tidyverse)
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
# Reading files
training.data <- read.csv("TrainingWiDS2021.csv")

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

### TRAINING & TESTING ###
# Cross-validation
indexes <- sample(nrow(training.data.spec),
                  size = nrow(training.data.spec) * 0.7)

train.subset <- training.data.spec[indexes, ] #grabbing rows from the indexes
test.subset <- training.data.spec[-indexes, ]

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

## Ensemble
algorithmList <- c("rf", "rpart", "knn", "lda") 
model3 <- caretList(factor(diabetes_mellitus) ~ d1_glucose_max + glucose_apache + d1_glucose_min + bmi + weight + age + d1_creatinine_max + d1_creatinine_min + creatinine_apache,
                    data = training.data.spec2,
                    methodList = algorithmList,
                    trControl = trainControl)

# basic parameter tuning to control for cross-validation // resampling
cv5 <- trainControl(method = "repeatedcv",
                    number = 5,
                    repeats = 5)
model4 <- train(factor(diabetes_mellitus) ~ d1_glucose_max + glucose_apache + d1_glucose_min + bmi + weight + age + d1_creatinine_max + d1_creatinine_min + creatinine_apache,
                data = training.data.spec2,
                method = "ranger",
                tuneGrid = data.frame(mtry = 2,
                                      min.node.size = 1,
                                      splitrule = "extratrees"),
                trControl = cv5)
model4 # accuracy = 0.8124

## Random Forest ##
# Trial 1
rf <- randomForest(factor(diabetes_mellitus) ~ .,
                   data = training.data.spec2,
                   ntree=1000, keep.forest=FALSE,
                   importance=TRUE)
varImpPlot(rf)
# Key variables in random forest include:
# d1_glucose_max, glucose_apache, d1_glucose_min, bmi, weight, age, d1_creatinine_max, d1_creatinine_min, and creatinine_apache

# Trial 2
ind <- sample(2, nrow(training.data.spec2), replace = TRUE, prob=c(0.8, 0.2))
rf.true.vals <- training.data.spec2[ind == 2,]$diabetes_mellitus
rf.tree.train <- randomForest(factor(diabetes_mellitus) ~ d1_glucose_max + glucose_apache + d1_glucose_min + bmi + weight + age + d1_creatinine_max + d1_creatinine_min + creatinine_apache,
                        data=training.data.spec2[ind == 1,]) #train data
rf.tree.preds <- predict(rf.tree.train, training.data.spec2[ind == 2,]) #test data
table(observed = training.data.spec2[ind==2, "diabetes_mellitus"], predicted = rf.tree.preds)
## Get prediction for all trees
rf.tree.preds <- predict(rf.tree, training.data.spec2[ind == 2,], predict.all=TRUE)
mean(rf.tree.preds == rf.true.vals) # accuracy = 0.815827

# Bagging
true.vals <- test.subset$diabetes_mellitus
bagging.model.train <- bagging(factor(diabetes_mellitus) ~ d1_glucose_max + glucose_apache + d1_glucose_min + bmi + weight + age + d1_creatinine_max + d1_creatinine_min + creatinine_apache,
                               data = train.subset)
bagging.preds <- predict(bagging.model.train, test.subset)
mean(bagging.preds == true.vals) # accuracy = 0.7992341

# rpart
rpart.tree.model.train <- rpart(factor(diabetes_mellitus) ~ d1_glucose_max + glucose_apache + d1_glucose_min + bmi + weight + age + d1_creatinine_max + d1_creatinine_min + creatinine_apache,
                          data = train.subset)
rpart.tree.preds <- ifelse(predict(tree.model.train, test.subset)[,2] > 0.5, 1, 0)

mean(bagging.preds == true.vals)
mean(tree.preds == true.vals) # accuracy = 0.8122265

