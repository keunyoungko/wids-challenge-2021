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
model <- train(factor(diabetes_mellitus) ~ ., 
               data = training.data.spec,
               method = "ranger",
               tuneLength = 5) 
var
# Random Forest

rf <- randomForest(factor(diabetes_mellitus) ~ .,
                   data = training.data.spec2,
                   ntree=1000, keep.forest=FALSE,
                   importance=TRUE)
varImpPlot(rf)
# Cross-validation

indexes <- sample(nrow(training.data.spec),
                  size = nrow(training.data.spec) * 0.7)

train.subset <- training.data.spec[indexes, ] #grabbing rows from the indexes
test.subset <- training.data.spec[-indexes, ]
