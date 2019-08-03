-------------------------------------------------------------------------------------------------------------
# Libraries ====
-------------------------------------------------------------------------------------------------------------
library(tidyverse)        # Package for tidying data
library(VIM)              # Visualizing and imputing missing values
library(Hmisc)            # for descriptive statistics
library(forecast)         # forcasting package
library(broom)            # Tidy statistical summary output
library(knitr)            # report generation
library(plotly)           # visualization
library(imputeTS)         # imputing
library(ggplot2)          # visualization
library(GGally)           # visualization
library(ggfortify)        
library(naniar)           # handling missing values
library(dplyr)            # data manipulation
library(simputation)      # imputing
library(ggcorrplot)       # PLOTTING CORELATIONS
library(FactoMineR)       # Dimensions reduction
library(factoextra)       # visualization for pca
library(rgl)
library(plyr)
library(h2o)              # ml by H2O
library(rstudioapi)       # needed for h2o
library(visdat)           # visualization of missing values
library(naniar)

-----------------------------------------------------------------------------------
# Importing and exploration ====
-------------------------------------------------------------------------------
Data<-read.csv("C:/Users/Constantinos/Desktop/Ubiqum/IOT/Datathon Mushroom/Datathon -Mushroom/data_train1.csv", na.strings="?")
# replaced ? with NA in the impport

# standar check
summary(Data)
str(Data)
head(Data)
ncol(Data)
nrow(Data)
sum(is.na(Data))#181
unique(Data)
ggpairs(Data)

Data(Data$stalk.root) # check if replaced was executed correctly (level "?" droped and values"?" replaced with NA)! Factor levels need to be changed
# check for NAs
mushroom %>% sapply(function(x) {sum(is.na(x))})
# Other option for dealing with NA
#Data[][Data[] == '?' ]<- NA
#sum(is.na(Data))# 1002


# Calculate number of class for each variable
z<-cbind.data.frame(Var=names(Data), Total_Class=sapply(Data,function(x){as.numeric(length(levels(x)))}))
print(z)

# density plots for all features
m.data = Data[,2:23]
m.class = Data[,1]
m.data <- sapply( m.data, function (x) as.numeric(as.factor(x)))

scales <- list(x=list(relation="free"),y=list(relation="free"), cex=0.6)
featurePlot(x=m.data, y=m.class, plot="density",scales=scales,
            layout = c(4,6), auto.key = list(columns = 2), pch = "|")

#relation between variables
a <- cor(Data)
corrplot(a, method = "number", type = "upper")

# calculate poisonous vs edible
class <- plyr::count(Data$class)
print(sprintf("Edible: %d | Poisonous: %d | Percent of poisonous classes: %.1f%%",class$freq[1],class$freq[2], round(class$freq[1]/nrow(Data)*100,1)))

# remove duplicates
Data<- distinct(Data)
nrow(Data)# 4179 no duplicates

# drop x(id column ),odor(rule of the datathon), veil type (is a constant column must be removed)
Data<-Data[,-17]
Data<-Data[,-1]
Data<-Data[,-5]
----------------------------------------------------------------------------------
# Quick Random forest to see importantant variables
--------------------------------------------------------------------------------

set.seed(2018)

quick_RF <- randomForest(x = m, y = Data$class, ntree=20,importance=F)
imp_RF <- importance(quick_RF)
imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]

ggplot(imp_DF[1:20,], aes(x=reorder(Variables, MSE), y=MSE, fill=MSE)) + 
  geom_bar(stat = 'identity') + 
  labs(x = 'Variables', y= '% increase MSE if variable is randomly permuted') + 
  coord_flip() + 
  theme(legend.position="none")

-----------------------------------------------------------------------------------
# Create training and test set ====
set.seed(42)
data_part <-caret::createDataPartition(y=Data$class,
                           p = 0.7, list = F)
test <- Data[-data_part,]  
train <- Data[data_part,]

-------------------------------------------------------------------------------------
# Binary clasification algorythms ====
-------------------------------------------------------------------------------------
# Set up for H2o models
------------------------------------------------------------------------------------------------
# setting parameters
set.seed(42)

# Initialize  h2o cluster
h2o.init(nthreads = -1)

# Convert data to h2o frame
train.h2o <- as.h2o(train) 
test.h2o <- as.h2o(test)

# Identify target and features

#dependent variable (BUILDINGID)
y.dep <- 20 #class

#independent variables (WAPS)
x.indep <- c(1:19) #all features 

---------------------------------------------------------------------------------------------------------------
#  -1) Random Forest  =====
---------------------------------------------------------------------------------------------------------------------  
# Random search grid for Random Forest
  system.time(
    grid.RF <- h2o.grid("randomForest", grid_id = "RF_max",
                        search_criteria = list(
                          strategy = "RandomDiscrete",
                          max_models = 20 
                        ),
                        hyper_params = list(
                          min_rows = c(10,20,5),
                          mtries = c(10, 15, 18),
                          col_sample_rate_per_tree = c(0.75, 0.9, 1.0),
                          sample_rate = c(0.5, 0.7, 0.9),
                          max_depth = c(20, 50, 100),
                          ntrees = c(120,800,250)
                        ),
                        x = x.indep, y = y.dep, training_frame = train.h2o,
                        validation_frame = test.h2o,
                        stopping_tolerance = 0.0001,
                        stopping_rounds = 4,
                        score_tree_interval = 3
    )
  )
grid.RF # print grid  
RFMmodels <- lapply(grid.RF@model_ids, h2o.getModel)
RFMmodels # print all models
RF.best<-h2o.getModel("RF_max_model_18")# best model # get best model
h2o.performance (RF.best) # print performance

predict.RF <- as.data.frame(h2o.predict(RF.best, test.h2o)) # make prediction and save in a dataframe

Res.RF<-caret::postResample(predict.RF, test$class)  # out of sample performance
Res.RF

---------------------------------------------------------------------------------------------------------------------
#  -2) GBM  =====
---------------------------------------------------------------------------------------------------------------------
  # Random search grid for Gradient Boosted Machine
  system.time(
    grid.GBM <- h2o.grid("gbm", grid_id ="GBM",
                         search_criteria = list(
                           strategy = "RandomDiscrete",
                           max_models = 20
                         ),
                         hyper_params = list(
                           max_depth = c(5, 20, 50),
                           min_rows = c(2, 5, 10),
                           sample_rate = c(0.5, 0.8, 0.95, 1.0),
                           col_sample_rate = c(0.5, 0.8, 0.95, 1.0),
                           col_sample_rate_per_tree = c(0.8, 0.99, 1.0),
                           learn_rate = c(0.1),
                           #Placemarker
                           seed = c(701)
                           #Placemarker
                         ),
                         x = x.indep, y = y.dep, training_frame = train.h2o, validation_frame = test.h2o,
                         stopping_tolerance = 0.001,
                         stopping_rounds=3,
                         score_tree_interval = 10,
                         ntrees = 500
    )
  )
grid.GBM
# From grid.GBM grid, first fetch all the models in it with:

GBMmodels <- lapply(grid.GBM@model_ids, h2o.getModel)
GBMmodels
GBM.best<-h2o.getModel("GBM_model_2")#  its no the best model (but had 100%)



h2o.performance (GBM.best)# needs the model( save as GBM.best the best from the grid) does not work with modelID

predict.GBM <- as.data.frame(h2o.predict(GBM.best, test.h2o))

Res.GBM<-caret::postResample(predict.GBM, test$class)
Res.GBM

-------------------------------------------------------------------------------------------------
#  -3) AutoML ====
------------------------------------------------------------------------------------
# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
  Automodel <- h2o.automl(x = x.indep, y = y.dep,
                              training_frame = train.h2o,
                              max_models = 20,
                              seed = 1)

# View the AutoML Leaderboard
Leaderboard <- Automodel@leaderboard
print(Leaderboard, n = nrow(Leaderboard))  # Print all rows instead of default (6 rows)

# The leader model is stored here
BestAutomodel<-Automodel@leader

# If you need to generate predictions on a test set, you can make
# predictions directly on the `"H2OAutoML"` object, or on the leader
# model object directly

pred.Automodel <- as.data.frame(h2o.predict(Automodel@leader, test.h2o))  # predict(aml, test) also works
Res.Automodel<-caret::postResample(pred.Automodel,test$class)
Res.Automodel

# checking the top 6 models that have  AUC 1
-----------------------------------------------------------------------------------
  # Model that acvieved best ACCURACY and KAPPA ====
------------------------------------------------------------------------------------------
Auto.best2<-h2o.getModel("StackedEnsemble_BestOfFamily_AutoML_20190725_142748")
h2o.performance (Auto.best2)
pred.Auto.best2 <- as.data.frame(h2o.predict(Auto.best2, test.h2o))  # predict(aml, test) also works
Res.Auto.best2<-caret::postResample(pred.Auto.best2,test$class)
Res.Auto.best2
---------------------------------------------------------------------------------------------------
# Other top models ====
---------------------------------------------------------------------------------
Auto.best3<-h2o.getModel("GBM_2_AutoML_20190725_142748")
h2o.performance (Auto.best3)
pred.Auto.best3 <- as.data.frame(h2o.predict(Auto.best3, test.h2o))  # predict(aml, test) also works
Res.Auto.best3<-caret::postResample(pred.Auto.best3,test$class)
Res.Auto.best3
---------------------------------------------------------------------------------
Auto.best4<-h2o.getModel("GBM_3_AutoML_20190725_142748")
h2o.performance (Auto.best4)
pred.Auto.best4 <- as.data.frame(h2o.predict(Auto.best4, test.h2o))  # predict(aml, test) also works
Res.Auto.best4<-caret::postResample(pred.Auto.best4,test$class)
Res.Auto.best4
--------------------------------------------------------------------------------
Auto.best5<-h2o.getModel("GBM_4_AutoML_20190725_142748")
h2o.performance (Auto.best5)
pred.Auto.best5 <- as.data.frame(h2o.predict(Auto.best5, test.h2o))  # predict(aml, test) also works
Res.Auto.best5<-caret::postResample(pred.Auto.best5,test$class)
Res.Auto.best5
-------------------------------------------------------------------------------
Auto.best6<-h2o.getModel("DRF_1_AutoML_20190725_142748")
h2o.performance (Auto.best6)
pred.Auto.best6 <- as.data.frame(h2o.predict(Auto.best6, test.h2o))  # predict(aml, test) also works
Res.Auto.best6<-caret::postResample(pred.Auto.best6,test$class)
Res.Auto.best6
--------------------------------------------------------------------------------
Auto.best7<-h2o.getModel("GBM_grid_1_AutoML_20190725_142748_model_4")
h2o.performance (Auto.best7)
pred.Auto.best7 <- as.data.frame(h2o.predict(Auto.best7, test.h2o))  # predict(aml, test) also works
Res.Auto.best7<-caret::postResample(pred.Auto.best7,test$class)
Res.Auto.best7

---------------------------------------------------------------------------------
#  XGBoost model   ====
---------------------------------------------------------------------------------  
# prepare the data to run the model,create xgb.DMatrix objects for each trainand test set
Mtrain <- xgb.DMatrix(as.matrix(train %>% select(-class)), label = train$class)
dMtest <- xgb.DMatrix(as.matrix(test %>% select(-class,-predicted)), label = test$class)
  
# set the XGBoost parameters for the model. We will use a binary logistic objective function. The evaluation metric will be AUC (Area under curve).
# We start with η = 0.012, subsample=0.8, max_depth=8, colsample_bytree=0.9 and min_child_weight=5.  
  
params <- list(
  "objective"           = "binary:logistic",
  "eval_metric"         = "auc",
  "eta"                 = 0.012,
  "subsample"           = 0.8,
  "max_depth"           = 8,
  "colsample_bytree"    =0.9,
  "min_child_weight"    = 5
)  
  
# Train the model using cross variation with 5 folds. We are using a number of rounds equal with 5000, with early stopping criteria for 100 steps. 
# We are also setting the frequency of printing partial results every 100 steps.  
  
nRounds <- 5000
earlyStoppingRound <- 100
printEveryN = 100
model_xgb.cv <- xgb.cv(params=params,
                       data = dMtrain, 
                       maximize = TRUE,
                       nfold = 5,
                       nrounds = nRounds,
                       nthread = 3,
                       early_stopping_round=earlyStoppingRound,
                       print_every_n=printEveryN)

# train
model_xgb <- xgboost(params=params,
                     data = dMtrain, 
                     maximize = TRUE,
                     nrounds = nRounds,
                     nthread = 3,
                     early_stopping_round=earlyStoppingRound,
                     print_every_n=printEveryN)

# predict the test data:
test$predicted <- round(predict(model_xgb ,dMtest),0)
test_xgboost <- test$predicted

plotConfusionMatrix(testset,"Prediction using XGBoost")  
  
