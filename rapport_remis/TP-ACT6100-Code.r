
library(dplyr)

# importer les donnees

vins <- read.csv("/Users/gabydupre/Desktop/ACT6100/wines_SPA.csv",sep = ";")

# Regrouper la variable a predire en 3 categories

vins$quality <- ifelse(vins$rating < 4.5, "satisfaisant", ifelse(vins$rating < 4.8, "bon", "excellent"))

table(vins$quality)

# nettoyage des donnees

# year
vins$year <- as.numeric(vins$year)

summary(vins$year)
vins$year_clean <- ifelse(is.na(vins$year), 2015, vins$year)
summary(vins$year_clean)

# body
table(vins$body, useNA = "ifany")

body_clean_data <- vins %>%
  group_by(type, body, acidity) %>% 
  na.omit() %>% 
  count()

vins <- right_join(vins, body_clean_data[, 1:3], by = "type") %>% 
  rename(body_clean = body.y,
         acidity_clean = acidity.y) %>% 
  select(-body.x,
         -acidity.x)
summary(vins$acidity_clean)

vins$body_clean <- as.factor(vins$body_clean)
vins$acidity_clean <- as.factor(vins$acidity_clean)

########################## 1er modele ##################################
############################# knn ######################################

library(mlr)
library(tidyverse)
set.seed(1234)

vins_clean <- as_tibble(vins[, c("num_reviews", "price", "year_clean", "body_clean", "acidity_clean", "quality")])
vins_clean$body_clean <- as.numeric(vins_clean$body_clean)
vins_clean$acidity_clean <- as.numeric(vins_clean$acidity_clean)

vins_clean$quality <- as.factor(vins_clean$quality)

train_sample <- sample(6955, 0.9*6955)
vins_train <- vins_clean[train_sample, ]
vins_test <- vins_clean[- train_sample, ]

# les proportions des satisfaisants/bons/excellents vins dans les 3 groupes sont semblables

prop.table(table(vins_train$quality))
prop.table(table(vins_test$quality))
prop.table(table(vins$quality))

# definir la tache

vinsTask <- makeClassifTask(data = vins_train, target = "quality")
vinsTask

# trouver le meilleur k

knnParamSpace <- makeParamSet(makeDiscreteParam("k", values = 5:20))

gridSearch <- makeTuneControlGrid()
cvForTuning <- makeResampleDesc(method = "RepCV", folds = 10, reps = 20)
tunedK <- tuneParams("classif.knn", task = vinsTask, resampling = cvForTuning, par.set = knnParamSpace, control = gridSearch)

knnTuningData <- generateHyperParsEffectData(tunedK)
knnTuningData$data

plotHyperParsEffect(knnTuningData, x = "k", y = "mmce.test.mean", plot.type = "line")


# definir l'apprenant

knn <- makeLearner("classif.knn", par.vals = tunedK$x)


# entrainer le modele 

knnModel <- train(learner = knn, task = vinsTask)

knnPred <- predict(knnModel, newdata = vins_test)

calculateConfusionMatrix(knnPred, relative = T)

performance(knnPred, measure = list(mmce, acc))

########################### 2e modele ##################################
########################## foret aleatoire #############################

library(C50)
library(gmodels)

vins_modele2 <- C5.0(vins_train[-6], vins_train$quality)
summary(vins_modele2)

vins_pred <- predict(vins_modele2, vins_test)

CrossTable(vins_test$quality, vins_pred, prop.chisq = F, prop.c = F, prop.r = F,
           dnn = c("actual quality", "predicted quality"))

# ameliorer le modele en ajouter le boosting

vins_boost <- C5.0(vins_train[-6], vins_train$quality, trials = 10) 
vins_boost
summary(vins_boost)

vins_boost_pred <- predict(vins_boost, vins_test)

CrossTable(vins_test$quality, vins_boost_pred, prop.chisq = F, prop.c = F, prop.r = F,
           dnn = c("actual quality", "predicted quality"))












