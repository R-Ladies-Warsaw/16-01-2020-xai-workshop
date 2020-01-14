#' No. 1
#' In that part I would like to answer the question why do we need wrapper over models based on simple example. On the top of that
#' we will create some models in order to use them in further sections. Please ask questions when You meet any problem with either
#' understanding or executing codes.
library("mlr")
library("ranger")
library("DALEX")

data <- titanic_imputed
data$survived <- as.factor(data$survived)
task_classif_titanic <- makeClassifTask(id = "titanic",
                                        data = data,
                                        target = "survived")
lrn_classif_titanic <- makeLearner("classif.ranger", predict.type = "prob")
model_mlr_classif_titanic <- train(lrn_classif_titanic, task_classif_titanic)
p_mlr_target <- predict(model_mlr_classif_titanic, newdata = data)

p_mlr_target

p_mlr_target$data

p_mlr <- predict(model_mlr_classif_titanic, newdata = data[,-which(names(data) == "survived")])

p_mlr$data

#' As You can see p_mlr$data and p_mlr_target$data are different are different object, despie the fact that they were created
#' using the same model.

model_ranger_classif_titanic <- ranger(survived~., data = data, probability = TRUE)
p_ranger <- predict(model_ranger_classif_titanic, data = data)

p_ranger
p_ranger$predictions

#' p_ranger and p_mlr should be the same object, returned by predict function. Unfortunatley in R we have no unification between
#' machine learning models and the way they perform things.


#' No. 2
#' Therefore we need wrapper over models, that will allow as to use any machine learning model in the say way and therefore perform 
#' the same computations for any type of task. 

explainer_ranger_titanic <- explain(model_ranger_classif_titanic, data = data, y = data$survived)

#' As we can see there is plenty of warnings, let's try to solve them.
#' First of all, it is nor recommended to have `y` in `data`. The reason is that some of the applications may not work properly.
#' Today we will not have occasion to meet that problem, but I put it here for purpose of good practice.

explainer_ranger_titanic <- explain(model_ranger_classif_titanic, data = data[,-8], y = data$survived)

#' Warning about `y` being a facor is way more important. We have to way to solv that problem. First is presented below, second 
#' is connected to residual function and we will see it later. Keep in mind that `y` parameter is just a vector. Changing it
#' does not affect original data at all.

explainer_ranger_titanic <- explain(model_ranger_classif_titanic, data = data[,-8], y = as.numeric(as.character(data$survived)),
                                    label = "Ranger")

#' At that point we have no warnings. Let's inspect properites of models used in our explainer.

explainer_ranger_titanic$model_info
explainer_ranger_titanic$model

#' Next parameter that I would like to bring closer to You is `predict_function`. To to so, I will use one already existing model, and
#' create one more.
library("readr")
fifa19 <- as.data.frame(read_csv("fifa19.csv"))

fifa19$Value <- substr(fifa19$Value,2,200)
fifa19$ValueNum <- sapply(as.character(fifa19$Value), function(x) {
  unit <- substr(x, nchar(x), nchar(x))
  if (unit == "M") return (as.numeric(substr(x, 1, nchar(x)-1)) * 1000000)
  if (unit == "K") return (as.numeric(substr(x, 1, nchar(x)-1)) * 1000)
  as.numeric(x)
})

rownames(fifa19) <- make.names(fifa19$Name, unique = TRUE)
fifa19_selected <- fifa19[,c(4,8,14:18,55:88,90)]
fifa19_selected$`Preferred Foot` <- factor(fifa19_selected$`Preferred Foot`)
fifa19_selected$ValueNum <- sqrt(fifa19_selected$ValueNum)

fifa19_selected <- na.omit(fifa19_selected)
colnames(fifa19_selected) <- make.names(colnames(fifa19_selected))

library("gbm")
fifa_gbm <- gbm(ValueNum~.-Overall, data = fifa19_selected, n.trees = 250, interaction.depth = 3)

#' As we can see in libe 78, target predicted by our model is a squre root o actuall value. Of course we would like to revert
#' that transofrmation before creating explanations. Answer for that problem is `predict_function`. We just have to pass to `explain`
#' that will overwrite deafult `predict_function` for gbm models.

predict_fifa <- function(X.model, newdata) {
  (predict(X.model, newdata, n.trees = 250))^2
}

explainer_fifa <- explain(fifa_gbm, data = fifa19_selected[,-42], y = (fifa19_selected$ValueNum)^2, predict_function = predict_fifa)

#' Second example reffers to mlr model created in the first paragraph. DALEX does not have deafult predict_function for mlr model 
#' objects. Therfore explainer will not work properly (althought it will create)

explainer_mlr_titanic <- explain(model_mlr_classif_titanic, data = data[,-8], y = as.numeric(as.character(data$survived)))

#' As message says, predict functions returns an error when executed. It's because plain `predict` was used and it is not 
#' understandable for DALEX (as we have seen in the first paragraph). So we have to provide our own predict function. 

predict_mlr <- function(X.model, newdata) {
  predict(X.model, newdata = newdata)$data[,2]
}

explainer_mlr_titanic <- explain(model_mlr_classif_titanic, data = data[,-8], y = as.numeric(as.character(data$survived)),
                                 predict_function = predict_mlr)

#' Small spoiler, predict function used in DALEXtra https://github.com/ModelOriented/DALEXtra/blob/master/R/yhat.R#L27-L45

library("DALEXtra")
explainer_mlr_dalextra <- explain_mlr(model_mlr_classif_titanic, data = data[,-8], y = as.numeric(as.character(data$survived)),
                                      label = "mlr RANGER")

#' #' Time to show second solution for factor problem
#' 
#' residual_ranger <- function(model, data, y, predict_function) {
#'   as.numeric(as.character(y)) - predict_function(model, data)
#' }
#' 
#' explainer_ranger_titanic <- explain(model_ranger_classif_titanic, data = data[,-8], y = data$survived, 
#'                                     residual_function = residual_ranger)
#' 
#' Rest of the parameters:
#' - weights
#' - precalculate
#' - verbose
#' - colorize
#' - label

#' PRACTICE TASK
task <- mlr::makeRegrTask(
  id = "R",
  data = apartments,
  target = "m2.price"
)
learner_lm <- mlr::makeLearner(
  "regr.lm"
)
model_lm <- mlr::train(learner_lm, task)
predict(model_lm, newdata = apartmentsTest[,-1])$data

learner_rf <- mlr::makeLearner(
  "regr.randomForest"
)
model_rf <- mlr::train(learner_rf, task)
predict(model_rf, newdata = apartmentsTest[,-1])$data

stacked_model <- list(model_lm, model_rf)
#' Two models are given along with proper function to get prediction. Create explainer for `stacked_model`. For apartments located
#' in srodmiescie district, use `model_lm`, for any other `model_rf`. 






#' Now when we already have explainer, we can try to make use of them. for that purpose we will use iBreakDown and ingredients packages
mp_mlr <- model_performance(explainer_mlr_titanic)
mp_ranger <- model_performance(explainer_ranger_titanic)
plot(model_performance(explainer_mlr_titanic), model_performance(explainer_ranger_titanic))

#' Please make model performance for model_rf and stacked model built in previous task. Plot them together




lrn_classif_titanic_svm <- makeLearner("classif.ksvm", predict.type = "prob")
model_mlr_classif_svm_titanic <- train(lrn_classif_titanic_svm, task_classif_titanic)
explainer_mlr_dalextra_svm <- explain_mlr(model_mlr_classif_svm_titanic, data = data[,-8], y = as.numeric(as.character(data$survived)),
                                          label = "mlr SVM")

fi_ranger <- feature_importance(explainer_mlr_dalextra)
fi_svm <- feature_importance(explainer_mlr_dalextra_svm)
plot(fi_svm)
plot(fi_ranger)

fi_ranger_auc <- feature_importance(explainer_mlr_dalextra, loss_function = loss_one_minus_auc)
fi_svm_auc <- feature_importance(explainer_mlr_dalextra_svm, loss_function = loss_one_minus_auc)
plot(fi_svm)
plot(fi_ranger)

#' loss function structure - function(observed, predicted)

model_multilabel <- ranger(status~., data = HR, probability = TRUE)
explainer_multilabel <- explain(model_multilabel, data = HR, y = as.numeric(HR$status))

fi_ranger_multilabel <- feature_importance(explainer_multilabel, loss_function = loss_cross_entropy)
plot(fi_ranger_multilabel)

#' iBreakDown

library("iBreakDown")
bd_multilabel <- break_down(explainer_multilabel, new_observation = HR[1,])
plot(bd_multilabel)

bd_titanic_svm <- break_down(explainer_mlr_dalextra_svm, new_observation = data[1,])
bd_titanic <- break_down(explainer_mlr_dalextra, new_observation = data[1,])
plot(bd_titanic_svm)
plot(bd_titanic)

#' pdp profiles

vi_svm <- partial_dependency(explainer_mlr_dalextra_svm, variables  = "fare")
vi_ranger <- partial_dependency(explainer_mlr_dalextra, variables  = "fare")
vi_multilabel <- partial_dependency(explainer_multilabel, variables = "hours")
plot(vi_svm, vi_ranger)
plot(vi_multilabel)

#' No. 6 Champion Challenger

task <- mlr::makeRegrTask(
  id = "R",
  data = apartments,
  target = "m2.price"
)
learner_lm <- mlr::makeLearner(
  "regr.lm"
)
model_lm <- mlr::train(learner_lm, task)
explainer_lm <- explain_mlr(model_lm, apartmentsTest, apartmentsTest$m2.price, label = "LM")

learner_rf <- mlr::makeLearner(
  "regr.randomForest"
)
model_rf <- mlr::train(learner_rf, task)
explainer_rf <- explain_mlr(model_rf, apartmentsTest, apartmentsTest$m2.price, label = "RF")



plot_data_regr <- funnel_measure(explainer_lm, explainer_rf,
                            nbins = 5,
                            measure_function = DALEX::loss_root_mean_square)

plot_data_classif <- funnel_measure(explainer_mlr_dalextra, explainer_mlr_dalextra_svm,
                                    nbins = 5,
                                    measure_function = DALEX::loss_one_minus_auc)

data_overall <- overall_comparison(explainer_lm, explainer_rf, type = "regression")

plot(data_overall)

training_test <- training_test_comparison(explainer_lm, explainer_rf, apartments, apartments$m2.price)
plot(training_test)

# TASK
#' Create report using champion_challenger function


#' No. 7 Integracja
