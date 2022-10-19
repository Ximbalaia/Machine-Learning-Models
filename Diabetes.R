library(tidyverse)
library(corrplot)
library(summarytools)
library(caTools)
library(caret)
library(e1071)
library(class)
library(rpart)
library(randomForest)
library(pROC)

diabetes = read.csv('diabetes_data.csv', sep = ';')

str(diabetes)

matrix = model.matrix(~0+., data = diabetes) %>% cor(use = 'all.obs')

corrplot(matrix, method = 'shade',tl.col = 'black', tl.pos = 'lt', tl.cex = 0.8, cl.cex = 0.8, number.cex = 0.45, addCoef.col = T)

diabetes = diabetes %>% mutate(across(where(is.numeric), factor))
diabetes$age = NULL

diabetes$gender = factor(diabetes$gender, levels = c('Female', 'Male'), labels = c(0, 1))

ctable(diabetes$class, diabetes$gender,  dnn = c('Presence of Diabets', 'Gender'))

ctable(diabetes$class, diabetes$polyuria,  dnn = c('Presence of Diabets', 'Polyuria'))

ctable(diabetes$class, diabetes$polydipsia,  dnn = c('Presence of Diabets', 'Polydipsia'))

ctable(diabetes$class, diabetes$alopecia,  dnn = c('Presence of Diabets', 'Alopecia'))

set.seed(9)
sample = sample.split(diabetes$class, SplitRatio = 0.7)
train_x = subset(diabetes, sample == T)
test_x = subset(diabetes, sample == F)

fit.log = glm(class~., data = train_x, family = 'binomial')
prob.log = predict(fit.log, type = 'response', newdata = test_x)
pred.log = ifelse(prob.log > 0.5, 1, 0) %>% as.factor()
matrix.log = table(test_x$class, pred.log)
matrix.log

acc.log = (89+56)/(89+56+7+4)
sn.log = 89/(89+7)
sp.log = 56/(56+4)
acc.log
sn.log
sp.log

roc(test_x$class, prob.log, plot = T, legacy.axes = T, percent = T)

pred.knn = knn(train = train_x[, -16],
               test = test_x[, -16], 
               cl = train_x[, 16],
               k = 3)

matrix.knn = table(test_x$class, pred.knn)
matrix.knn

acc.knn = (88+57)/(88+57+3+8)
sn.knn = 88/(88+8)
sp.knn = 57/(57+3)
acc.knn
sn.knn
sp.knn

fit.svm = svm(class ~., data = train_x, type = 'C-classification', kernel = 'linear', probability = T)
pred.svm = predict(fit.svm, newdata = test_x)
matrix.svm = table(test_x$class, pred.svm)
matrix.svm

acc.svm = (90+57)/(90+57+3+6)
sn.svm = 90/(90+6)
sp.svm = 57/(57+3)
acc.svm
sn.svm
sp.svm

fit.svm2 = svm(class~., data = train_x, type = 'C-classification', kernel = 'radial')
pred.svm2 = predict(fit.svm2, newdata = test_x)
matrix.svm2 = table(test_x$class, pred.svm2)
matrix.svm2

acc.svm2 = (91+56)/(91+56+5+4)
sn.svm2 = 91/(91+5)
sp.svm2 = 56/(56+4)
acc.svm2
sn.svm2
sp.svm2

fit.nb = naiveBayes(x = train_x[, -16], 
                    y = train_x$class)
pred.nb = predict(fit.nb, newdata = test_x)
matrix.nb = table(test_x$class, pred.nb)
matrix.nb

acc.nb = (82+55)/(82+55+14+5)
sn.nb = 82/(82+14)
sp.nb = 55/(55+5)
acc.nb
sn.nb
sp.nb

fit.dt = rpart(class ~., data = train_x)
pred.dt = predict(fit.dt, newdata = test_x, type = 'class')
matrix.dt = table(test_x$class, pred.dt)
matrix.dt

acc.dt = (91+51)/(91+51+9+5)
sn.dt = 91/(91+5)
sp.dt = 51/(51+9)
acc.dt
sn.dt
sp.dt

fit.rf = randomForest(x = train_x[, -16], y = train_x$class, ntree = 50)
pred.rf = predict(fit.rf, newdata = test_x)
matrix.rf = table(test_x$class, pred.rf)
matrix.rf

acc.dt = (94+57)/(94+57+2+3)
sn.dt = 94/(94+2)
sp.dt = 57/(57+3)
acc.dt
sn.dt
sp.dt

#References
**World Health Organization.** (Website). accessed 19 October 2022. <https://www.who.int/news-room/fact-sheets/detail/diabetes>  
**McLaren Health Case.** (Website). accessed 19 October 2022. <https://www.mclaren.org/lansing/news/early-diabetes-detection-can-prevent-serious-compl-3074>  
**Buderer NM.** Statistical methodology: I. Incorporating the prevalence of disease into the sample size calculation for sensitivity and specificity. Acad Emerg Med. 1996 Sep;3(9):895-900. doi: 10.1111/j.1553-2712.1996.tb03538.x. PMID: 8870764.  
**Hoo, ZH; Candlish, J; Teare, D.** Emerg. Med. J.; 2017;34:357–359.  
**Islam M.M.F., Ferdousi R., Rahman S., Bushra H.Y.** (2020) Likelihood Prediction of Diabetes at Early Stage Using Data Mining Techniques. In: Gupta M., Konar D., Bhattacharyya S., Biswas S. (eds) Computer Vision and Machine Intelligence in Medical Image Analysis. Advances in Intelligent Systems and Computing, vol 992. Springer, Singapore. https://doi.org/10.1007/978-981-13-8798-2_12