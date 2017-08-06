
setwd("~/Desktop/vvnanalytics/titanic")

train = read.csv("train.csv", header = TRUE)
test = read.csv("test.csv", header = TRUE)
View(train)
str(train)
str(test)

require(gmodels)
require(vcd)

#relation between Pclass and Survived - to check binary association
CrossTable(train$Pclass,train$Survived,dnn=c("Pclass","Survived"),prop.r = T,prop.c = T,prop.t = T,prop.chisq = T,chisq = T,fisher = T) 
assocstats(table(train$Pclass,train$Survived,dnn=list("Pclass","Survived") ))
#Very Significant Phi = 0.34

CrossTable(train$Sex,train$Survived,dnn=c("Sex","Survived"),prop.r = T,prop.c = T,prop.t = T,prop.chisq = T,chisq = T,fisher = T) 
assocstats(table(train$Sex,train$Survived,dnn=list("Sex","Survived") ))
#Significant Phi = 0.54

CrossTable(train$SibSp,train$Survived,dnn=c("SibSp","Survived"),prop.r = T,prop.c = T,prop.t = T,prop.chisq = T,chisq = T,fisher = T) 
assocstats(table(train$SibSp,train$Survived,dnn=list("SibSp","Survived") ))
#Significant Phi = 0.20

CrossTable(train$Parch,train$Survived,dnn=c("Parch","Survived"),prop.r = T,prop.c = T,prop.t = T,prop.chisq = T,chisq = T,fisher = T) 
assocstats(table(train$Parch,train$Survived,dnn=list("Parch","Survived") ))
#Significant Phi = 0.17

CrossTable(train$Embarked,train$Survived,dnn=c("Embarked","Survived"),prop.r = T,prop.c = T,prop.t = T,prop.chisq = T,chisq = T,fisher = T) 
assocstats(table(train$Embarked,train$Survived,dnn=list("Embarked","Survived") ))
#Significant Phi = 0.18

#Remove unwanted columns

#Remove Columns without data
train_refined = subset(train,select = c(Pclass,Survived, Sex, SibSp, Parch, Embarked))
View(train_refined)
str(train_refined)
require(dummies)
cb<-as.data.frame(dummy(train_refined$Sex))
ch <- as.data.frame(dummy(train_refined$Embarked))
trainDummies <- cbind(train_refined[,c(-3,-6)],cb,ch)
View(trainDummies)


names(trainDummies)[names(trainDummies)=="Sex)female"] <- "Sexfemale"
names(trainDummies)[names(trainDummies)=="Sex)male"] <- "Sexmale"
names(trainDummies)[names(trainDummies)=="Embarked)"] <- "Embarked"
names(trainDummies)[names(trainDummies)=="Embarked)C"] <- "EmbarkedC"
names(trainDummies)[names(trainDummies)=="Embarked)Q"] <- "EmbarkedQ"
names(trainDummies)[names(trainDummies)=="Embarked)S"] <- "EmbarkedS"

trainDummies = subset(trainDummies,select = c(-Embarked))
str(trainDummies)

require(caTools)
set.seed(1234) #setting a common seed ensures reproducibility
split <- sample.split(trainDummies$Survived,SplitRatio = 0.75) 
head(split,20)
Train_split <- subset(trainDummies, split==T)
Test_split <- subset(trainDummies, split==F)

str(Train_split)
str(Test_split)


default<-glm(formula = Survived ~ Pclass + SibSp + Parch + Sexfemale + Sexmale + EmbarkedC + EmbarkedQ + EmbarkedS, family = binomial, data = Train_split)
summary(default)

# Removing Sexmale, Embarked, EmbarkedQ, Embarked S
default1<-glm(formula = Survived ~ Pclass + SibSp + Parch + Sexfemale - Sexmale + EmbarkedC - EmbarkedQ - EmbarkedS, family = binomial, data = Train_split)
summary(default1)

# Removing Sexmale, Embarked, EmbarkedQ, EmbarkedS, Parch
default2<-glm(formula = Survived ~ Pclass + SibSp - Parch + Sexfemale - Sexmale + EmbarkedC - EmbarkedQ - EmbarkedS, family = binomial, data = Train_split)
summary(default2)

list("DefAIC"=default$aic,"Def1AIC"=default1$aic,"Def2AIC"=default2$aic)

#assessing model performance with all variables on test data set cutoff 0.5
CrossTable(Train_split$Survived,ifelse(default2$fitted.values>0.5,1,0),dnn=c("Observed","Predicted"),prop.r = T,prop.c = T,prop.t = T,prop.chisq = F,chisq = F)
assocstats(table(Train_split$Survived,default2$fitted.values,dnn=list("Observed","Predicted") ))

require(ROCR)
ROCRLog <- prediction(default2$fitted.values,Train_split$Survived)
ROCRPerf <- performance(ROCRLog,"tpr","fpr")
plot(ROCRPerf, colorize=T, print.cutoffs.at=seq(0,1,0.1), text.adj=c(-0.2,1.7),lwd=3,
     main="ROC Curve for Predicting Survivors")
ROCRSens <-performance(ROCRLog,"sens","spec")
plot(ROCRSens,colorize=T, print.cutoffs.at=seq(0,1,0.1), text.adj=c(-0.2,1.7),lwd=3,main="Sensitivity Curve for Predicting Survivors")
CrossTable(Train_split$Survived,ifelse(default2$fitted.values>0.3,1,0),dnn=c("Observed","Predicted"),prop.r = T,prop.c = T,prop.t = T,prop.chisq = F,chisq = F)

#sensitivity or True Positive Rate,TP/Total number of Yes 208/256= 0.813
#specificity or True Negative Rate, TN/Total number of No 304/412= 0.738
# False Positive Rate 1- Specificity or TNR =1-0.813 = 0.187
# False Negative Rate 1- Sensitivity or TPR = 1-0.738 =0.262
#Accuracy 512/668 = 0.77

#evaluating model on test data
str(Test_split)
testPred <- data.frame("Observed"=Test_split$Survived,"Predicted"=ifelse(predict(default2,Test_split)>0.1,1,0))
CrossTable(testPred$Observed,testPred$Predicted,prop.r = T,prop.c = T,prop.t = T,prop.chisq = F,chisq = F)
testROC <- prediction(predict(default2,Test_split),Test_split$Survived)
testPerf <- performance(testROC,"tpr","fpr")
testAUC <- performance(testROC,"auc")
plot(testPerf, colorize=T, print.cutoffs.at=0.3, text.adj=c(-0.2,1.7),lwd=3,
     main="ROC Curve for Titanic Survivors \n Test Data")
lines(par()$usr[1:2],par()$usr[3:4]) # 50% line for lift chart
auc <- paste(c("AUC ="),round(as.numeric(performance(testROC,"auc")@y.values),digits=2),sep="")
legend("topleft",auc, bty="n")



######## Testing on Unseen data
#refine Test data
str(test)
test_refined = subset(test,select = c(Pclass,Sex, SibSp, Parch, Embarked))
View(test_refined)
str(test_refined)
require(dummies)
cb<-as.data.frame(dummy(test_refined$Sex))
ch <- as.data.frame(dummy(test_refined$Embarked))
testDummies <- cbind(test_refined[,c(-2)],cb,ch)
View(testDummies)
str(testDummies)

names(testDummies)[names(testDummies)=="Sex)female"] <- "Sexfemale"
names(testDummies)[names(testDummies)=="Sex)male"] <- "Sexmale"
names(testDummies)[names(testDummies)=="Embarked)C"] <- "EmbarkedC"
names(testDummies)[names(testDummies)=="Embarked)Q"] <- "EmbarkedQ"
names(testDummies)[names(testDummies)=="Embarked)S"] <- "EmbarkedS"
str(default2)

log_predict <- predict(default2,newdata = testDummies,type = "response")
log_predict <- ifelse(log_predict > 0.5,1,0)

testDummies$Survived = log_predict
testDummies
write.csv(testDummies,"Titanic.csv")
str(testDummies)
