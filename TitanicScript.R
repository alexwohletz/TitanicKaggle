train.data <- read.csv("C:/Users/awohl/OneDrive/MultivariateAnalysis/TitanicHW/TitanicTrain.csv", na.string = c("NA",""))

library(Amelia)
library(stringr)
library(party)

#Factorize the survived and Pclass variables for further analysis
train.data$Survived = factor(train.data$Survived)
train.data$Pclass = factor(train.data$Pclass)

#Examine results
str(train.data)

#Check for NAs
sum(is.na(train.data$Age) == TRUE)

#Calculated percentage of missing values
sum(is.na(train.data$Age) == TRUE)/length(train.data$Age)

#Look at the missing values in the dataset using the Amelia missmap function
missmap(train.data, main = "Missing Map")

#Impute missing values for Embarked
train.data$Embarked[which(is.na(train.data$Embarked))] = 'S';

#Do some regex for titles
train.data$Name = as.character(train.data$Name)
table_words = table(unlist(strsplit(train.data$Name, "\\s+")))
sort(table_words[grep('\\.', names(table_words))], decreasing = T)

#Find the missing values in the titles
tb = cbind(train.data$Age, str_match(train.data$Name,"[a-zA-Z]+\\."))
table(tb[is.na(tb[,1]),2])

#Impute missing values in age using an inference based on the titles
mean.mr = mean(train.data$Age[grepl(" Mr\\.",train.data$Name) & !is.na(train.data$Age)])
mean.mrs = mean(train.data$Age[grepl(" Mrs\\.",train.data$Name) & !is.na(train.data$Age)])
mean.dr = mean(train.data$Age[grepl(" Dr\\.",train.data$Name) & !is.na(train.data$Age)])
mean.miss = mean(train.data$Age[grepl(" Miss\\.",train.data$Name) & !is.na(train.data$Age))
mean.master = mean(train.data$Age[grepl(" Master\\.",train.data$Name) & !is.na(train.data$Age)]))                                

train.data$Age[grepl(" Mr\\.",train.data$Name) & is.na(train.data$Age)] = mean.mr
train.data$Age[grepl(" Mrs\\.",train.data$Name) & is.na(train.data$Age)] = mean.mrs
train.data$Age[grepl(" Dr\\.",train.data$Name) & is.na(train.data$Age)] = mean.dr
train.data$Age[grepl(" Miss\\.",train.data$Name) & is.na(train.data$Age)] = mean.miss
train.data$Age[grepl(" Master\\.",train.data$Name) & is.na(train.data$Age)] = mean.master

#Okay now lets look at some descriptive statistics concerning our problem statement
barplot(table(train.data$Survived), main = "Passengers Survived", names = c("Perished","Survived"))
barplot(table(train.data$Pclass), main = "Passenger Class", names = c("First","Second","Third"))
hist(train.data$Age, main = "Passenger Age", xlab = "Age")

#Look at the gender more likely to survive
counts = table(train.data$Survived,train.data$Sex)
barplot(counts, col = c("red","darkgreen"), legend = c("Perished","Survived"), main = "Passenger Survival by Gender")

#Look at the passenger class more likely to survive
counts = table(train.data$Survived,train.data$Pclass)
barplot(counts, col = c("red","black"),legend = c("Perished","Survived"), main = "Passenger survival by class")
#Ages that did not survive
hist(train.data$Age[which(train.data$Survived == "0")], main = "Passenger Age Histogram", xlab = "Age",ylab = "Count", col = "blue", breaks = seq(0,80, by = 2))
#Ages that did survive overlayed
hist(train.data$Age[which(train.data$Survived == "1")], add = T, col = "red", breaks = seq(0,80, by = 2))

#Create some groupings by age
train.child <- train.data$Survived[train.data$Age < 13]
length(train.child[which(train.child == 1)])/length(train.child)

train.youth <- train.data$Survived[train.data$Age >= 15 & train.data$Age < 25]
length(train.youth[which(train.youth==1)])/length(train.youth)

train.adult <- train.data$Survived[train.data$Age >= 20 & train.data$Age < 65]
length(train.adult[which(train.adult==1)])/length(train.adult)

train.senior <- train.data$Survived[train.data$Age >= 65]
length(train.senior[which(train.senior==1)])/length(train.senior)

####################################################################
#Build a decision tree to predict passenger survival

#First split training into a train and test set based on the 70% rule of thumb
split.data = function(data, p = .7,s = 666){
  set.seed(s)
  index = sample(1:dim(data)[1])
  train = data[index[1:floor(dim(data)[1]*p)],]
  test = data[index[((ceiling(dim(data)[1]*p))+1):dim(data)[1]], ]
  return(list(train = train, test = test))
}

#Split
allset = split.data(train.data, p = 0.7)
trainset = allset$train
testset = allset$test

#train a ctree model
train.ctree = ctree(Survived~ Pclass + Sex + Age + SibSp + Fare + Parch + Embarked, data = trainset)
plot(train.ctree, main = "Conditional inference tree of Titanic Dataset")

#train an svm model
require(e1071)
svm.model = svm(Survived ~ Pclass + Sex + Age +SibSp + Fare + Parch + Embarked, data = trainset, probability = T)

#Predict using svm
svm.predict = predict(svm.model,testset)
confusionMatrix(svm.predict, testset$Survived)

#Predict using ctree
require(caret)
ctree.predict = predict(train.ctree, testset)
confusionMatrix(ctree.predict, testset$Survived)

#Use caret to try and optimize results
fit <- trainControl(method = "repeatedcv", number =  8, repeats = 10, adaptive = list(min = 10, alpha = .05, method = "gls", complete = T))
gbm.fit <- train(Survived ~ Pclass + Sex + Age + SibSp + Fare + Parch + Embarked, data = trainset, method = "ctree",trControl = fit)
gmb.pred <- predict(gbm.fit,testset)
confusionMatrix(gmb.pred,testset$Survived)
