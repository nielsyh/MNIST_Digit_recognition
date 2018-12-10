#pr 1.3
library(nnet)
library(dplyr)
library(glmnet)

setwd('C:/Users/Niels/Google Drive/Utrecht University/Vakken Master/Pattern Recognision/assignment 1')
digits.dat <- read.csv(file="mnist.csv")

#[row, column]

# Convert the first column from numeric to factor (categorical).
digits.dat[,1] <- as.factor(digits.dat[,1])


# Features
symmetry <- function (x) 
{
  x.mat <- matrix(x,ncol=28,nrow=28,byrow=T)
  sum(x.mat[,1:14])-sum(x.mat[,15:28])
}

vsymmetry <- function (x) 
{
  x.mat <- matrix(x,ncol=28,nrow=28,byrow=T)
  sum(x.mat[1:14,])-sum(x.mat[15:28,])
}

whitepixels <- function (x){
  whiteness <- 0
  for(i in 1:length(x)){
    if(x[i] == 0){whiteness <- whiteness + 1}
  }
  return(whiteness)
}

blackpixels <- function (x){
  blackness <- 0
  for(i in 1:length(x)){
    if(x[i] > 0){blackness <- blackness + 1}
  }
  return(blackness)
}


# compute sum ink
ink <- rowSums(digits.sample[,-1])
ink <- scale(ink)

# Compute the horizontal symmetry of each pixel image.
horsym <- apply(digits.sample[,-1],1,symmetry)
horsym <- scale(horsym)

#other features
versym <- apply(digits.sample[,-1],1,vsymmetry)
versym <- scale(versym)

whitepxls <- apply(digits.sample[,-1],1, whitepixels)
whitepxls <- scale(whitepxls)

blackpxls <- apply(digits.sample[,-1],1, blackpixels)
blackpxls <- scale(blackpxls)
 

#add features to digits.small (training set)
digits.small$ink = ink
digits.small$horsym = horsym
digits.small$versym = versym
digits.small$whitepxls = whitepxls
digits.small$blackpxls = blackpxls


# Make a dataframe with the class labels (digits) and the two computed features.
digits.df <- data.frame(digit=digits.sample[,1],ink=ink,horsym=horsym, versym=versym,  whitepxls=whitepxls, blackpxls=blackpxls)


#sampling and reducing data
sample.size <- 5000
digits.sample <- sample_n(digits.dat, sample.size)
digits.small <- data.frame(matrix(ncol = 202, nrow = sample.size))
x1 <- c("label")
x2 <- c(1:196)
x3 <- c('ink', 'horsym', 'versym', 'whitepxls', 'blackpxls')
colnames(digits.small) <- c(x1,x2,x3)


# 28*28 to 14*14
for(d in 1:sample.size){
  #digits.small[d,1] <- (digits.sample[d,1])
  elem <- as.numeric(digits.sample[d,1])
  #somehow it increases with 1 magically
  elem <- (elem - 1)
  digits.small[d,1] <- elem
  index.small <-2
  for (i in seq(2, length(digits.sample),56)){
    for(j in seq(i, i+26, 2)){
      digits.small[d,index.small] <- c(rowSums(digits.sample[d,c(j,j+1,j+28,j+29)]))/4
      index.small <- index.small + 1
    }
  }
}


##### making test set
#sampling and reducing data

test.sample.size <- 500
test.sample <- sample_n(digits.dat, test.sample.size)
digits.test <- data.frame(matrix(ncol = 202, nrow = test.sample.size))
x1 <- c("label")
x2 <- c(1:196)
x3 <- c('ink', 'horsym', 'versym', 'whitepxls', 'blackpxls')
colnames(digits.test) <- c(x1,x2,x3)


# compute sum ink
ink <- rowSums(test.sample[,-1])
ink <- scale(ink)

# Compute the horizontal symmetry of each pixel image.
horsym <- apply(test.sample[,-1],1,symmetry)
horsym <- scale(horsym)

#other features
versym <- apply(test.sample[,-1],1,vsymmetry)
versym <- scale(versym)

whitepxls <- apply(test.sample[,-1],1, whitepixels)
whitepxls <- scale(whitepxls)

blackpxls <- apply(test.sample[,-1],1, blackpixels)
blackpxls <- scale(blackpxls)


#add features to digits.small (training set)
digits.test$ink = ink
digits.test$horsym = horsym
digits.test$versym = versym
digits.test$whitepxls = whitepxls
digits.test$blackpxls = blackpxls


# 28*28 to 14*14
for(d in 1:test.sample.size){
  #digits.small[d,1] <- (digits.sample[d,1])
  elem <- as.numeric(test.sample[d,1])
  #somehow it increases with 1 magically
  elem <- (elem - 1)
  digits.test[d,1] <- elem
  index.test <-2
  for (i in seq(2, length(test.sample),56)){
    for(j in seq(i, i+26, 2)){
      digits.test[d,index.test] <- c(rowSums(test.sample[d,c(j,j+1,j+28,j+29)]))/4
      index.test <- index.test + 1
    }
  }
}



#1.2 not lasso
ink_multinom <- multinom(label ~ ., data=digits.small, maxit = 1000, MaxNWts =10000000) # multinom Model
#train and test on same data
digit.pred.ink <- predict(ink_multinom, digits.small, type="class")
t1 <- table(digits.small[,1] , digit.pred.ink)
t1
accuracy1 = sum(diag(t1))/sum(t1)
accuracy1


#1.3 lasso get lambda value
optdigits.lasso.cv <-
  cv.glmnet(as.matrix(digits.small),
            digits.small[,1],family="multinomial", type.measure="class")

# plot lambda agains misclassification error
plot(optdigits.lasso.cv)

#predict class label on test set using the best cv model
optdigits.lasso.cv.pred <- predict(optdigits.lasso.cv, as.matrix(digits.small), type="class")

#confusion matrix 
t2 <- table(digits.small[,1],optdigits.lasso.cv.pred)
t2
accuracy2 = sum(diag(t2))/sum(t2)
accuracy2


#1.2 not lasso
#predict test set
digit.pred.ink <- predict(ink_multinom, digits.test[,-1], type="class")
t4 <- table(digits.test[,1] , digit.pred.ink)
t4
accuracy4 = sum(diag(t4))/sum(t4)
accuracy4


####1.3 with lasso
#predict class label on test set using the best cv model
optdigits.lasso.cv.pred <- predict(optdigits.lasso.cv, as.matrix(digits.test), type="class")

#confusion matrix 
t3 <- table(digits.test[,1],optdigits.lasso.cv.pred)
t3
accuracy3 = sum(diag(t3))/sum(t3)
accuracy3

#1.4 KNN
library(class)
library(caret)

#################################







##Implementation 5 cross validation

size <- sample.size/5

k <- c(1:5)
accuracies_avg <- 0
best.acc <- 0
best.k <- 0

for(i in k){
  print(i)
  
  
  folds <- cut(seq(1,nrow(digits.small)),breaks=10,labels=FALSE)
  accuracies <- 0
  #Perform 10 fold cross validation
  for(j in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds == j,arr.ind=TRUE)
    testData <- digits.small[testIndexes, -1]
    trainData <- digits.small[-testIndexes, -1 ]
    
    #Use the test and train data partitions however you desire...
    pred <- knn(train = trainData, test = testData, cl = digits.small[-testIndexes ,1], k = i)
    
    cf <- table(pred, digits.small[testIndexes,1])
    acc <- sum(diag(cf))/sum(cf)
    accuracies[j] <- acc
  }
  
  
  acc.avg <- mean(accuracies)
  accuracies_avg[i] <- acc.avg 
  print(accuracies)
  print(acc.avg)
  
  if(acc.avg > best.acc){
    best.acc <- acc.avg
    best.k <- i
  }
}

best.k
plot(accuracies, xlab = 'k', ylab = 'accuracy')


knn.pred <- knn(train = digits.small[,-1], test =  digits.test[,-1], cl = digits.small[,1], k = best.k)
confmat.knn <- table(knn.pred, digits.test[,"label"])[1:9,1:9]

acc <- sum(diag(confmat.knn))/sum(confmat.knn)


#####SVN

library("e1071")


model <- svm(digits.small[,-1], factor(digits.small[,1]), kernel="radial", cost=0.1, gamma = 0.5, scale = F)
pred <- predict(model, digits.small[,-1])
table(pred,digits.small[,1])

model <- svm(digits.small[,-1], factor(digits.small[,1]), kernel="radial", cost=1,gamma = 0.5, scale = F)
pred <- predict(model, digits.small[,-1])
table(pred,digits.small[,1])

pred1 <- predict(model, digits.test[,-1])
table(pred1,digits.test[,1])


svm_tune <- tune(svm, train.x=digits.small[,-1], train.y=factor(digits.small[,1]), 
                 kernel="radial", ranges=list(cost=c(1), gamma=c(.5,1,2,0.01,0.001)))
print(svm_tune)