library(mice)
library(ggplot2)
library(moments)
library(e1071)
library(Metrics)
library(lattice)
library(glmnet)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library("DiagrammeR")

df <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)

# Understand the data
summary(df)
dim(df)
head(df)


# check how many NAs in each feature in combined df
colSums(sapply(df, is.na))
sort(sapply(df, function(x) { sum(is.na(x)) }), decreasing=TRUE)
# The percentage of data missing in combined df
sum(is.na(df)) / (nrow(df) *ncol(df))

#bar plot pool quality
barplot(table(df$PoolQC))
# 2909 missing values in pool quality
barplot(table(df$PoolQC))

#PoolQC: NA mean no pool => None
df$PoolQC[which(is.na(df$PoolQC))] <- "None"
# Alley : NA mean no alley access -> None
df$Alley[which(is.na(df$Alley))] <- "None"
#Fence: no fence -> None
df$Fence[which(is.na(df$Fence))] <- "None"
#FireplaceQu: na mean no fireplace -> None
df$FireplaceQu[which(is.na(df$FireplaceQu))] <- "None"
#MiscFeature : None
df$MiscFeature[which(is.na(df$MiscFeature))] <- "None"

#GarageYrBlt None->0
df$GarageYrBlt[which(is.na(df$GarageYrBlt))] <- 0 
# GarageType : NA to None
df$GarageType[which(is.na(df$GarageType))] <- "None"
#GarageFinish: NA to None
df$GarageFinish[which(is.na(df$GarageFinish))] <- "None"
# GarageQual: NA to None
df$GarageQual[which(is.na(df$GarageQual))] <- "None"
# GarageCond NA to None
df$GarageCond[which(is.na(df$GarageCond))] <- "None"

# BsmtCond : NA to None
df$BsmtCond[which(is.na(df$BsmtCond))] <- "None"
# BsmtExposure : NA to None
df$BsmtExposure[which(is.na(df$BsmtExposure))] <- "None"
#BsmtQual ： NA to None
df$BsmtQual[which(is.na(df$BsmtQual))] <- "None"
# BsmtFinType2 : NA to None
df$BsmtFinType2[which(is.na(df$BsmtFinType2))] <-"None"
# BsmtFinType1 : NA to None
df$BsmtFinType1[which(is.na(df$BsmtFinType1))] <-"None"

#calculate the skewness for numerical variables
classes <- lapply(df,function(x) class(x))
numeric = names(which(sapply(df, is.numeric)))
numeric <- names(classes[classes=="integer" | classes=="numeric"])
skewed <- sapply(numeric, function(x) skewness(df[[x]]))
skewed <- skewed[abs(skewed) > .75]
names(skewed)
skewed = skewed[!is.na(skewed)]
# log transform skewed features
for (x in names(skewed)) {df[[x]] <- log(df[[x]]+1)}

#feature engineering
# Total SF for house (incl. basement)
df$AllSF <- with(df, GrLivArea + TotalBsmtSF)
# Total number of bathrooms
df$TotalBath <- with(df, BsmtFullBath + 0.5 * BsmtHalfBath + FullBath + 0.5 * HalfBath)
# how new the house
df$old <- 'VeryOld'
df$old[df$YearBuilt > 2008] <- 'New'
df$old[df$YearBuilt > 2004 & df$YearBuilt <= 2008] <- 'VeryNew'
df$old[df$YearBuilt > 1990 & df$YearBuilt <= 2004] <- 'NotNew'
df$old[df$YearBuilt > 1965 & df$YearBuilt <= 1990]  <- 'Old'
df$old <- as.factor(df$old)

df$MasVnrType <- as.factor(df$MasVnrType)
df$Electrical <- as.factor(df$Electrical)
#df$MSZoning <- as.factor(df$MSZoning)
#df$Utilities <- as.factor(df$Utilities)
#df$Functional <- as.factor(df$Functional)
#df$Exterior1st <- as.factor(df$Exterior1st)
#df$Exterior2nd <- as.factor(df$Exterior2nd)
#df$KitchenQual <- as.factor(df$KitchenQual)
#df$SaleType <- as.factor(df$SaleType)
#df$AllSF <- as.factor(df$AllSF)

df$MSZoning[which(is.na(df$MSZoning))] <- "None"
df$Utilities[which(is.na(df$Utilities))] <- "None"
df$Functional[which(is.na(df$Functional))] <- "None"
df$Exterior1st[which(is.na(df$Exterior1st))] <- "None"
df$Exterior2nd[which(is.na(df$Exterior2nd))] <- "None"
df$Utilities[which(is.na(df$Utilities))] <- "None"
df$KitchenQual[which(is.na(df$KitchenQual))] <- "None"
df$SaleType[which(is.na(df$SaleType))] <- "None"
df$AllSF[which(is.na(df$AllSF))] <- "None"

imp.df <- mice(df, m=10, method='cart', printFlag=FALSE)
sort(sapply(complete(imp.df), function(x) { sum(is.na(x)) }), decreasing=TRUE)

df_complete <- complete(imp.df)

xyplot(imp.df, LotFrontage ~ LotArea)
densityplot(imp.df, ~LotFrontage)

sum(sapply(df_complete, function(x) { sum(is.na(x)) }))
sort(sapply(df_complete, function(x) {length(which(is.na(x)))}))
# linear regression model
df_complete$LogPrice <- log(df_complete$SalePrice)
mod.all <- lm(LogPrice ~., data = df_complete)
mod.all
summary(mod.all)
summary(lm(LogPrice ~ PoolQC, data = df_complete))

plot(mod.all$residuals)
plot(mod.all$fitted.values, mod.all$residuals)
plot(mod.all)

#fit with sig feature
mod_sig <- lm(LogPrice~MSZoning+OverallQual+Neighborhood+Condition1+Condition2+HouseStyle
              +RoofStyle+RoofMatl+Exterior1st+Exterior2nd+ExterQual+BsmtQual+BsmtCond+BsmtExposure
              +TotalBsmtSF+Heating+Electrical+X1stFlrSF+X2ndFlrSF+LowQualFinSF
              +GrLivArea+KitchenQual+Functional ,data = df_complete)
summary(mod_sig)
plot(mod_sig$fit,mod_sig$residuals)
abline(h=0,col="red")
plot(mod_sig)

set.seed(111)
train.ind <- sample(1:dim(df_complete)[1], dim(df_complete)[1] * 0.75)
train.data <- df_complete[train.ind, ]
test.data <- df_complete[-train.ind, ]

prediction <- predict(mod_sig,test.data,type = "response")
length(prediction)

log_SalePrice <- log(test.data$SalePrice)

#RMSE
rmse(log_SalePrice,prediction)
#sum of square error
sum((prediction - log_SalePrice)^2)


# lasso
train <- read.csv("train.csv", stringsAsFactors = FALSE)
sort(sapply(train, function(x) {length(which(is.na(x)))}))
miss.col <- which(sapply(train, function(x) {length(which(is.na(x)))}) > 0)
ind <- model.matrix( ~., train[, -c(1, 81, miss.col)])
dep <- log(train$SalePrice)
fit <- glmnet(x=ind, y=dep) # default is alpha = 1, lasso
plot(fit)
# Understand the plot
# The y axis indicates the number of nonzero coefficients at the current λ, 
# which is the effective degrees of freedom (df) for the lasso.
plot(fit, label = T)
plot(fit, xvar = "lambda", label = T)

print(fit)
# Df is the non zero beta, 
# Deviance_model = 2*(loglikelihood_saturate_model - loglikelihood_current_model)
# Deviance_null = 2*(loglikelihood_saturate_model - loglikelihood_intercept_only_model)
# Deviance percentage = 1 -  Deviance_model / Deviance_null
# lambda value

# We can choose lambda by checking the picture, Still kinda subjective
# use cross validation to get optimal value of lambda, 
cvfit <- cv.glmnet(ind, dep)
plot(cvfit)
# Two selected lambdas are shown, 
cvfit$lambda.min # value of lambda gives minimal mean cross validated error
cvfit$lambda.1se # most regularized model such that error is within one std err of the minimum
x = coef(cvfit, s = "lambda.min")
coef(cvfit, s = "lambda.1se")

#decision tree
formula <- (log(train.data$SalePrice)~MSZoning+OverallQual+Neighborhood+Condition1+Condition2+HouseStyle
            +RoofStyle+RoofMatl+Exterior1st+Exterior2nd+ExterQual+BsmtQual+BsmtCond+BsmtExposure
            +TotalBsmtSF+Heating+X1stFlrSF+X2ndFlrSF+LowQualFinSF
            +GrLivArea+KitchenQual+Functional)

set.seed(111)
tree1 <- rpart(formula, method = 'anova', data = train.data, 
               control=rpart.control(cp=0)) # cp = 1, 0.1, 0.01, 0
# deviance is determined as Sum(observed_i - mu_i)^2, sum of squared error of data in that node
mean(log(train.data$SalePrice)) # yi is actually mu_i
printcp(tree1)
plotcp(tree1) 
plot(tree1)
text(tree1,cex = 0.5, xpd = TRUE)
# step 2 Pick the tree size that minimizes xerror.
bestcp <- tree1$cptable[which.min(tree1$cptable[,"xerror"]),"CP"]

# Step 3: Prune the tree using the best cp.
tree.pruned <- prune(tree1, cp = bestcp)
tree.pruned

test.pred <- predict(tree.pruned, test.data)

for(i in 1:dim(train.data)[2]) {
  if(is.character(train.data[, i]) & 
     length(which(!unique(test.data[, i]) %in% unique(train.data[, i]))) > 0) {
    print(paste("this column: ", colnames(train.data)[i], "has new levels in test"))
  } 
}

test.data$Condition2[which(!test.data$Condition2 %in% train.data$Condition2)] <- NA
test.data$RoofMatl[which(!test.data$RoofMatl %in% train.data$RoofMatl)] <- NA
test.data$Heating[which(!test.data$Heating %in% train.data$Heating)] <- NA
test.data$Functional[which(!test.data$Functional %in% train.data$Functional)] <- NA
test.data$MiscFeature[which(!test.data$MiscFeature %in% train.data$MiscFeature)] <- NA
test.data$Electrical[which(!test.data$Electrical %in% train.data$Electrical)] <- NA

test.pred <- predict(tree.pruned, test.data)
prp(tree.pruned, faclen = 0, cex = 0.2)
#rmse for decision tree
rmse(log(test.data$SalePrice),test.pred)
#sum of square error for decision tree
sum((test.pred - log(test.data$SalePrice))^2)

# Random Forest
set.seed(111)

rf.formula <- (log(train.data$SalePrice)~MSZoning+OverallQual+Neighborhood+Condition1+Condition2+HouseStyle
            +RoofStyle+RoofMatl+Exterior1st+Exterior2nd+ExterQual+BsmtQual+BsmtCond+BsmtExposure
            +TotalBsmtSF+Heating+X1stFlrSF+X2ndFlrSF+LowQualFinSF
            +GrLivArea+KitchenQual+Functional)

rf <- randomForest(as.formula(rf.formula), data = train.data, importance = TRUE)
rf_2 <- randomForest(SalePrice ~ .-AllSF, data=train.data)

getTree(rf, k = 1, labelVar = TRUE)# output tree 1 for example with variable labeled
par(mar=rep(2,4))
# check the setting in par(), like 
par()$mfrow
par(mfrow = c(1,1))

varImpPlot(rf)
importance(rf, type = 1)
importanceOrder= order(-rf$importance[, "%IncMSE"])
names=rownames(rf$importance)[importanceOrder]
names
for (name in names[1:2]) {
  partialPlot(rf, train.data, eval(name), main=name, xlab=name)
}

plot(rf)
plot(rf_2)

for (f in names(train.data)) {
  if (class(train.data[[f]])=="character") {
    levels <- unique(c(train.data[[f]], test.data[[f]]))
    train.data[[f]] <- as.integer(factor(train.data[[f]], levels=levels))
    test.data[[f]]  <- as.integer(factor(test.data[[f]],  levels=levels))
  }
}

# Predict using the test set
prediction <- predict(rf_2, test.data)
#RMSE for random forest
rmse(log(test.data$SalePrice),log(prediction))
#sum of square error for random forest
sum(( log(prediction) - log(test.data$SalePrice))^2)

#xgboost
train.label <- log(train.data$SalePrice)
test.label <- log(test.data$SalePrice)
feature.matrix <- model.matrix( ~ ., data = train.data[, -c(81,85)])
dim(feature.matrix)
set.seed(111)

gbt <- xgboost(data =  feature.matrix, 
               label = train.label, 
               max_depth = 8, # for each tree, how deep it goes
               nround = 20, # number of trees
               objective = "reg:linear",
               nthread = 3,
               verbose = 2)
importance <- xgb.importance(feature_names = colnames(feature.matrix), model = gbt)
head(importance)
prediction = predict(gbt, model.matrix(~.,data = test.data[,-c(81,85)]))
prediction
sum((prediction-log(test.data$SalePrice))^2)
rmse(prediction, log(test.data$SalePrice))

xgb.plot.tree(model = gbt)
xgb.plot.tree(feature_names = colnames(feature.matrix), model = gbt, n_first_tree = 1)

plot(gbt.cv)
# what's the optimal parameter, for example, number of trees?
par <- list( max_depth = 8,
             objective = "reg:linear",
             nthread = 3,
             verbose = 2)
gbt.cv <- xgb.cv(params = par,
                 data = feature.matrix, label = train.label,
                 nfold = 5, nrounds = 100)

par(mfrow=c(1, 1))
plot(gbt.cv$train.rmse.mean, type = 'l')
lines(gbt.cv$test.rmse.mean, col = 'red')
nround = which(gbt.cv$test.rmse.mean == min(gbt.cv$test.rmse.mean)) # 36
gbt <- xgboost(data = feature.matrix, 
               label = train.label,
               nround = nround,
               params = par)

# grid searching for parameters.
all_param = NULL
all_test_rmse = NULL
all_train_rmse = NULL
best_param = list()
best_seednumber = 1234
best_rmse = Inf
best_rmse_index = 0
min_rmse = 0
for (iter in 1:20) {
  param <- list(objective = "reg:linear",
                max_depth = sample(5:12, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround = 200
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=feature.matrix, label = train.label, params = param, nthread=6, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = F, # early_stop_round=8, 
                 maximize=FALSE)
  min_train_rmse = min(mdcv$evaluation_log$train_rmse_mean)
  min_test_rmse = min(mdcv$evaluation_log$test_rmse_mean)
  
  all_param <- rbind(all_param, unlist(param)[-1])
  all_train_rmse <- c(all_train_rmse, min_train_rmse)
  all_test_rmse <- c(all_test_rmse, min_test_rmse)
    #if (min_rmse < best_rmse) {
     #  best_rmse = min_rmse
      # best_rmse_index = min_rmes_index
       #best_seednumber = seed.number
       #best_param = param
   #}
}
all_param <- as.data.frame(as.numeric(all_param))
best.gbt <- xgboost(data = feature.matrix, 
                    label = train.label,
                    nround = 500,
                    max_depth = 12,
                    eta = 0.158,         
                    gamma = 0.0795,
                    subsample = 0.77,
                    colsample_bytree = 0.794,
                    min_child_weight = 6,
                    max_delta_step = 2)

# prediction
prediction = predict(best.gbt, model.matrix(~.,data = test.data[,-c(81,85)]))
# sum of squared error in XGBoost
sum((prediction - log(test.data$SalePrice))^2)
#rmse for XGBoost
sqrt(sum((prediction - log(test.data$SalePrice))^2)/dim(test.data)[1])
#rmse for XGBoost
rmse(log(test.data$SalePrice),prediction)

