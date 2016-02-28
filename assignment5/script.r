# library("caret")
library("rpart")
library("tree")
library("randomForest")
library("e1071")
# library("ggplot2")

t = read.csv("seaflow_21min.csv")

summary(t$pop);

summary(t)
summary(t$fsc_small, digits=12)

# splitdf function will return a list of training and testing sets
splitdf <- function(dataframe, seed=NULL) {
	if (!is.null(seed)) set.seed(seed)
	index <- 1:nrow(dataframe)
	trainindex <- sample(index, trunc(length(index)/2))
	trainset <- dataframe[trainindex, ]
	testset <- dataframe[-trainindex, ]
	list(trainset=trainset,testset=testset)
}

splits = splitdf(t, seed=1234)
training = splits$trainset
test = splits$testset

# Question 3: mean time of training set
mean(training$time)

# Decision Tree model --  the "class" avoids having to round and remap data.
fol <- formula(pop ~ fsc_small + fsc_perp + fsc_big + pe + chl_big + chl_small)
dt_model <- rpart(fol, method="class", data=training)

# Question 5: what populations, if any, is the tree *incapable* of measuring?
# (Hint: look for the one that's not listed.)
# Question 6: Verify there's a threshold on PE learned in the model.
# Question 7: Based on the tree, which variables appear most important
print(dt_model)

# dt_predict <- predict(dt_model, newdata=test, type="class")
# dt_result = dt_predict == test$pop
# summary(dt_result)


# rf_model <- randomForest(fol, data=training)
# rf_predict <- predict(rf_model, newdata=test)
# rf_result <- rf_predict == test$pop
# summary(rf_result)

# importance(rf_model)

svm_model = svm(fol, data=training)
svm_predict = predict(svm_model, newdata=test)
svm_result = svm_predict == test$pop
summary(svm_result)

# # Question 12: Confusion matrices.
# # What appears to be the most common error?  I found the DT one more helpful.
# table(pred = dt_predict, true = test$pop) # Decision tree
# table(pred = rf_predict, true = test$pop) # Random Forest
# table(pred = svm_predict, true = test$pop) # Support Vector Machine

t2 = subset(t, t$file_id != 208)

# Resample.
splits = splitdf(t2, seed=1234)
new_training = splits$trainset
new_test = splits$testset

new_svm_model = svm(newfol, data=new_training)
new_svm_predict = predict(new_svm_model, newdata=new_test)
new_svm_result = new_svm_predict == new_test$pop
summary(new_svm_result)