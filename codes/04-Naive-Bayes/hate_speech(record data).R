# Balancing Data
library('tidyverse')
library('ggplot2')
library(naivebayes)
library(maditr)
library(caret)
library(ROCR)
library(pROC)
# read the data
df <- read.csv('./hate_speech.csv')
head(df)
sapply(df, function(x) table(is.null(x)))
df$label = as.factor(df$label)
# Exploratory data analysis
str(df)
dim(df)
# drop the unused columns
df = subset(df,select = -c(X,location))
df$id = 1:nrow(df)

# Splitting test and train Data Partition
train <- df %>% dplyr::sample_frac(0.75)
test  <- dplyr::anti_join(df, train, by = 'id')
dim(train)
dim(test)
# Feature Scaling
x_train = subset(train,select = -c(label,text,id))
y_train = train$label
x_test = subset(test,select = -c(label,text,id))
y_test = test$label

# Model Building
model <- naive_bayes(x=x_train,y=y_train) 

# Prediction
Predict <- predict(model,test) #Get the confusion matrix to see accuracy value and other parameter values > confusionMatrix(Predict, testing$Outcome )
tab1 = table(Predict,y_test)

sum(diag(tab1)) / sum(tab1)

# Variable Importance Plot
# X <- varImp(model)
# plot(X)

# Confusion Matrix 
cm = confusionMatrix(as.factor(test$label), Predict)
print(cm)
confusion = as.data.frame(tab1)
names(confusion) = c("Predicted","Actual","Freq")
confusion$Percent = confusion$Freq/313*100

tile <- ggplot() +
  geom_tile(aes(x=Actual, y=Predicted,fill=Percent),data=confusion, color="black",size=0.1) +
  labs(x="Actual",y="Predicted")
tile = tile + 
  geom_text(aes(x=Actual,y=Predicted, label=sprintf("%.1f", Percent)),data=confusion, size=3, colour="black") +
  scale_fill_gradient(low="Grey",high="red")+ggtitle('Confusion Matrix for the record data')+ theme(plot.title = element_text(hjust = 0.5))

# lastly we draw diagonal tiles. We use alpha = 0 so as not to hide previous layers but use size=0.3 to highlight border
tile = tile + 
  geom_tile(aes(x=Actual,y=Predicted),data=subset(confusion, as.character(Actual)==as.character(Predicted)), color="black",size=0.3, fill="black", alpha=0) 

#render
tile

Predict = as.numeric(Predict)
y_test = as.numeric(y_test)

pred = prediction(Predict, y_test)
perf <- performance(pred,"tpr","fpr")
pROC_obj <- roc(Predict, y_test,
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)
plot(perf,colorize=TRUE,main=paste('ROC curve(GaussianNB Naive Bayes) The AUC score is',0.4562))

heatmap(tab1, Rowv = NA, Colv = NA)
