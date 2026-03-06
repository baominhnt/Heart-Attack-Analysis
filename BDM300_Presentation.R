##############################################
# HEART DISEASE PROJECT - FULL R SCRIPT
# Methods: Decision Tree, KNN, Random Forest,
#          Apriori, FP-Growth
##############################################

# Install packages if needed

library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(class)
library(arules)
library(arulesViz)
library(arulesCBA)
library(naivebayes)
library(e1071)

##############################################
# 1. LOAD DATA
##############################################

heart <- read.csv("heart.csv")

# Convert target to factor (classification label)
heart$target <- as.factor(heart$target)
set.seed(123)

##############################################
# 1B. PREPROCESSING
##############################################

# Check missing values
colSums(is.na(heart))

# Convert categorical variables to factors
factor_cols <- c("sex", "cp", "fbs", "restecg", "exang",
                 "slope", "ca", "thal", "target")

heart[factor_cols] <- lapply(heart[factor_cols], factor)

# Optional: Summary
str(heart)

# Optional: remove duplicates
heart <- unique(heart)

# Optional: Handle NA rows
heart <- na.omit(heart)


##############################################
# 2. TRAIN/TEST SPLIT
##############################################

train_index <- createDataPartition(heart$target, p=0.8, list=FALSE)
train <- heart[train_index, ]
test  <- heart[-train_index, ]

##############################################
# 3. DECISION TREE
##############################################

dt_model <- rpart(target ~ ., data = train, method = "class")
rpart.plot(dt_model)

dt_pred <- predict(dt_model, test, type="class")

cat("\n===== Decision Tree Accuracy =====\n")
print(confusionMatrix(dt_pred, test$target))

##############################################
# NAIVE BAYES
##############################################
train_index <- createDataPartition(heart$target, p=0.8, list=FALSE)
train <- heart[train_index, ]
test  <- heart[-train_index, ]
nb_model <- naive_bayes(target ~ ., data=train)
nb_pred  <- predict(nb_model, test)

cat("\n===== Naive Bayes Accuracy =====\n")
print(confusionMatrix(nb_pred, test$target))

train$cp <- as.factor(train$cp)
nb_cp <- naive_bayes(target ~ cp, data=train)
print(nb_cp$tables)
chestpain_levels <- data.frame(cp = factor(c(0,1,2,3)))
predict(nb_cp, chestpain_levels, type="prob")

cp_levels <- data.frame(cp = factor(c(0,1,2,3)))
posterior <- predict(nb_cp, cp_levels, type="prob")

# Convert to a plotting-friendly data frame
library(ggplot2)
posterior_df <- data.frame(
  cp = factor(c(0,1,2,3),
              levels=c(0,1,2,3),
              labels=c("0 = Typical", "1 = Atypical", 
                       "2 = Non-anginal", "3 = Asymptomatic")),
  Prob_NoDisease = posterior[,1],
  Prob_Disease   = posterior[,2]
)
library(reshape2)
posterior_long <- melt(posterior_df, id.vars="cp",
                       variable.name="Outcome",
                       value.name="Probability")

# Plot
ggplot(posterior_long, aes(x=cp, y=Probability, fill=Outcome)) +
  geom_bar(stat="identity", position="dodge") +
  labs(title="Posterior Probability of Heart Disease by Chest Pain Type",
       x="Chest Pain Level (cp)",
       y="Posterior Probability") +
  theme_minimal()


##############################################
# 4. K-NEAREST NEIGHBORS (KNN)
##############################################

# KNN requires numeric + scaled data
num_cols <- sapply(train, is.numeric)

# ---- 2. Combine train + test BEFORE scaling (IMPORTANT) ----
combined <- rbind(train[, num_cols], test[, num_cols])
combined_scaled <- scale(combined)

# Split back into scaled train and test
train_scaled <- combined_scaled[1:nrow(train), ]
test_scaled  <- combined_scaled[(nrow(train)+1):nrow(combined), ]


train_labels <- train$target
test_labels  <- test$target

knn_pred <- knn(train_scaled, test_scaled, train_labels, k = 5)

knn_cm <- table(Predicted = knn_pred, Actual = test_labels)
cm_df <- as.data.frame(knn_cm)
colnames(cm_df) <- c("Predicted", "Actual", "Freq")

ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "KNN Confusion Matrix (k = 5)",
       x = "Actual Label",
       y = "Predicted Label") +
  theme_minimal()

ggplot(heart, aes(x = trestbps, y = chol, color = factor(target))) +
  geom_point(alpha = 0.7, size = 3) +
  labs(
    title = "KNN Feature Space Visualization (trestbps vs chol)",
    x = "Resting Blood Pressure (trestbps)",
    y = "Cholesterol (chol)",
    color = "Target"
  ) +
  theme_minimal()


cat("\n===== KNN Accuracy (k=5) =====\n")
print(confusionMatrix(knn_pred, test_labels))

##############################################
# 5. RANDOM FOREST
##############################################

rf_model <- randomForest(target ~ ., data=train, ntree=500, mtry=4, importance=TRUE)
rf_pred  <- predict(rf_model, test)

cat("\n===== Random Forest Accuracy =====\n")
print(confusionMatrix(rf_pred, test$target))

# Variable importance plot
importance_df <- data.frame(
  Feature = rownames(importance(rf_model)),
  MeanDecreaseGini = importance(rf_model)[, "MeanDecreaseGini"]
)

ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini), 
                          y = MeanDecreaseGini)) +
  geom_col(fill="steelblue") +
  coord_flip() +
  labs(title="Random Forest Variable Importance",
       x="Feature",
       y="Mean Decrease Gini") +
  theme_minimal()
##############################################
# 6. APRIORI ASSOCIATION RULES
##############################################

# Discretize numeric features (3 bins: Low, Medium, High)
heart_disc <- as.data.frame(lapply(heart, function(x){
  if(is.numeric(x)) cut(x, breaks=3, labels=c("Low","Medium","High"))
  else x
}))

# Convert to transactions
heart_trans <- as(heart_disc, "transactions")

# Run Apriori
rules_ap <- apriori(heart_trans, 
                    parameter=list(supp=0.1, conf=0.6))

cat("\n===== Apriori Rules (Top 10) =====\n")
inspect(head(rules_ap, 10))

# Optional rule visualization
plot(rules_ap, method="graph", engine="htmlwidget")

acc_dt  <- confusionMatrix(dt_pred, test$target)$overall["Accuracy"]
acc_nb  <- confusionMatrix(nb_pred, test$target)$overall["Accuracy"]
acc_knn <- confusionMatrix(knn_pred, test_labels)$overall["Accuracy"]
acc_rf  <- confusionMatrix(rf_pred, test$target)$overall["Accuracy"]
accuracy_df <- data.frame(
  Model = c("Decision Tree", "Naive Bayes", "KNN", "Random Forest"),
  Accuracy = c(acc_dt, acc_nb, acc_knn, acc_rf)
)
print(accuracy_df)

ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat="identity") +
  ylim(0, 1) +
  labs(
    title = "Accuracy Comparison of ML Models",
    y = "Accuracy",
    x = "Model"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

