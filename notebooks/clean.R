rm(list=ls())
gc()

library(pryr)
mem_used()

install.packages(c("tidyverse", "caret", "skimr", "corrplot", "VIM"))
install.packages("RANN")
library(tidyverse)
library(caret)
library(skimr)
library(corrplot)
library(VIM)
library(mltools)
library(data.table)
library(RANN)

dim(diabetic_data)
data = diabetic_data
names(data)

colnames(data) <- data[1, ]   # set row 1 as column names
data <- data[-1, ]            # drop row 1
rownames(data) <- NULL        # reset index

readmit_map <- c("NO" = 0, ">30" = 1, "<30" = 2)
data$readmitted <- readmit_map[data$readmitted]

table(data$readmitted)

data1 <- data %>% select(-encounter_id, -readmitted, -patient_nbr) #not require columns and target


#---------------------DUMMY ENCODING---------------------#
data1 <- data1 %>% mutate(across(where(is.character), as.factor))

#Remove constant columns (only 1 unique value)
data1 <- data1 %>% select(where(~ n_distinct(.) > 1))

dummy <- dummyVars("~ .", data = data1, fullRank = TRUE)
data_dummy <- data.frame(predict(dummy, newdata = data1))

#---------------------MISSING VALUES IMPUTE USING KNN---------------------#
# Check missing values

data_dummy[data_dummy == "?"] <- NA
sum(is.na(data_dummy))

dim(data_dummy)
#There is no NA or missing values 
#--------------------- NZV ---------------------#

nzv <- nearZeroVar(data_dummy, saveMetrics = TRUE)
data_nzv <- data_dummy[, !nzv$nzv]

dim(data_nzv)

#---------------------HIGHLY CORRELATED---------------------#

numeric_cols <- colnames(data_nzv)

# Compute correlation matrix
corr_matrix <- cor(data_nzv[, numeric_cols], use = "pairwise.complete.obs")

# Find highly correlated columns with threshold 0.95
high_corr <- findCorrelation(corr_matrix, cutoff = 0.95, verbose = TRUE)

# Remove highly correlated columns
if (length(high_corr) > 0) {
  data_final <- data_nzv[, -high_corr]
} else {
  data_final <- data_nzv  # nothing to remove
}

cat("Number of columns removed due to high correlation:", length(high_corr), "\n")
cat("Remaining columns:", ncol(data_final), "\n")

#Final data(data_final) is cleaned by following :dummy , missing, nzv, high corr. 

#---------------------SPATIAL SIGN---------------------#

# Redefine numeric_cols after filtering
numeric_cols <- colnames(data_final)

# Center the data
numeric_data <- scale(data_final[, numeric_cols], center = TRUE, scale = FALSE)

#Vectorized spatial sign (divide each row by its norm) as row by row will be slow
norms <- sqrt(rowSums(numeric_data^2))
norms[norms == 0] <- 1  # avoid division by zero for zero rows

#Apply transformation
data_spatial <- data_final
data_spatial[, numeric_cols] <- numeric_data / norms

cat("Spatial Sign transformation applied.\n")
cat("Dimensions of transformed data:", dim(data_spatial), "\n")


#---------------------BOX-COX---------------------#
# Redefine numeric_cols after spatial sign
numeric_cols <- colnames(data_spatial)

# Shift columns to make all values positive (Box-Cox requires > 0)
min_vals <- sapply(data_spatial[, numeric_cols], min, na.rm = TRUE)
shift <- ifelse(min_vals <= 0, abs(min_vals) + 1e-6, 0)

data_boxcox <- data_spatial
for (col in numeric_cols) {
  data_boxcox[[col]] <- data_boxcox[[col]] + shift[col]
}

# Apply Box-Cox
pre_boxcox <- preProcess(data_boxcox[, numeric_cols], method = "BoxCox")
data_boxcox[, numeric_cols] <- predict(pre_boxcox, data_boxcox[, numeric_cols])
cat("Box-Cox transformation applied on spatially transformed data.\n")
cat("Dimensions after Box-Cox:", dim(data_boxcox), "\n")

#---------------------ADD TARGET BACK---------------------#
data_boxcox$readmitted <- data$readmitted
cat("Target variable 'readmitted' added back.\n")
cat("Class distribution of readmitted:\n")
print(table(data_boxcox$readmitted))

#---------------------PLOT TARGET DISTRIBUTION---------------------#
library(ggplot2)

ggplot(data = data_boxcox, aes(x = factor(readmitted))) +
  geom_bar(fill = "forestgreen", color = "black") +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5, size = 4) +
  labs(
    title = "Distribution of Readmitted (Target Variable)",
    x = "Readmitted (0 = NO, 1 = >30 days, 2 = <30 days)",
    y = "Count of Observations"
  ) +
  theme_minimal() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))

#---------------------SAVE TO CSV---------------------#
write.csv(data_boxcox, "diabetes_preprocessed.csv", row.names = FALSE)
cat("Saved to diabetes_preprocessed.csv\n")


#---------------------TRAIN/TEST SPLIT---------------------#
status <- as.factor(data_boxcox$readmitted)

# Use only feature columns (exclude readmitted) for X
data_final <- data_boxcox[, !colnames(data_boxcox) %in% "readmitted"]

set.seed(100)
train_index <- createDataPartition(status, p = 0.75, list = FALSE)

X_train <- data_final[train_index, ]
X_test  <- data_final[-train_index, ]
y_train <- status[train_index]
y_test  <- status[-train_index]

train_data <- cbind(X_train, Status = y_train)
test_data  <- cbind(X_test,  Status = y_test)

train_data$Status <- as.factor(train_data$Status)
test_data$Status  <- as.factor(test_data$Status)

cat("Train dimensions:", dim(train_data), "\n")
cat("Test dimensions:",  dim(test_data),  "\n")
cat("Class distribution in train:\n")
print(table(train_data$Status))
cat("Class distribution in test:\n")
print(table(test_data$Status))


# Recode factor levels to valid R variable names
levels_map <- c("0" = "NO", "1" = "GT30", "2" = "LT30")

train_data$Status <- factor(levels_map[as.character(train_data$Status)],
                            levels = c("NO", "GT30", "LT30"))

test_data$Status  <- factor(levels_map[as.character(test_data$Status)],
                            levels = c("NO", "GT30", "LT30"))

# Verify
print(table(train_data$Status))
print(table(test_data$Status))

library(doParallel)
cl <- makePSOCKcluster(parallel::detectCores() - 1)  # leave 1 core free
registerDoParallel(cl)
#------------------- LOGISTIC REGRESSION -------------------#

set.seed(100)
model_multinom <- train(
  Status ~ .,               
  data = train_data,        
  method = "multinom",
  metric = "Kappa",
  trControl = trainControl(method = "cv",
                           number = 3,
                           classProbs = TRUE,
                           savePredictions = "final",
                           summaryFunction = defaultSummary),
  preProcess = c("center", "scale", "pca"),
  trace = FALSE
)

print(model_multinom$results)
model_multinom
plot(model_multinom)


pred_multinom <- predict(model_multinom, newdata = test_data)
confusionMatrix(pred_multinom, test_data$Status)



#---------------------------Elastic Net-------------------------------------------------#

enet_grid <- expand.grid(alpha = seq(0, 1, length = 5),  
                         lambda = 10^seq(-3, 1, length = 10))

set.seed(100)
model_enet <- train(Status ~ .,               
                    data = train_data,
                    method = "glmnet",
                    preProcess = c("center", "scale", "pca"),
                    tuneGrid = enet_grid,
                    metric = "Kappa",
                    trControl = trainControl(method = "cv",
                                             number = 3,
                                             classProbs = TRUE,
                                             savePredictions = "final",
                                             summaryFunction = defaultSummary))

print(model_enet$results)
plot(model_enet)

pred_enet <- predict(model_enet, newdata = test_data)

cm_enet <- confusionMatrix(pred_enet, test_data$Status)
print(cm_enet)


#-----------------------NEURAL-NET?-----------------------------------------------------#
nnetGrid <- expand.grid(
  .size = 1:10,        # number of hidden units
  .decay = c(0, 0.1, 1, 2)  # weight decay
)


p <- ncol(train_data) - 1       # number of predictors
maxSize <- max(nnetGrid$.size)
numWts <- (p + 1) * maxSize + (maxSize + 1) * length(levels(train_data$Status))
numWts


set.seed(123)
nn_fit <- train(
  Status ~ .,  
  data = train_data,
  method = "nnet",
  preProcess = c("center", "scale", "pca"),
  trControl = trainControl(
    method = "cv",
    number = 3,
    classProbs = TRUE,
    savePredictions = "final"
  ),
  tuneGrid = nnetGrid,
  trace = FALSE,
  MaxNWts = numWts,
  maxit = 500,
  metric = "Kappa"
)

print(nn_fit)
plot(nn_fit)


nn_pred <- predict(nn_fit, newdata = test_data)

cm_nn <- confusionMatrix(nn_pred, test_data$Status)
print(cm_nn)


#---------------------RANDOM FOREST---------------------#
rfGrid <- expand.grid(
  .mtry = c(2, 4, 6, 8, 10, 12)   # number of predictors sampled at each split
)

set.seed(123)
rf_fit <- train(
  Status ~ .,
  data      = train_data,
  method    = "rf",
  trControl = trainControl(
    method          = "cv",
    number          = 3,
    classProbs      = TRUE,
    savePredictions = "final",
    summaryFunction = defaultSummary
  ),
  tuneGrid  = rfGrid,
  ntree     = 500,         # number of trees
  importance = TRUE,       # enables variable importance
  metric    = "Kappa"
)

print(rf_fit)
plot(rf_fit)

# Predictions
rf_pred <- predict(rf_fit, newdata = test_data)
cm_rf   <- confusionMatrix(rf_pred, test_data$Status)
print(cm_rf)

stopCluster(cl)


# Variable Importance
rf_imp <- varImp(rf_fit, scale = TRUE)
print(rf_imp)
plot(rf_imp, top = 20, main = "Random Forest - Top 20 Variable Importance")
