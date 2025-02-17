library(tidyverse)
library(randomForest)
library(caret)


data <- read.csv("iris.csv")
View(data)

glimpse(data)

# Checking for missing values
sum(is.na(data))

count(data, Species)
# replacing the Species column with numeric values
# 1 = setosa, 2 = versicolor, 3 = virginica
data$Species <-  as.numeric(data$Species)

View(data)

data$Species <- as.factor(data$Species)

# Splitting the data into training and testing data
set.seed(42)
trainIndex <- createDataPartition(data$Species, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train <- data[ trainIndex,]
test  <- data[-trainIndex,]

# Training the classifier
model <- randomForest(Species ~ ., data = train, ntree = 100, importance = TRUE)

# Predicting the Species
predictions <- predict(model, test)
predictions

# Confusion matrix
confusionMatrix(predictions, test$Species)

# Feature importance
importance(model)


# saving the model
saveRDS(model, "model.rds")


#testing the model on a new data


new_data <- data.frame(Sepal.Length = 5.1, Sepal.Width = 3.5, Petal.Length = 1.4, Petal.Width = 0.2)
new_data

predict(model, new_data)



