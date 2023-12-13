heartdata <- read.csv("C:\\Users\\olufu\\Downloads\\archive\\2022\\heart_2022_no_nans.csv")
num_observations <- dim(heartdata) #shows dimension of data, there are 246022 objects(rows) and 40 variables(column)
# https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/ 
# above is link to dataset.
library(ggplot2)     # Popular library in R for visualization
missing_values <- colSums(is.na(heartdata))    # checks for missing values
missing_values  # we can see that there is not a single missing value in the data set

# visualization of heart attack status vs age category using grouped bar chart
ggplot(heartdata, aes(x = AgeCategory, fill = HadHeartAttack)) +
  geom_bar(position = "dodge", color = "black", stat = "count") +
  labs(x = "Age Category", y = "Count", title = "Bar Chart of Had Heart Attack vs Age Category") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Visualization Distribution of ethnicity in the datset
ggplot(heartdata, aes(x = factor(RaceEthnicityCategory))) +
  geom_bar(fill = c("skyblue"), color = "black") +
  geom_text(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5)) +
  labs(title = "Distribution of Race", x = "Race", y = "Count") +
  scale_x_discrete(labels = c("0" = "Black only nonHispanic", "1" = "Hispanic",
                              "2" = "Multiracial nonHispanic", "3" = "Other race only, non Hispanic",
                              "4" ="white only non Hispanic")) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Create a box plot
ggplot(heartdata, aes(x = HadHeartAttack, y = SleepHours)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  labs(title = "Box Plot of heartattack by sleephours",
       x = "Category",
       y = "Value")

# you can have as many visualization as you might want to explore the dataset


# For this project we are going to be using tidymodels to run a logistic regression and also random forest 
# we are going to compare the results
library(tidymodels)
library(recipes)
library(modeldata)
library(yardstick)
library(dplyr)
library(caret)
set.seed(123)    #always set seed for the purpose of reproducibility


# Explicitly set the outcome variable as a factor
heartdata$HadHeartAttack <- as.factor(heartdata$HadHeartAttack)

#define the recipe
heartdata_recipe <- 
  recipe(HadHeartAttack ~ ., data = heartdata) %>%
  # Cast all numeric variables to numeric
  step_mutate(across(where(is.numeric), as.numeric)) %>%
  # Use step_factor2string to make all factor variables with ranges output their ranges
  step_factor2string(AgeCategory) %>%
  # Cast all character variables to factors
  step_mutate(across(where(is.character), as.factor)) %>%
  # Cast all integer variables to integer
  #step_mutate(across(where(is.integer), as.integer)) %>%
  # Remove single unique values
  step_zv(all_predictors()) %>%
  # Decorrelate all highly correlated values 
  step_corr(all_numeric_predictors(), threshold = 0.7)


# split the data into training (75%) and testing (25%)
heartdata_split <- initial_split(heartdata, 
                                  prop = 3/4)


# extract training and testing sets
heartdata_train <- training(heartdata_split)
heartdata_test <- testing(heartdata_split)
# create Cross validation object from training data
heartdata_cv <- vfold_cv(heartdata_train, v=5)


#applying the recipe to the train and test dataset
heartdata_train_preprocessed <- heartdata_recipe |>
  # apply the recipe to the training data
  prep(heartdata_train) |>
  # extract the pre-processed training dataset
  juice()
heartdata_test_preprocessed <- heartdata_recipe |>
  # apply the recipe to the test data
  prep(heartdata_test) |>
  # extract the pre-processed test dataset
  juice()
heartdata_preprocessed <- heartdata_recipe |>
  # apply the recipe to the test data
  prep(heartdata) |>
  # extract the pre-processed test dataset
  juice()



# Fitting the logistic regression model
logit_fit <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification") |>
  fit(as.factor(HadHeartAttack) ~ ., data = heartdata_train_preprocessed)

# Printing the tidy output with proper alignment
tidy_heartdata <- tidy(logit_fit, exponentiate = TRUE)

# Format the p-values to have consistent decimal places
tidy_heartdata$p.value <- format(tidy_heartdata$p.value, digits = 3)

# Print the formatted tidy output
print(tidy_heartdata, n = 100)

#filtering the significant variables using 0.05 as threshold
logit_heartdata = tidy(logit_fit, exponentiate = TRUE)
print(logit_heartdata, n=100)
significant_variables <- logit_heartdata %>%
  filter(p.value < 0.05)
# Display the significant variables
print(significant_variables, n = 100)

# Make predictions on the test set
pred_class <- predict(logit_fit,
                      new_data = heartdata_test_preprocessed,
                      type = "class")
test_logit_result <- heartdata_test_preprocessed |>
  select(HadHeartAttack) |>
  bind_cols(pred_class)
heartdata_test_preprocessed$HadHeartAttack <- as.factor(heartdata_test_preprocessed$HadHeartAttack)

conf_mat(test_logit_result, truth = HadHeartAttack,
         estimate = .pred_class)
accuracy(test_logit_result, truth = HadHeartAttack,
         estimate = .pred_class)

heartdata$HadHeartAttack <- as.factor(heartdata$HadHeartAttack)

library(yardstick)
library(dplyr)

# Ensure that the variables are factors
test_logit_result$HadHeartAttack <- as.factor(test_logit_result$HadHeartAttack)
test_logit_result$.pred_class <- as.factor(test_logit_result$.pred_class)

# Calculate sensitivity using yardstick
sensitivity <- sens(data = test_logit_result, truth = HadHeartAttack, estimate = .pred_class, positive = "Yes")
# Calculate specificity, precision, and recall using yardstick
specificity <- yardstick::spec(data = test_logit_result, truth = HadHeartAttack, estimate = .pred_class, negative = "No")
precision <- yardstick::precision(data = test_logit_result, truth = HadHeartAttack, estimate = .pred_class, positive = "Yes")
recall <- yardstick::recall(data = test_logit_result, truth = HadHeartAttack, estimate = .pred_class, positive = "Yes")











