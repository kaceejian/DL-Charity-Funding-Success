# Deep Learning: Charity Funding Predictor

## Background

The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. Applying the knowledge of machine learning and neural networks,
a binary classifier was created using the features in the dataset. This classifier is capable of predicting wheather applicants will be successful if funded by Alphabet Soup.

There are more than 34,000 organizations that have received funding from Alphabet Soup over the years, and all these information had been stored in this CSV file. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- **EIN** and **NAME**—Identification columns
- **APPLICATION_TYPE**—Alphabet Soup application type
- **AFFILIATION**—Affiliated sector of industry
- **CLASSIFICATION**—Government organization classification
- **USE_CASE**—Use case for funding
- **ORGANIZATION**—Organization type
- **STATUS**—Active status
- **INCOME_AMT**—Income classification
- **SPECIAL_CONSIDERATIONS**—Special consideration for application
- **ASK_AMT**—Funding amount requested
- **IS_SUCCESSFUL**—Was the money used effectively

---

## Process

### Step 1: Preprocess the data

Utilize Pandas and the Scikit-Learn’s `StandardScaler()`, preprocess the dataset in order to compile, train, and evaluate the neural network model later in Step 2.

1. Read in the charity_data.csv to a Pandas DataFrame, and identify the following in the dataset:

- What variable(s) are considered the target(s) for the model?
- What variable(s) are considered the feature(s) for the model?

2. Drop the `EIN` and `NAME` columns.
3. Determine the number of unique values for each column.
4. For those columns that have more than 10 unique values, determine the number of data points for each unique value.
5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then confirm if the binning was successful.
6. Use `pd.get_dummies()` to encode categorical variables

### Step 2: Compile, Train, and Evaluate the Model

Using TensorFlow, design a neural network/deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. Consider how many inputs there are before determining the number of neurons and layers in the model. Then, compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy.

1. Continue after Step 1, create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
2. Create the first hidden layer and choose an appropriate activation function.
3. If necessary, add a second hidden layer with an appropriate activation function.
4. Create an output layer with an appropriate activation function.
5. Check the structure of the model.
6. Compile and train the model.
7. Create a callback that saves the model's weights every 5 epochs.
8. Evaluate the model using the test data to determine the loss and accuracy.
9. Save and export the results to an HDF5 file, `AlphabetSoupCharity.h5`.

### Step 3: Optimize the Model

Using TensorFlow, optimize the model in order to achieve a target predictive accuracy higher than 75%. If can't achieve an accuracy higher than 75%, try to make at least three attempts to do so.

Optimize the model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

- Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
  - Dropping more or fewer columns.
  - Creating more bins for rare occurrences in columns.
  - Increasing or decreasing the number of values for each bin.
- Adding more neurons to a hidden layer.
- Adding more hidden layers.
- Using different activation functions for the hidden layers.
- Adding or reducing the number of epochs to the training regimen.

1. Create a new Jupyter Notebook file: `AlphabetSoupCharity_Optimzation.ipynb`.
2. Import the dependencies, and read in the `charity_data.csv` to a Pandas DataFrame.
3. Preprocess the dataset as did in Step 1, taking into account any modifications to optimize the model.
4. Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
5. Save and export the results to an HDF5 file, `AlphabetSoupCharity_Optimization.h5`.

### Step 4: Create a Report on the Neural Network Model

Write a report on the performance of the deep learning model created for AlphabetSoup.

The report contains the following:

1. **Overview** of the analysis: Explain the purpose of this analysis.

- I wanted to create a binary classifier to predict whether applicants will be successful if funded by Alphabet Soup.  
  In order to be able to predict this, the approach is to create a machine learning model that can solve the task and be able to optimize the result through tuning the parameters.

2. **Results**: Use bulleted lists and images to support answers, address the following questions.

- Data Preprocessing

  - What variable(s) are considered the target(s) for the model?
    - The “IS_SUCCESSFUL” column is the target for the model. The variable for it is target_y.
    - After splitting into training and testing data sets, for training, the variable for it is y_train. For testing set, it is y_test.
  - What variable(s) are considered to be the features for the model?

    - Most of the rest of the columns are considered features, the variable for them is features_x.
    - After splitting into training and testing data sets, for training, the variable for it is x_train. For testing set, it is x_test.
    - See screenshot of notebook below:
      ![ScreenShot1](Resources/Images/1.png)

  - What variable(s) are neither targets nor features, and should be removed from the input data?
    - "EIN" and "NAME" columns were removed, because they are not features. They are not useful information for what I'm trying to predict in this project.
    - See screenshot of notebook below:
      ![ScreenShot2](Resources/Images/2.png)

- Compiling, Training, and Evaluating the Model

  - How many neurons, layers, and activation functions were selected for the neural network model, and why?
  - Was the target model performance achieved?
  - What steps were taken to try and increase model performance?

    - I’ve tested 3 combinations of different number of neurons, batch size and number of layers:
    - nn_1 = build_and_test_model(n_hidden_neurons=12, batch_size=128, n_layers=3)
    - nn_2 = build_and_test_model(n_hidden_neurons=4, batch_size=64, n_layers=3)
    - nn_3 = build_and_test_model(n_hidden_neurons=8, batch_size=64, n_layers=5)
    - The accuracy of nn_1 is about: 0.732 (see below screenshot)
      ![ScreenShot3](Resources/Images/3.png)
    - The accuracy of nn_2 is about: 0.726 (see below screenshot)
      ![ScreenShot4](Resources/Images/4.png)
    - The accuracy of nn_3 is about: 0.730 (see below screenshot)
      ![ScreenShot5](Resources/Images/5.png)

  - Based on the above results, I would select nn_1 for the neural network model, since it has the highest accuracy among the three combinations/attempts.

  - During the 3 attempts, I tried to increase the model performance by trying out different combinations of number of neurons, batch size and number of layers. So far, from the 3 combinations, we can see that “nn_1”, which has a combo of 12 neurons, a batch size of 128, and 3 layers, gave us the best accuracy among them; but to achieve the target model performance of 0.75, I would keep testing different combinations of these parameters, and try to find a combo with optimized results.

3.  **Summary**: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain then recommendation.

        - In summary, in order to be able to predict whether the loan applicant will be success, I’ve created a neural network classification model that so far has a 73.2% accuracy.

    I’ve tested different combinations of the parameters, but I haven’t been able to achieve the 75% accuracy desired. As mentioned above, we can keep testing different combinations of the parameters of number of neurons, batch size and number of layers, or we can also look deeper into what other features might also determine “Is Successful”, and keep adjusting the model to come up with optimized results.
    Another model that could also be used for this project would be the Random Forest Classification, because it is also good with the binary classification problem. Also, the performance of Random Forest Classification could be comparable with the neural network model when it only has few hidden layers. And also, since there are fewer parameters to work with, and overall more intuitive, Random Forest Classification could be a good alternative to the deep learning model for this project.

---

## Thanks!
