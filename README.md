PROBLEM STATEMENT
Ames Realty Company has in the past relied heavily on the opinions and expertise of their realtors to manually determine the prices of the homes that they list. However, as the industry moves to automating the appraisal of houses they are ready to join the growing number of real estate companies that uses machine learning to estimate the sale prices of their properties.

My task: As a member of the newly-hired data science team, I will use housing data collected from 2006 to 2010 to create a model that can take in the housing data and return the predicted fair sale price of houses. Model success will be evaluated on the Root Mean Squared Error(RMSE) score.


EXPLORATORY DATA ANALYSIS
1 Check for nulls
•There were a lot of NA entries in this dataset. Almost 6% of th data was missing.
•After digging into the provided data dictionary, I learned that categorical columns with NA usually meant that feature did not exist. For example, the quality/conditions of garages, basements or pools was Excellent, Good, Average/Typical, Fair and NA. The missing numerical values were lined up evenly with these. NA for the sq ft of a pool or basment for a house without them. To fix this problem, I made a for loop to iterate through the columns and any nulls with an int or float type fill with 0 and any with a object type reaplce with the string 'none'.

2.Change column names
•I changed all the column names to be snake case and renamed some to be more understandable and easier to remember.
•Then using dictionary method, I created a rename_columns function so I could easily rename the columns for both the training data and test data.
3.Consolidate columns such as full bath and half bath & 1st floor sq ft and 2nd floor sqft and drop originals.

4.Used feature engineering
•.map to bianarize columns with yes/no
•for caterogical columns that would be more useful as a number ranking
•to create a 'newly_renovated' column which has a 1 if the house has been renovated since 2000 and a 0 if it has older or no renovations.

5.Created a function to run 3 and 4 on both the training data and test data.

6.Created dummy columns for
•'basement' to separate finished, unfinished and none
•functionality to be typical or poor
•building type to be single family, duplex, condo etc.

7.Saved my clean csv as train_clean.csv. Found in the data folder of this notebook.

CORRELATION AND FEATURE SELECTION
1.Created a list of 'good columns' based on my assumptions about which columns would be most important for predicting the price of a home.

2.Plotted histograms for each column to check their distributions.

3.Plotted a heatmap of the features including SalePrice.
checked for correlations between features and SalePrice
checked for correlations among features, looking out for anything too high to either avoid using both or find relationships that could be elevated by interacting.

4.Based on the strong positive and negative correlations on the heatmap, I narrowed to a list of better columns.

5.Plotted a pairplot and new heatmap to compare the better columns to SalePrice specifically.

6.Created a features list for all the better columns, excluding SalePrice.

MODELING
1.Defined X(features) and y(target).

2.Polynomial Features:
instatiate
fit & transform X
call the built in .get_feature_names function
create a new DataFrame with the trasformed X and poly features.
join y onto new DataFrame.


3.Set new X for the polynomial features.

4.Train/test/split to randomly separate my training data in two parts, train and test. Since I will only fit my model to the train portion, I will be able to test the model on the test data.

5.Next, I need to scale the data to make all the units like feet, years, quality rating are comparable.
Using StandardScaler, I fit & transform the train portion of the data and only transform the test portion.

6.Now I need to run a Cross Validation Score on each model (Linear Regression, Ridge, LASSO) to see which one will potentially perform best. I use a K of 5 which tests 5 different variations of the 75/25 split from Train/Test/Split.
Linear Regression performed the best, but since I'm using so many features I want to proceed with LASSO. It is the best model for feature selection and feature elimination when necessary.

7.I find my optimal alpha to be 0.001 and fit my training set to lasso.

8.Then I can score my model on both the train and test, making sure to use the scaled versions.
R2 for train = 0.89
R2 for test = 0.88
These R2 score tell me that 89% of the variance in the SalePrice is explained by the lasso model compared to the baseline prediction. The fact that the R2 scores are nearly the same and both close to 1.0 tells me that the model has low variance and low bias.

9.Now I can plot my residuals, the difference between my predicted prices and the actual prices, and check for homoscedasticity.
This plot shows I have some outliers in the higher priced homes, but overall the errors have a pretty normal distribution, evenly split above and below zero, which is what we want.
RMSE score = $28,143.48

After checking the LASSO coeficients, I see that model did not actually drive down any of them to 0 like it would if it didn't need all of them. This tells me I should try Ridge instead.

Start by fitting the Ridge model to my scaled training set. Then I find the R2 scores for the training and test sets.
R2 for train = 0.88
R2 for test = 0.89
These scores tell me the same thing as LASSO, that 89% of the variability is accounted for with my model and that I have low variance and bias. However, this time the test score is just a bit higher than the train score which means the model performed better on data it hadn't seen before, which is a good thing!

Lastly, I plotted the Ridge residuals.
There are a few outliers in the top right corner. This tells me that the model is having a harder time predicting the price of some of the more expensive homes. Overall, the errors are somewhat linear and normally distributed. -RMSE Score = $27,570

14.Since I didn't generate an optimal alpha on the first try for Ridge, I do that next to see if it will help my score. My optimum alpha comes out to be 1353.0477, however, I actually get a worse R2 score.

15.I decide to dig in to my Ridge model more and sort by residuals to find that the ones over 140000 were 3 houses with large square footage at the higher end of the price scale.

16.I sorted the data to show only houses  $500,000 or higher. I discovered that there were only 13.
I used .loc to check different features when trying to see what they all had in common that I could possibly use to improve the model, the biggest standout was neighborhood. All 13 were in one of two neighborhoods.
I also check to see if the numerical features, Pool Quality and/or Misc Value have any correlation to the high priced homes and am very surprised to see none of them have either.

EXECUTIVE SUMMARY
My model gives credence to generalization. Therefore I am very confident with it. With a high accuracy score that is similar for both the train and test data, I am really happy my organization will have less problem predicting future sale price.And with this in mind, I have low variance and low bias. There are a few outliers that tell me the model is not scoring well on a few high price homes so there is still room for improvement. With more time and resources I would dive deeper into the categorical feature neighborhoods because I found that houses over $500,000 were all in one of two neighborhoods.