#### Introduction

For the third project of the [Metis Data Science Bootacmp](www.thisismetis.com), I built a classification model to predict the probability of an animal being returned from adoption. I used [Sonoma County's Animal intake and outake data](https://data.sonomacounty.ca.gov/Government/Animal-Shelter-Intake-and-Outcome/924a-vesw) to build my model. After comparing multiple models, I used XGBoost to build my final classification model. The metric I prioritized was recall because I want my model to capture as many adoption returns as possible. For feature importance, the most important feature was whether an animal was spayed or neutered. All in all, my classification model wasn't perfect. I think I'm missing some key data, such as animal behavioral data, that could better explain why an animal was returned from adoption.

#### Individual Contributor

* Jacky Lu

#### Project Motivation

Initially, I wanted my project to focus on animal adoptions and which traits lead to animals being adopted. However, I found the large majority of animals in my dataset were adopted or returned to their owner. This didn't present  much of a classificatioin problem.

Instead, I focused on another interesting problem: when an animal is taken in because they were returned from adoption. This is a spin on the classic customer churn classification problem. 

My main question is:

- What features are most associated with adoption returns?

#### Project Submission Directory

Data Collection Jupyter Notebook

- Workflow to collect data from Sonoma County's web API and store as a pandas dataframe
- Data is prepped and cleaned before being saved into a pickle object

EDA (exploratory data analysis) Jupyter Notebook

- Notebook to explore my data and how my features correlate to the target variable

Classification Exploration Jupyter Notebook

- Notebook used to explore different classification algorithms such as kNN, Logistic Regression, Random Forest, Decision Tree, Naive Bayes, and XGBoost

Final Model Jupyter Notebook

- Notebook used to build my final classification model using XGBoost
- Data is prepped to build a Tableau visualization

#### Data Science Analysis

1. Data Collection

   [Sonoma County's Animal intake and outake data](https://data.sonomacounty.ca.gov/Government/Animal-Shelter-Intake-and-Outcome/924a-vesw) was accessed through the Socrata web API. Information about 19,690 animals was recorded. The following features were used in my classification model:

   - Type, Size, Age, Gender, Intake condition, Intake location
   - Whether the animal had a name, is spayed/neutered, is a breed of pit bull

2. Data Preparation

   The data provided by Sonoma County was fairly complete. I only dropped 31 rows (0.1% of my data) where the size attribute was missing.

   I used the existing column specifying sex, which had 5 categories: spayed, neutered, male, female, and unknown, to make 2 new columns. The two new columns I made were for gender and for whether the animal was spayed or neutered.

   Additionally, I created a new column for age based on the birth year for each animal.

   For breed, there were hundred of different unique breeds. I thought about grouping them into bins, but many animals were mixed breeds or their breeds were not specific (domestic SH (short haired?) for example). The only feature for breed I used was whether an animal was a pit bull or not. Like many other areas in America, Sonoma County has an ordinance mandating sterilization of pit bulls. Pit bulls have a negative reputation for being aggressive dogs, and I wanted to examine whether this owuld play a role in an animal being more likely to be returned from adoption.

   I created a new column specifying whether an animal had a name or not.

   Last, to prepare my data for different classification algorithms that can't handle missing values, I imputed any data points missing the age attribute with the median age.

3. Exploratory Data Analysis

   How many animals were adoption returns? Only 1.7% (around 339 animals out of 19,351) were adoption returns. I have a big class imbalance problem.

   For my exploratory data analysis, I plotted comparative bar charts looking at different features between adoption returns and other animals. 

   With my categorical variables, I didn't see much of a difference in the distributions for gender, size, and type of animal. For intake condition, there was a higher distribution of healthy animals that were adoption returns. Many of the animals that were not adoption returns had an unknown or treatable condition. Interestingly, intake location had some interesting differences. The large majority of animals that were not adoption returns were listed as taken in Sonoma County or Santa Rosa whereas most of the adoption returns were also in different cities like Windsor, Petaluma, or Rohnert Park. My hypothesis is that a large majority of animals were taken from outside and in the field. However, adoption returns are more likely to take place within a cities' animal shelter.

   With the binary variables, there wasn't a clear distinction on whether adoption returns have a name or not. Similarly, pit bulls weren't any more or less likely to be adoption returns. However, pretty much all spayed or neutered animals were adoption returns, which makes sense. Animals have to be spayed or neutered before being adopted. 

   Finally, for my only continous variable (age), the distribution for animals that were adoption returns versus non-returns was pretty similar.

   Overall, I didn't find too many features that differentiate animals that were adoption returns. The features I think are the most important would be whether the animal intake location was in a city and whether an animal was spayed or neutered.

4. Classification Model Building

   First, I split my data using an 80/20 train test split. I made sure to stratify the train, test, split datsets so the minority class would be equally represent.

   Because of my class imbalance issue, I tried both oversampling my data and undersampling my data. I also adjusted class weights and the probability threshold.

   The models I tested were: [K Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [Naive Bayes (Bernoulli)](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html), and [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html). To compare the models, I look at both F and ROC/AUC scores. The F score represents the harmonic mean between precision and recall. ROC/AUC score represents the ability of a model to distinguish between a positive and negative class. Random Forest, Logistic Regression, and XGBoost had the highest scores.

   Ultimately, I used XGBoost for my final model. Logistic Regression required me to drop some dummy variables to avoid the dummy variable trap and avoid colinearity. I want to see the potential impact of all my features, so I didn't want to use Logistic Regression. With Random Forest, there aren't as many hyperparameters to tune. XGBoost has higher potential to tune and build a better final model, so that's why I didn't use Random Forest.

   For my final model with XGBoost, I undersampled my data by a tenth and set a probability threshold of 0.4 instead of 0.5 to increase the recall score. I used RandomizedSearchCV and GridSearchCV to find the XGBoost Classifier with the highest ROC/AUC and F scores. My final model tested on my test set had 0.79 accuracy, 0.06 precision, 0.74 recall, 0.11 F score, and 0.76 ROC/AUC score. The metric I most care about is recall as I want to capture all animals that were adoption returns. Unfortunately, my precision score had to be extremely low to get a 0.74 recall score.

5. Conclusions

   The top 10 features that affect the gain score (measures the relative contribution of a feature) of my final XGBoost Classifier model are:

   - Spay/Neuter, Intake Location: Sonoma County, Intake Location: Windsor, Intake Location: Santa Rosa, Intake Condition: Healthy, Type: Dog, Intake Locaiton: Out of COunty, Intake Location: Rohnert Park, Intake Condition: Manageable, Size: Medium

   The feature with the highest gain score is whether an animal is spayed or neutered. I would expect all pets returned from adoption to be spayed or neutered. Many of the other top features relate to location. Adoption returns would occur at animal shelters in specific cities, rather than being outdoors or in the field. It makes sense that an animal would have a higher chance of having an intake location that's a city in Sonoma County. Interesting to note, whether an animal was a pit bull was not an important feature. This indicates being a pit bull doesn't necessarily make an animal more likely to be an adoption return. Other features like size, gender, or age also didn't have high feature importance.

   Overall, trying to predict whether an animal will be returned from adoption is a hard problem. It was hard to build a model that had high recall without sacrificing too much precision. Looking at my model feature importance, it doesn't seem like any characteristics (type, size, gender, age, etc.) are more associated with adoption returns. Rather, I hypothesize that individual animal behaviors or prohibitive living circumstances (certain living complexes may not allow pets) are more correlated to whether an animal will be returned from adoption.

#### Design Decisions

- I dropped any rows where size was a missing attribute. This turned out to be 31 rows or 0.1% of my data.
- Additionally, I imputed any rows with missing ages with the median age (7 years). This was about 4,700+ rows or 23% of my data. I felt age was an important attribute to examine and was my only continous variable, so I would rather keep that value rather than drop 23% of my data.

#### Additional Notes

Techniques

* Exploratory Data Analysis
* Web API Data Collection
* Classification
  * kNN
  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Naive Bayes
  * XGBoost

Programs

* Jupyter Notebook
* Matplotlib
* Numpy
* Pandas
* Seaborn
* Scikit-learn
* XGBoost

Language

* Python