# Auto-Tagging-System

## Problem Statement

The project deals with how to develop tags for the given text. For example, on a stackoverflow question page, title and description are mentioned and below are the tags. Now there are 100s of thousands of questions and it may not be possible for domain experts to read and then answer them individually. To address this issue, tags comes into the picture. 

![1](https://user-images.githubusercontent.com/36281158/86476020-95d79180-bcfa-11ea-8da5-dfbdbb15916a.PNG)

In the above picture , we can see the title, description and tags. Through tags, we can have an idea what the question is about.The tags and number of tags can be different for different texts. In this project we will build an auto tagging system that will automatically generate the tags for the given texts.  


## Dataset 

![2](https://user-images.githubusercontent.com/36281158/86477519-6fffbc00-bcfd-11ea-9e27-2ebed6f4af63.PNG)

We will be using F1 score as a Performance metric for this problem. F1 score is the harmonic mean of precision and recall values.

#### Precision
It is defined as ratio of true positives by (true positives + false positives)

#### Recall 
It is defined as ratio of true positives by (true positives + false negatives)

#### F1 Score = 2* Precision * Recall / (Precision + Recall) 

## Text Cleaning and Pre Processing

First major step is to clean the data. It contains variou html and urls links, punctuations, single letter alphabets and stopwords which dosent convey much information regarding what topics they are related to. 

Steps are as follows.

1. Combining the title and text part of our dataset.

2. Removing all the punctuations, alphabets through the regex and white spaces and stopwords 

3. Finally lowercasing all the words present in the text. 

![3](https://user-images.githubusercontent.com/36281158/86495256-71e17380-bd2d-11ea-870c-e8ea1a0bc636.PNG)


4. Reshaping our target variable(tag). Since they are 100 , we will be apply MultiLableBinarizer for sckit learn library. 100 columns more will be more in palce of one.

5. Applying Tfidf vectorizer over the text_cleaned part of our dataset having max_features= 10000 (keping only 10000 top words) and max_df=0.8 (words appearing in more than 80 % of text are removed)

6. Splitting the data using train_test_split using sklearn.model_selection library in 80/20 ratio 

## Fitting the model 

We will use the OneVsRestclassifier model (with logistic regression in it ) to fit over the training data as it calculates over the 100 tags simultaneously. We will test it on the test data and then will check on the unknow question and see its performance 

Fitting the model and then predicting the f1 score using the test data, we got a score of 0.46 which is ok, if we set the theshold of 0.45

## Results

If we predict on the first three questions, we get the following 

![4](https://user-images.githubusercontent.com/36281158/86500082-52d5f680-bdac-11ea-9a2e-1790611bdf27.PNG)



In above, if we apply inverse transform for multibinarizer, we will get the required tags below for the first three questions, as all the tags are represented by an array of size 100* 1 

[('prediction',), ('distributions', 'mean', 'variance'), ('r',)]





