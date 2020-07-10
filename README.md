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

5. Applying Tfidf vectorizer over the text_cleaned part of our dataset having max_features= 10000 (keping only 10000 top words) and max_df=0.8 (words appearing in more than 80 % of text are removed) and Word2Vec model (keeping number of features = 100 initially and then varying) for feature engineering 

6. Splitting the data using train_test_split using sklearn.model_selection library in 80/20 ratio and then using the logistic regression for prediction. 

## Feature Engineering 

### TFIDF: Term frequency and Inverse Document Frequency 

Tf-Idf stands for term frequency inverse document frequency. Term Frequency summarizes how often a given word appears within a document and Inverse Document Frequency downscales words that appear a lot across documents. Tfidf scores are the product of these two terms. Without inverse document term, words which appear more common (that carries little information) would be weighed higher and words appearing rare(which contains more information) would be weighed less.

Inverse document frequency is defined as:

idf(t,D)=log (N/ |d∈D:t∈d|)

Where N is the total number of documents (questions) in the corpus, and |d∈D:t∈d| is the number of documents where t appears

A vecotizer object is created by calling the TfidfVectorizer class. Some arguments for it are as follows.

max_features = 1000, it meanns we want only 1000 words to decribe the context of the given question 

max_df= 0.8, it means removing those words which have appeared in more than 80 % of the questions

Tfidf matrix has a shape of 76365* 1000 where 76365 is the number of the questions and 1000 is the number of features describing that question. 

![1](https://user-images.githubusercontent.com/36281158/87052266-3576b180-c21e-11ea-9c8b-439a8dae8658.PNG)

In the above snapshot the maxtrix , value address the importance of the word to that question. High value means more importance. We can increase the max_features but too much of 0 values will make it more sparse and computalionally expensive. We can tweak it around a bit and see the change when trying to fit the model. 


### Word2Vec Word Embeddings 

Word2vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located close to one another in the space. There are two models for generating the embeddings, CBOW(Continous Bag of words) and skip gram model. CBOW is used for small corpus, it is fast but for larger corpus skip gram is better, takes a little more time than CBOW. Here skip gram is used, below its explanation.

#### Skip Gram 
It is used to get the word embeddings matrix for the words present in our corpus. It tries to predict the context given the word. So given the window size of 2, two words before and two after are predicted for the given word. 

![2](https://user-images.githubusercontent.com/36281158/87121435-5bdd3100-c2a0-11ea-9794-2ef4b3d8cb0a.PNG)

Skip Gram is a two layer network where the weight matrix of the hidden layer is calculated. First the vocabulary is contructed where each unique word in the corpus is indexed. Then all the words are one-hot encoded. Training examples are created as depicted below. Pairs are selected for each word given window size is 2. For a given word, 4 pairs will exits. If there is suppose 10,000 words vocabulary and 300 features, then given word is taken as an input to the network and output is output is a 10000 * 1 matrix where it contains probability values of all the words in vocab. So for each training example , corresponding words for the "given input word", their probablity values should be maximized and due to backpropogation , errors are minimzed and weights are updated. In the end, weight matrix obatained for the hidden layer (10,000 * 300) is the embedding matrix. This matrix is based on statistics on how much two words have appeared together or are close in the corpus. One close to the given word will have high probability values and one far away will have low probability value. To get the embedding of the given word, just multiply the one hot encoding vector of it with the hidden matrix layer.  

![5](https://user-images.githubusercontent.com/36281158/87122809-37cf1f00-c2a3-11ea-9ee2-305bc642abc8.PNG)

![3](https://user-images.githubusercontent.com/36281158/87121532-8e872980-c2a0-11ea-89f5-41bc2228cbd2.PNG)

![4](https://user-images.githubusercontent.com/36281158/87122804-36055b80-c2a3-11ea-9079-1c5574eb3a98.PNG)

Here, the number of features have been tweaked from 50 to 500 to see the change in the accuracy. Word2Vec model is being trained on the corpus of ~59000 words using the "Gensim" library. Also for the words, "value", "stastitics", "algorithm" are plotted in a 2D space (being 100 dimensional vectors) using PCA dimesionality reduction technique. Below is the projection. 

![6](https://user-images.githubusercontent.com/36281158/87123485-73b6b400-c2a4-11ea-93c0-cd5a77cf050f.PNG)


## Fitting the model 

We will use the OneVsRestclassifier model (with logistic regression in it ) to fit over the training data as it calculates over the 100 tags simultaneously. We will test it on the test data and then will check on the unknow question and see its performance 

Fitting the model and then predicting the f1 score using the test data, we got a score of 0.46 which is ok, if we set the theshold of 0.45

## Results

If we predict on the first three questions, we get the following 

![4](https://user-images.githubusercontent.com/36281158/86500082-52d5f680-bdac-11ea-9a2e-1790611bdf27.PNG)



In above, if we apply inverse transform for multibinarizer, we will get the required tags below for the first three questions, as all the tags are represented by an array of size 100* 1 

[('prediction',), ('distributions', 'mean', 'variance'), ('r',)]





