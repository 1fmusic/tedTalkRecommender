# Ted Talk Recommender

Order of the files: (keep reading for descriptions) 

1. [ted_clean_explore.nb][1] 
2. [topic_modeling_ted_1.nb][6] 
3. [Recommender_ted.nb][3] 
4. [ted_rec.html][4] 
5. [ted_app.py][5] 
6. [ted_talks_2_elastic_slides.ipynb][7]
7. [tsne_environment.yml][8]
8. [topic_modeling_tSNE_tutorial3.ipynb][9]


[7]: https://github.com/1fmusic/tedTalkRecommender/blob/master/ted_talks_2_elastic_slides.ipynb

[8]: https://github.com/1fmusic/tedTalkRecommender/blob/master/tsne_environment.yml
[9]: https://github.com/1fmusic/tedTalkRecommender/blob/master/topic_modeling_tSNE_tutorial3.ipynb


This repo contains Ipython/Jupyter notebooks for basic exploration of transcripts of Ted Talks using Natural Language Processing (NLP), topic modeling, and a recommender that lets you enter key words from the title of a talk and finds 5 talks that are similar.
    The data consists of transcripts from Ted and TedX talks. Thanks to the lovely Rounak Banik and his web scraping I was able to dowload transcripts from 2467 Ted and TedX talks from 355 different Ted events. I downloaded this corpus from Kaggle, along with metadata about every talk. I encourage you to go to kaggle and download it so that he can get credit for his scraping rather than put it in this repo.
    https://www.kaggle.com/rounakbanik/ted-talks
    

The initial cleaning and exploration are done in 
   
[ted_clean_explore.nb][1] 
    
[1]: https://github.com/1fmusic/tedTalkRecommender/blob/master/ted_clean_explore.ipynb/

   Start by importing the csv files and looking at the raw data. Combine the metadata and transcripts and save as 'ted_all' (both datasets have a url column, so we merge on that one). Create a variable that holds only the transcripts called 'talks'. Below is a sample of the transcript from the most popular (highest views) Ted Talk. 'Do Schools Kill Creativity? by Sir Ken Robinson. 

    Good morning. How are you?(Laughter)It\\'s been great, hasn\\'t it? 

The first thing I saw when looking at these transcripts was that there were a lot of parentheticals for various non-speech sounds. For example, (Laughter) or (applause) or (Music).  There were even some cute little notes when the lyrics of a performance were transcribed
    
    someone like him ♫♫ He was tall and strong,
 
   I decided that I wanted to look at only the words that the speaker said, and remove these words in parentheses. Although, it would be interesting to collect these non-speech events and keep a count in the main matrix, especially for things like 'laughter' or applause or multimedia (present/not present) in making recommendations or calculating the popularity of a talk.
   
    Lucky for me, all of the parentheses contained these non-speech sounds and any of the speaker's words that required parenthesis were in brackets, so I just removed them with a simple regular expression. Thank you, Ted transcribers, for making my life a little easier!!!
    
    clean_parens = re.sub(r'\\([^)]*\\)', ' ', text)
    
# Cleaning Text with NLTK
Four important steps for cleaning the text and getting it into a format that we can analyze:
1)tokenize
2)lemmatize
3)remove stop words/punctuation
4)vectorize
    
[NLTK (Natural Language ToolKit)][2] is a python library for NLP. I found it very easy to use and highly effective.
    
[2]: http://www.nltk.org/
    
 * **tokenize**- This is the process of splitting up the document (talk) into words. There are a few tokenizers in NLTK, and one called **wordpunct** was my favorite because it separated the punctuation as well.
    ```
    from nltk.tokenize import wordpunct_tokenize
    doc_words2 = [wordpunct_tokenize(docs[fileid]) for fileid in fileids]
    print('\\n-----\\n'.join(wordpunct_tokenize(docs[1])))
 
    OUTPUT:
    Good
    morning
    .
    How
    are
    you
    ?
    ```
   
The notes were easy to remove by adding them to my stop words. Stopwords are the words that don't give us much information, (i.e., the, and, it, she, as) along with the punctuation. We want to remove these from our text, too. 
    
* We can do this by importing NLTKs list of **stopwords** and then adding to it. I added a lot of words and little things that weren't getting picked up, but this is a sample of my list. I went through many iterations of cleaning in order to figure out which words to add to my stopwords.

```
      from nltk.corpus import stopwords,
      stop = stopwords.words('english')
      stop += ['.',\" \\'\", 'ok','okay','yeah','ya','stuff','?']
```
**Lemmatization** - In this step, we get each word down to its root form. I chose the lemmatizer over the stemmer because it was more conservative and was able to change the ending to the appropriate one (i.e. children-->child, capacities-->capacity). This was at the expense of missing a few obvious ones (starting, unpredictability).

```
        from nltk.stem import WordNetLemmatizer
        lemmizer = WordNetLemmatizer()
        clean_words = []
      
        for word in docwords2:
       
            #remove stop words
            if word.lower() not in stop:
                low_word = lemmizer.lemmatize(word)
     
                #another shot at removing stopwords
                if low_word.lower() not in stop:
                    clean_words.append(low_word.lower())
```
   
 Now we have squeaky clean text! Here's the same excerpt that I showed you at the top of the README.
 
 ```
    good morning great blown away whole thing fact leaving three theme running conference relevant want talk one extraordinary evidence human creativity
```
    
As you can see it no longer makes a ton of sense, but it will still be very informative once we process these words over the whole corpus of talks.

* **Vectorization** is the important step of turning our words into numbers. The method that gave me the best results was count vectorizer. This function takes each word in each document and counts the number of times the word appears. You end up with each word as your columns and each row is a document (talk), so the data is the frequency of each word in each document. As you can imagine, there will be a large number of zeros in this matrix; we call this a sparse matrix. 
    
```
    c_vectorizer = CountVectorizer(ngram_range=(1,3), 
                                 stop_words='english',
                                 max_df = 0.6, 
                                 max_features=10000)
    
    # call `fit` to build the vocabulary
    c_vectorizer.fit(cleaned_talks)
    
    # finally, call `transform` to convert text to a bag of words
    c_x = c_vectorizer.transform(cleaned_talks)
```
    
# Now we are ready for topic modeling!
Open:

[topic_modeling_ted_1.nb][6] 
    
[6]: https://github.com/1fmusic/tedTalkRecommender/blob/master/topic_modeling_ted_1.ipynb

    
First get the cleaned_talks from the previous step. Then import the models

```
    from sklearn.decomposition import LatentDirichletAllocation,  TruncatedSVD, NMF
```
    
We will try each of these models and tune the hyperparameters to see which one gives us the best topics (ones that make sense to you). It's an art.
    
This is the main format of calling the model, but I put it into a function along with the vectorizers so that I could easily manipulate the paremeters like 'number of topics, number of iterations (max_iter),n-gram size (ngram_min,ngram_max), number of features (max_df): 

```
    lda = LatentDirichletAllocation(n_components=topics,
                                        max_iter=iters,
                                        random_state=42,
                                        learning_method='online',
                                        n_jobs=-1)
       
    lda_dat = lda.fit_transform(vect_data)
```
The functions will print the topics and the most frequent 20 words in each topic.
    
The best parameter to tweak is the number of topics, higher is more narrow, but I decided to stay with a moderate number (20) because I didn't want the recommender to be too specific in the recommendations.

Once we get the topics that look good, we can do some clustering to improve it further. However, as you can see, these topics are already pretty good, so we will just assign the topic with the highest score to each document. 
```
    topic_ind = np.argmax(lda_data, axis=1)
    topic_ind.shape
    y=topic_ind
```
    
Then, you have to decide what to name each topic. Do this and save it for plotting purposes in topic_names. Remember that LDA works by putting all the noise into one topic, so there should be a 'junk' topic that makes no sense. I realize that as you look at my code, you will see that I have not named a 'junk' topic here.  The closest was the 'family' topic but  I still felt like it could be named.  Usually, when running the models with a higher number of topics (25 or more) you would see one that was clearly junk.   
``` 
        topic_names = tsne_labels
        topic_names[topic_names==0] = \"family\" 
        . . .
```
    
Then we can use some visualization tools to 'see' what our clusters look like. The pyLDAviz is really fun, but only plots the first 2 components, so it isn't exactly that informative. I like looking at the topics using this tool, though. Note: you can only use it on LDA models.
    
The best way to 'see' the clusters, is to do another dimensionality reduction and plot them in a new (3D) space. This is called tSNE (t-Distributed Stochastic Neighbor Embedding. When you view the tSNE ,it is important to remember that the distance between clusters isn't relevant, just the clumpiness of the clusters. For example, do the points that are red clump together or are they really spread out? If they are all spread out, then that topic is probably not very cohesive (the documents in there may not be very similar).  
    
After the tSNE plot, you will find the functions to run the other models (NMF, Truncated SVD). 
    

# Recommender

[Recommender_ted.nb][3] 
    
[3]: https://github.com/1fmusic/tedTalkRecommender/blob/master/Recommender_ted.ipynb
    
Load the entire data set, and all the results from the LDA model.
The function will take in a talk (enter the ID number) and find the 10 closest talks using nearest neighbors.  
    
The distance, topic name, url, and ted's tags for the talk will print for the talk you enter and each recommendation. 

# Flask App
    
There is an even better verion of the recommender in the form of a flask app. This app can also 'find' your talk even if you don't remember the title.

[ted_rec.html][4] 
    
[4]: https://github.com/1fmusic/tedTalkRecommender/blob/master/ted_rec.html

[ted_app.py][5] 
    
[5]: https://github.com/1fmusic/tedTalkRecommender/blob/master/ted_app.py

    
You enter the keywords, or words from the title. 
Then, it returns your talk's title and url along with 5 similar ted talks (urls) that are similar to yours.  

# Prep and push into an elasticsearch database

I gave a talk at the Devfest DC 2019 where I discussed taking our ted talk data and model results and ingesting them into an elasticsearch index so that we can use Kibana to view our results and search the data. The last notebook is the code that prepares our dataframe for ingestion (some cleaning). I show several examples for various ways to get data of this type (one big dataframe) into elasticsearch since most tutorials focus on other types of streaming data. 

[ted_talks_2_elastic_slides.ipynb][7]

# t-SNE plotting with plotly

This workbook is from a 'Cakes and Tensors' presentation at Booz Allen Hamilton. It focuses on t-SNE plotting using matplotlib and plotly, as well as saving the data to upload to plotly. There is also a yml file if you want to recreate my conda environment (the instructions for this are at the top of the notebook).  You can view this notebook with nbviewer at https://nbviewer.jupyter.org/github/1fmusic/tedTalkRecommender/blob/master/topic_modeling_tSNE_tutorial2.ipynb

[topic_modeling_tSNE_tutorial3.ipynb][8]
[tsne_environment.yml][9]
 
