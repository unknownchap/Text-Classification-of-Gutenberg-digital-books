from urllib import request
import nltk 
import nltk.corpus  
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import random
from text_filter import text_filter
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import statistics as stat




# =============================================================================
# Getting the 1st book from the Gutenberg library 
#The Three Musketeers by Alexandre Dumas, Pere
url = "https://www.gutenberg.org/files/1257/1257-0.txt"
response = request.urlopen(url)
TheThreeMusketeers = response.read().decode('utf8')
TheThreeMusketeers = TheThreeMusketeers[
        TheThreeMusketeers.find("AUTHOR’S PREFACE"):
            TheThreeMusketeers.rfind("END OF THIS PROJECT GUTENBERG EBOOK")]
tokens_TheThreeMusketeers = word_tokenize(TheThreeMusketeers)
text_TheThreeMusketeers = nltk.Text(tokens_TheThreeMusketeers)

# =============================================================================

# =============================================================================
# Getting the 2nd book from the Gutenberg library 
#Adventures of Sherlock Holmes, by A. Conan Doyle
url = "https://www.gutenberg.org/files/1661/1661-0.txt"
response = request.urlopen(url)
SherlockHolmes = response.read().decode('utf8')


SherlockHolmes = SherlockHolmes[SherlockHolmes.find("I. A SCANDAL IN BOHEMIA"):
    SherlockHolmes.rfind("End of Project Gutenberg's")]
tokens_SherlockHolmes = word_tokenize(SherlockHolmes)
text_SherlockHolmes = nltk.Text(tokens_SherlockHolmes)

# =============================================================================

# =============================================================================
#  Getting the 3rd book from the Gutenberg library 
#Dorothy and the Wizard in Oz, by L. Frank Baum.
url = "http://www.gutenberg.org/cache/epub/420/pg420.txt"
response = request.urlopen(url)
Oz = response.read().decode('utf8')


Oz = Oz[Oz.find("To My Readers"):
    Oz.rfind("End of Project Gutenberg's")]
tokens_Oz = word_tokenize(Oz)
text_Oz = nltk.Text(tokens_Oz)
# =============================================================================

# =============================================================================
#  Getting the 4th book from the Gutenberg library 
# The Mysterious Island, by Jules Verne
url = "https://www.gutenberg.org/files/1268/1268-0.txt"
response = request.urlopen(url)
MysteriousIsland = response.read().decode('utf8')


MysteriousIsland = MysteriousIsland[
        MysteriousIsland.find("PART 1--DROPPED FROM THE CLOUDS"):
            MysteriousIsland.rfind("End of the Project Gutenberg EBook")]
tokens_MysteriousIsland = word_tokenize(MysteriousIsland)
text_MysteriousIsland = nltk.Text(tokens_MysteriousIsland)
# =============================================================================

# =============================================================================
#  Getting the 5th book from the Gutenberg library
# Mechanical Drawing Self-Taught, by Joshua Rose
url = "http://www.gutenberg.org/cache/epub/23319/pg23319.txt"
response = request.urlopen(url)
MechanicalDrawing = response.read().decode('utf8')


MechanicalDrawing = MechanicalDrawing[
        MechanicalDrawing.find("_THE DRAWING BOARD._"):
            MechanicalDrawing.rfind("End of Project Gutenberg's")]
tokens_MechanicalDrawing = word_tokenize(MechanicalDrawing)
text_MechanicalDrawing = nltk.Text(tokens_MechanicalDrawing)
# =============================================================================

# =============================================================================
# Getting the 6th book from the Gutenberg library
# The History and Practice of the Art of Photography, by Henry H. Snelling
url = "http://www.gutenberg.org/cache/epub/168/pg168.txt"
response = request.urlopen(url)
ArtofPhotography = response.read().decode('utf8')


ArtofPhotography = ArtofPhotography[
        ArtofPhotography.find("INTRODUCTION"):
            ArtofPhotography.rfind("End of the Project Gutenberg EBook")]
tokens_ArtofPhotography = word_tokenize(ArtofPhotography)
text_ArtofPhotography = nltk.Text(tokens_ArtofPhotography)
# =============================================================================

# =============================================================================
#  Getting the 7th book from the Gutenberg library
# ChristmasCarol, by Charles Dickens
url = "https://www.gutenberg.org/files/24022/24022-0.txt"
response = request.urlopen(url)
ChristmasCarol = response.read().decode('utf8')


ChristmasCarol = ChristmasCarol[
        ChristmasCarol.find("Marley was dead, to begin with"):
            ChristmasCarol.rfind("End of the Project Gutenberg EBook")]
tokens_ChristmasCarol = word_tokenize(ChristmasCarol)
text_ChristmasCarol = nltk.Text(tokens_ChristmasCarol)

# =============================================================================

# =============================================================================
books= [tokens_ArtofPhotography,
        tokens_MechanicalDrawing,
        tokens_MysteriousIsland,
        tokens_Oz,
        tokens_ChristmasCarol,
        tokens_SherlockHolmes,
        tokens_TheThreeMusketeers]

# =============================================================================
    


# =============================================================================
# Making the dataset
df_books = pd.DataFrame(data={"token_words": [books[0], 
                                              books[1],
                                              books[2],
                                              books[3],
                                              books[4],
                                              books[5],
                                              books[6]],
                                        "Author": ["Henry H. Snelling",
                                                       "Joshua Rose",
                                                       "Jules Verne",
                                                       "L. Frank Baum",
                                                       "Charles Dickens",
                                                       "A. Conan Doyle",
                                                       "Alexandre Dumas"]})
score_table = np.zeros((20, 5))
score_table_train = np.zeros((20, 5))
for n in range(10, 200, 10): #number of words in each document
    df = pd.DataFrame(data={"token_words": ([None] * 1400), "Author":([None] * 1400)})
    df["token_words"] = df["token_words"].astype(object)
    for i in range(7):
        for j in range(200):
            rn = random.sample(range(0, len(df_books.at[i, "token_words"])), n)
            select_doc = []
            for r in rn:
                select_doc.append(df_books.at[i, "token_words"][r])
            df.at[((i*200)+j), "token_words"] = select_doc
            df.at[((i*200)+j), "Author"] = df_books.at[i, "Author"]
    
    # =============================================================================
    
    # =============================================================================
    # Removal of Punctuation Marks and Stopwords, and Lemmatisation of verbs
    # Using pre-defined function of text_filter
    df_joined = pd.DataFrame(index=range(1400), columns=["Text", "Author"])
    df_joined_filtered=pd.DataFrame(index=range(1400), columns=["Text", "Author"])
    for i in range(1400):
        df_joined.at[i,"Text"] = " ".join(df.at[i, "token_words"])
        df_joined.at[i,"Author"] = df.at[i, "Author"]
        df_joined_filtered.at[i,"Text"] = text_filter(df_joined.at[i,"Text"])
        df_joined_filtered.at[i,"Author"] = df.at[i, "Author"]
    # =============================================================================
    
    
    
    
    # =============================================================================
    # Label Encoding of Classes
    y = df_joined_filtered['Author']
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    # =============================================================================
    
    # =============================================================================
    # Splitting the dataset (80%: Training and 20%: Test)
    X = df_joined_filtered["Text"]
    X_train, X_test, y_train, y_test = train_test_split(X, y
                                      ,test_size=0.2)
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2)
    # =============================================================================
    
    # =============================================================================
    # Bag of Words Transformation
    bow_transformer = CountVectorizer().fit(X_train)
    text_bow_train = bow_transformer.transform(X_train)
    text_bow_test = bow_transformer.transform(X_test)
    
    # =============================================================================
    
    
    # =============================================================================
    # Classification Using Multinomial Naive Bayes
    classifier1 = MultinomialNB()
    classifier1 = classifier1.fit(text_bow_train, y_train)
    score_table_train[int(n/10-1), 0] = classifier1.score(text_bow_train, y_train)
    score1 = cross_val_score(classifier1, bow_transformer.transform(X), y, cv=cv)
    score_table[int(n/10-1), 0] = stat.mean(score1)
    pred1 = classifier1.predict(text_bow_test)
    pred_name1 = labelencoder.inverse_transform(pred1)
    cm1 = confusion_matrix(y_test, pred1)
    # =============================================================================
    
    # =============================================================================
    # Classification Using K Nearest Neighbors (K-NN)
    classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier2.fit(text_bow_train, y_train)
    pred2 = classifier2.predict(text_bow_test)
    pred_name2 = labelencoder.inverse_transform(pred2)
    score_table_train[int(n/10-1), 1] = classifier2.score(text_bow_train, y_train)
    score2 = cross_val_score(classifier2, bow_transformer.transform(X), y, cv=cv)
    score_table[int(n/10-1), 1] = stat.mean(score2)
    cm2 = confusion_matrix(y_test, pred2)
    # =============================================================================
    
    # =============================================================================
    # Classification Using Support Vector Machine (SVM)
    classifier3 = SVC(kernel = 'linear')
    classifier3.fit(text_bow_train, y_train)
    pred3 = classifier3.predict(text_bow_test)
    pred_name3 = labelencoder.inverse_transform(pred3)
    score_table_train[int(n/10-1), 2] = classifier3.score(text_bow_train, y_train)
    score3 = cross_val_score(classifier3, bow_transformer.transform(X), y, cv=cv)
    score_table[int(n/10-1), 2] = stat.mean(score3)
    cm3 = confusion_matrix(y_test, pred3)
    # =============================================================================
    
    # =============================================================================
    # Classification Using Decision Tree
    classifier4 = DecisionTreeClassifier() 
    classifier4.fit(text_bow_train, y_train)
    pred4 = classifier4.predict(text_bow_test)
    pred_name4 = labelencoder.inverse_transform(pred4)
    score_table_train[int(n/10-1), 3] = classifier4.score(text_bow_train, y_train)
    score4 = cross_val_score(classifier4, bow_transformer.transform(X), y, cv=cv)
    score_table[int(n/10-1), 3] = stat.mean(score4)
    cm4 = confusion_matrix(y_test, pred4)
    # =============================================================================
    
    # =============================================================================
    # Classification Using Random Forests
    classifier5 = RandomForestClassifier(n_estimators=100)
    classifier5.fit(text_bow_train, y_train)
    pred5 = classifier5.predict(text_bow_test)
    pred_name5 = labelencoder.inverse_transform(pred5)
    score_table_train[int(n/10-1), 4] = classifier5.score(text_bow_train, y_train)
    score5 = cross_val_score(classifier5, bow_transformer.transform(X), y, cv=cv)
    score_table[int(n/10-1), 4] = stat.mean(score5)
    cm5 = confusion_matrix(y_test, pred5)
    # =============================================================================
    
    # =============================================================================
    # Printing Results
    print("Multinomial Naive Bayes\n Score for n = {}: {}\n Classification Report for n = {}:\n {}.".
          format(n, score1, n, classification_report(y_test, pred1)))
    print("K Nearest Neighbors (K-NN)\n Score for n = {}: {}\n Classification Report for n = {}:\n {}.".
          format(n, score2, n, classification_report(y_test, pred2)))
    print("Support Vector Machine (SVM)\n Score for n = {}: {}\n Classification Report for n = {}:\n {}.".
          format(n, score3, n, classification_report(y_test, pred3)))
    print("Decision Tree\n Score for n = {}: {}\n Classification Report for n = {}:\n {}.".
          format(n, score4, n, classification_report(y_test, pred4)))
    print("Random Forests\n Score for n = {}: {}\n Classification Report for n = {}:\n {}.".
          format(n, score5, n, classification_report(y_test, pred5)))
    # =============================================================================
        
    from print_top10 import print_top10
    print_top10(bow_transformer, classifier1, labelencoder.classes_)

new_X = input("Enter the text from the book: ")
new_X = pd.Series(new_X)
new_X = pd.Series(text_filter(new_X))
text_bow_new_X = bow_transformer.transform(new_X)
new_pred = labelencoder.inverse_transform(classifier1.predict(text_bow_new_X))
print("The Author is {}".format(str(new_pred)))

classifiers = ["Multinomial Naive Bayes", "K Nearest Neighbors", 
               "Support Vector Machine (SVM)", "Decision Tree",
               "Random Forests"] 
for i in range(0, 5):
    plt.plot()
    plt.plot(range(10, 200, 10), score_table[0:19, i], label="Test")
    plt.plot(range(10, 200, 10), score_table_train[0:19, i], label="Train")    
    plt.xlabel("Number of words in each document")
    plt.ylabel("Accuracy")
    plt.title("{} - Bag of Words Error Analysis".format(classifiers[i]))
    plt.legend() 
    plt.show()
    plt.savefig("{} - Bag of Words Error Analysis.png".format(classifiers[i]))
    plt.clf()
np.save("BoW.csv", score_table)
np.savetxt("BoW.csv", score_table, delimiter=",")
np.savetxt("BoW - Train.csv", score_table_train, delimiter=",")