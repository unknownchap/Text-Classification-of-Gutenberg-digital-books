from urllib import request
import nltk 
import nltk.corpus  
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt






lemmatiser = WordNetLemmatizer()
stop_words=set(stopwords.words("english"))

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
# Word Cloud Visualization
wordcloud1 = WordCloud().generate(TheThreeMusketeers)
plt.imshow(wordcloud1, interpolation='bilinear')
plt.title("WordCloud of The Three Musketeers")
plt.show()
plt.savefig("WordCloud of The Three Musketeers.png")

wordcloud2 = WordCloud().generate(SherlockHolmes)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.title("WordCloud of Adventures of Sherlock Holmes")
plt.show()
plt.savefig("WordCloud of Adventures of Sherlock Holmes.png")

wordcloud3 = WordCloud().generate(Oz)
plt.imshow(wordcloud3, interpolation='bilinear')
plt.title("WordCloud of Dorothy and the Wizard in Oz")
plt.show()
plt.savefig("WordCloud of Dorothy and the Wizard in Oz.png")

wordcloud4 = WordCloud().generate(MysteriousIsland)
plt.imshow(wordcloud4, interpolation='bilinear')
plt.title("WordCloud of The Mysterious Island")
plt.show()
plt.savefig("WordCloud of The Mysterious Island.png")

wordcloud5 = WordCloud().generate(MechanicalDrawing)
plt.imshow(wordcloud5, interpolation='bilinear')
plt.title("WordCloud of Mechanical Drawing Self-Taught")
plt.show()
plt.savefig("WordCloud of Mechanical Drawing Self-Taught.png")

wordcloud6 = WordCloud().generate(ArtofPhotography)
plt.imshow(wordcloud6, interpolation='bilinear')
plt.title("WordCloud of The History and Practice of the Art of Photography")
plt.show()
plt.savefig("WordCloud of The History and Practice of the Art of Photography.png")

wordcloud7 = WordCloud().generate(ChristmasCarol)
plt.imshow(wordcloud7, interpolation='bilinear')
plt.title("WordCloud of ChristmasCarol, by Charles Dickens")
plt.show()
plt.savefig("WordCloud of ChristmasCarol, by Charles Dickens.png")

# =============================================================================