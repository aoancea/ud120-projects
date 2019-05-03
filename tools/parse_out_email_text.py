#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string
from sklearn.feature_extraction import stop_words


def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        # words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        stemmer = SnowballStemmer('english')
        
        # print text_string
        # print text_string.split(' ')

        # print text_string.replace('\n', ' ').replace('\t', ' ')
        # print text_string.replace('\n', ' ').replace('\t', ' ').split(' ')

        # text_string = [stemmer.stem(w.replace('\n', ' ').replace('\t', '')) for w in text_string.split(' ') if w != '']

        text_string = [stemmer.stem(w) for w in text_string.replace('\n', ' ').replace('\t', ' ').split(' ') if w != '']
        
        # print text_string

        # text_string = [w for w in text_string if w not in stop_words.ENGLISH_STOP_WORDS]

        words = ' '.join(text_string)
        # print words

    return words

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()

