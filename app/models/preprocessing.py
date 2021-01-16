#!/usr/bin/env python
# coding: utf-8

# In[1]:

import re
abbr_dict={
    "what's":"what is",
    "what're":"what are",
    "who's":"who is",
    "who're":"who are",
    "where's":"where is",
    "where're":"where are",
    "when's":"when is",
    "when're":"when are",
    "how's":"how is",
    "how're":"how are",

    "i'm":"i am",
    "we're":"we are",
    "you're":"you are",
    "they're":"they are",
    "it's":"it is",
    "he's":"he is",
    "she's":"she is",
    "that's":"that is",
    "there's":"there is",
    "there're":"there are",

    "i've":"i have",
    "we've":"we have",
    "you've":"you have",
    "they've":"they have",
    "who've":"who have",
    "would've":"would have",
    "not've":"not have",

    "i'll":"i will",
    "we'll":"we will",
    "you'll":"you will",
    "he'll":"he will",
    "she'll":"she will",
    "it'll":"it will",
    "they'll":"they will",

    "isn't":"is not",
    "wasn't":"was not",
    "aren't":"are not",
    "weren't":"were not",
    "can't":"can not",
    "couldn't":"could not",
    "don't":"do not",
    "didn't":"did not",
    "shouldn't":"should not",
    "wouldn't":"would not",
    "doesn't":"does not",
    "haven't":"have not",
    "hasn't":"has not",
    "hadn't":"had not",
    "won't":"will not",
}
#https://www.kaggle.com/life2short/data-processing-replace-abbreviation-of-word


# In[ ]:


spell_dict={
    "^(whats)":"what is",
    "^(whatre)":"what are",
    "^(whos)":"who is",
    "^(whore)":"who are",
    "^(wheres)":"where is",
    "^(wherere)":"where are",
    "^(whens)":"when is",
    "^(whenre)":"when are",
    "^(hows)":"how is",
    "^(howre)":"how are",

    "^(im)":"i am",
    "^(were)":"we are",
    "^(youre)":"you are",
    "^(theyre)":"they are",
    "^(its)":"it is",
    "^(hes)":"he is",
    "^(shes)":"she is",
    "^(thats)":"that is",
    "^(theres)":"there is",

    "^(ive)":"i have",
    "^(weve)":"we have",
    "^(youve)":"you have",
    "^(theyve)":"they have",
    "^(whove)":"who have",
    "^(wouldve)":"would have",
    "^(notve)":"not have",

    "^(isnt)":"is not",
    "^(wasnt)":"was not",
    "^(arent)":"are not",
    "^(werent)":"were not",
    "^(cant)":"can not",
    "^(couldnt)":"could not",
    "^(dont)":"do not",
    "^(didnt)":"did not",
    "^(shouldnt)":"should not",
    "^(wouldnt)":"would not",
    "^(doesnt)":"does not",
    "^(havent)":"have not",
    "^(hasnt)":"has not",
    "^(hadnt)":"had not",
    "^(wont)":"will not",
}

def remove_url(string):
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', str(string))


# In[ ]:


def remove_especial_char(string):
    return re.sub(r':.*?:|".*?"|\([^()]*\)|[\r\n\s\t]|&gt|“.*?”|\[.*?\]+|~.*?~|\.+|―+|-+|\(+|\)+|\*.*?\*',' ', string)


# In[ ]:


def remove_slash(string):
    return re.sub('/./|/', ' ', string)


# In[ ]:


def change_quotes(string):
    return re.sub('’', "'", string)


# In[ ]:


def replace_abbreviation(string):
    for key, val in list(abbr_dict.items()):
        string = re.sub(key, val, string)
    return string


# In[ ]:


def change_abbreviation(string):
    for key, val in list(spell_dict.items()):
        string = re.sub(key, val, string)
    return string


# In[ ]:


def remove_punctuation(string):
    return re.sub(r'["\'\?,;:¡\$\%!\[\]#\^\.\°\&\/\¨\{\}\@\\]', ' ', string)


# In[ ]:


def remove_all_except_letter(string):
    return re.sub('[^a-z ]','',string)


# In[ ]:


def remove_single_character(string):
    return re.sub('(^| )[^i|a](( ).)*( |$)', ' ', string)


# In[ ]:
def remove_strip(string):
    return string.strip()


# In[ ]:


def replace_multiple_space(string):
    return re.sub(r'\s+', ' ', string)


def process_text(string):
    string = remove_url(string)
    string = remove_especial_char(string)
    string = remove_slash(string)
    string = change_quotes(string)
    string = replace_abbreviation(string)
    string = change_abbreviation(string)
    string = remove_punctuation(string)
    string = remove_all_except_letter(string)
    string = remove_single_character(string)
    string = replace_multiple_space(string)
    string = remove_strip(string)
    return string