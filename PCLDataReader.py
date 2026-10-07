##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

from abc import ABC, abstractmethod
from itertools import islice
from collections import Counter
from html import unescape
from lxml import etree
from sklearn.feature_extraction import DictVectorizer
import sys

#####################################################################
# HELPER FUNCTIONS
#####################################################################

def do_xml_parse(fp, tag, max_elements=None, progress_message=None):
    """ 
    Parses cleaned up spacy-processed XML files and gives us one XML element of the given tag up to max elements at a time  
    """
    fp.seek(0) #start at the beginning of the file

    elements = enumerate(islice(etree.iterparse(fp, tag=tag), max_elements)) #islice returns selected elements without storing the whole thing in memory
    for i, (event, elem) in elements:
        yield elem #returns one element to the called and then pauses until the next element is requested
        elem.clear() #this removes the previous elements contents
        if progress_message and (i % 1000 == 0): 
            print(progress_message.format(i), file=sys.stderr, end='\r')
    if progress_message: print(file=sys.stderr)

def short_xml_parse(fp, tag, max_elements=None): 
    """ 
    Parses cleaned up spacy-processed XML files (but not very well)

    read the whole xml and return a list of the matching elements  
    """

    #this one does not start from the beginning like the function above 
    elements = etree.parse(fp).findall(tag) #loads the whole tree into memory and then finds all the elements with the given tag
    N = max_elements if max_elements is not None else len(elements)
    return elements[:N]

#####################################################################
# PCLVocab
#####################################################################

class PCLVocab(): 
    def __init__(self, vocab_file, vocab_size, num_stop_words): 
        """
        skips the first num_stop_words and keep up to vocab_size words

        """
        start_index = 0 if num_stop_words is None else num_stop_words #stripping the num_stop_words - placing the index there
        end_index = start_index + vocab_size if vocab_size is not None else None #just up to vocab_size

        self._words = [w.strip() for w in islice(vocab_file, start_index, end_index)] # creates a list of words from the vocab_file, stripping whitespace and taking only the specified range
        #using islice to just get the selected lines
        self._dict = dict([(w, i) for (i, w) in enumerate(self._words)]) # keep a dictionary of word to index 

    def __len__(self): 
        """
        simply retuns the number of distinct words in the vocabulary, which is the length of the dictionary
        """
        return len(self._dict)

    def index_to_label(self, i): 
        """
        returns the word at index i in the vocabulary  
        """
        return self._words[i]

    def __getitem__(self, key):
        """
        if the word is in the vocabulary, return the index of the word, otherwise return None
        """
        if key in self._dict: return self._dict[key]
        else: return None

#####################################################################
# PCLLabels
#####################################################################

class PCLLabels(ABC):
    def __init__(self):
        """
        initializing things
        """ 
        self.labels = None
        self._label_list = None

    def __getitem__(self, index):
        """ return the label at this index """
        return self._label_list[index]

    def process(self, label_file, max_instances=None):

        """
        converts the labels to numerical values (the index of the list of possible labels)
        """
        y_labeled = list(map(self._extract_label, do_xml_parse(label_file, 'example', max_elements=max_instances))) #creates a list of labels by extracting the label from each example in the XML file
        if self.labels is None:
            self._label_list = sorted(set(y_labeled)) #cleans up the list of labels and sorts them
            self.labels = dict([(x,i) for (i,x) in enumerate(self._label_list)]) #creates a dictionary mapping each key (label) to its value (index)
            
        y = [self.labels[x] for x in y_labeled] #replacing labels with their numerical value in y_labeled which has the examples labels in order
        return y

    @abstractmethod  #this marks that a method must be implemented in a subclass      
    def _extract_label(self, example):
        """ Return the label for this instance...a subclass implements this method """
        return "Unknown"

#####################################################################
# PCLFeatures
#####################################################################

class PCLFeatures(ABC): 
    def __init__(self, vocab):
        self.initial_vocab = vocab #saving the vocab for later use
        self.vectorizer = DictVectorizer(sparse=True) #initializing a DictVectorizer which will convert feature dictionaries to a sparse matrix representation

    def extract_text(self, example):
        """
        extracting the text from the XML example, unescaping HTML entities, converting to lowercase, and splitting into words
        """
        return unescape("".join([x for x in example.itertext()]).lower()).split()

    def process(self, data_file, max_instances=None):
        """
        makes a matrix of features where each row is an example and each column is a feature 
        """
        if max_instances == None:
            N = len([1 for example in do_xml_parse(data_file, 'example')])
        else:
            N = max_instances
        
        ids = []
        feature_counters = []
        for example in do_xml_parse(data_file, 'example', max_elements=N, progress_message="Example {}"):
            ids.append(example.get("id"))
            features = self._extract_features(example)
            feature_counters.append(Counter(features))
        X = self.vectorizer.fit_transform(feature_counters)
        return X, ids #returns both the matrix and the ids of the examples in the same order as the rows of the matrix

    @abstractmethod
    def _get_feature_name(self, i):
        """ Returns a human-readable name for the ith feature in the DictVectorizer's internal vocabulary. need to be implemented in a subclass """
        return "Unknown"

    @abstractmethod            
    def _extract_features(self, example):
        """ Returns a list of the features in the example. need to be implemented in a subclass """
        return []

    @abstractmethod        
    def _get_num_features(self):
        """ Return the total number of features. need to be implemented in a subclass """
        return -1

#####################################################################