# NLP

Implementation of various Natural Language processing algorithms including pre processing of text, topic modeling, word embeddings, sentiment classification.

Natural Language processing technology that is used by machines to understand, analyse, manipulate, and interpret human's languages. It helps developers to organize knowledge for performing tasks such as translation, automatic summarization, Named Entity Recognition (NER), speech recognition, relationship extraction, and topic segmentation.

Steps for Natural language processing:
Pre processing: Tokenization- Tokenization is breaking the raw text into small chunks called tokens.
                Stemming: Stemming is the process of reducing derived words to their word stem, base, or root form
                Lemmatization: Lemmatization is the process of reducing a group of words into their lemma or dictionary form. It takes into account things like POS(Parts                                of Speech), the meaning of the word in the sentence, the meaning of the word in the nearby sentences
                POS tagging: Associate parts of speech to the tokens and check grammar, arrangements of words, to understand the interrelationship between the words.
                Named Entity Recognition: Extracting information to classify named entities mentioned in unstructured text into pre-defined categories such as                                                     person names, organizations, locations, codes, time.
                https://github.com/SomdeepAcharyya/NLP/blob/main/NLP_Preprocessing.ipynb

Word Embedding: Word Embedding is a language modeling technique used for mapping words to vectors of real numbers. It represents words or phrases in vector space with                   several dimensions. 
    Word2Vec: Word2Vec creates vectors of the words that are distributed numerical representations of word features – these word features could comprise of words that               represent the context of the individual words present in our vocabulary. 
              https://github.com/SomdeepAcharyya/NLP/blob/main/Word2vec_Embedding.ipynb
    ELMO:     ELMo word vectors are computed on top of a two-layer bidirectional language model (biLM). This biLM model has two layers stacked together. Each layer has               2 passes — forward pass and backward pass. It uses a character-level convolutional neural network (CNN) to represent words of a text string into raw word               vectors.The information from the forward and backward pass, forms the intermediate word vectors These intermediate word vectors are fed into the next                   layer of biLM. The final representation (ELMo) is the weighted sum of the raw word vectors and the 2 intermediate word vectors.
              https://github.com/SomdeepAcharyya/NLP/blob/main/Word_Embeddings_Glove_and_Elmo.ipynb
    Glove:    GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-                   occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.
              https://github.com/SomdeepAcharyya/NLP/blob/main/Word_Embeddings_Glove_and_Elmo.ipynb
              
Topic modelling: In statistics and natural language processing, a topic model is a type of statistical model for discovering the abstract "topics" that occur in a collection of documents. Topic modeling is a frequently used text-mining tool for discovery of hidden semantic structures in a text body. Topic models are also referred to as probabilistic topic models, which refers to statistical algorithms for discovering the latent semantic structures of an extensive text body.
 
TF-IDF topic modeling: 
TF-IDF stands for term frequency-inverse document frequency and it is a measure, used in the fields of information retrieval (IR) and machine learning, that can quantify the importance or relevance of string representations. 

LDA Topic modelling
Latent Dirichlet Allocation (LDA) is one of the ways to implement Topic Modelling. It is a generative probabilistic model in which each document is assumed to be consisting of a different proportion of topics.
https://github.com/SomdeepAcharyya/NLP/blob/main/LDA_Topic_Modelling.ipynb

