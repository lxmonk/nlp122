from __future__ import division
from collections import defaultdict, Counter
import nltk
import pylab

def PlotNumberOfTags(corpus):
    word_tag_dict = defaultdict(set)

    for (word, tag) in corpus:
        word_tag_dict[word].add(tag)

    C = Counter(len(val) for val in word_tag_dict.itervalues())

    pylab.subplot(211)
    pylab.plot(C.keys(), C.values(), '-go', label='Linear Scale')
    pylab.suptitle('Word Ambiguity:')
    pylab.title('Number of Words by Possible Tag Number')
    pylab.box('off')                 # for better appearance
    pylab.grid('on')                 # for better appearance
    pylab.ylabel('Words With This Number of Tags (Linear)')
    pylab.legend(loc=0)

    pylab.subplot(212)
    pylab.plot(C.keys(), C.values(), '-bo', label='Logarithmic Scale')
    pylab.yscale('log') # to make the graph more readable, for the log graph version
    pylab.box('off')                 # for better appearance
    pylab.grid('on')                 # for better appearance
    pylab.xlabel('Number of Tags per Word')
    pylab.ylabel('Words With This Number of Tags (Log)')
    pylab.legend(loc=0)
    
    pylab.show()

def MostAmbiguousWords(corpus, N):
    word_tag_dict = defaultdict(set)

    for (word, tag) in corpus:
        word_tag_dict[word].add(tag)
        
    filtered_tagged_words = [(word, tag) for (word, tag) in corpus if len(word_tag_dict[word]) > N]
    return nltk.ConditionalFreqDist(filtered_tagged_words)

def TestMostAmbiguousWords(cfd, N):
    all_good = True
    for word in cfd.conditions():
        all_good &= (len(cfd[word]) > N)

    if all_good:
        print 'All words occur with more than {} tags.'.format(N)
    else:
        print 'ERROR: Some words occur with less (or exactly) {} tags'.format(N)

def ShowExamples(word, cfd, corpus):
    for tag in cfd[word].keys():
        print '\'{}\' as {}: {}\n'.format(word, tag, example(word, tag, corpus))


def example(word, tag, corpus):
    idx = corpus.index((word, tag))
    sent = corpus[idx-10:idx] + [(word.upper(), tag)] + corpus[idx+1:idx+11]
    return ' '.join(word for (word, tag) in sent)

def correl_plot3D(corpus):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    word_tag_dict = defaultdict(set)
    for (word, tag) in corpus:
        word_tag_dict[word].add(tag)
        
    raw_wordlist = [word for (word, tag) in corpus]
    wordset = set(raw_wordlist)
    wordlist = list(wordset)
    word_fd = nltk.FreqDist(raw_wordlist)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([len(w) for w in wordlist], [word_fd[w] for w in wordlist],
               [len(word_tag_dict[w]) for w in wordlist])
    ax.set_xlabel('word length (charachters)')
    ax.set_ylabel('word frequency')
    ax.yaxis.set_scale('log')
    ax.set_zlabel('word ambiguity')

    plt.show()
    

    
    
if __name__ == '__main__':
    pass

    