from __future__ import division
from collections import defaultdict, Counter
from numpy import log
import nltk
import pylab


def PlotNumberOfTags(corpus):
    word_tag_dict = defaultdict(set)

    for (word, tag) in corpus:
        word_tag_dict[word].add(tag)
    # using Counter for efficiency (leaner than FreqDist)
    C = Counter(len(val) for val in word_tag_dict.itervalues())

    pylab.subplot(211)
    pylab.plot(C.keys(), C.values(), '-go', label='Linear Scale')
    pylab.suptitle('Word Ambiguity:')
    pylab.title('Number of Words by Possible Tag Number')
    pylab.box('off')                 # for better appearance
    pylab.grid('on')                 # for better appearance
    pylab.ylabel('Words With This Number of Tags (Linear)')
    pylab.legend(loc=0)
    # add value tags
    for x,y in zip(C.keys(), C.values()):
        pylab.annotate(str(y), (x,y + 0.5))

    pylab.subplot(212)
    pylab.plot(C.keys(), C.values(), '-bo', label='Logarithmic Scale')
    pylab.yscale('log') # to make the graph more readable, for the log graph version
    pylab.box('off')                 # for better appearance
    pylab.grid('on')                 # for better appearance
    pylab.xlabel('Number of Tags per Word')
    pylab.ylabel('Words With This Number of Tags (Log)')
    pylab.legend(loc=0)
    # add value tags
    for x,y in zip(C.keys(), C.values()):
        pylab.annotate(str(y), (x,y + 0.5))
        
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

    word_tag_dict = defaultdict(set)
    for (word, tag) in corpus:
        word_tag_dict[word].add(tag)
        
    raw_wordlist = [word for (word, tag) in corpus]
    wordset = set(raw_wordlist)
    wordlist = list(wordset)
    word_fd = nltk.FreqDist(raw_wordlist)
    
    fig = pylab.figure(figsize=(15,15))
    ax = fig.add_subplot(224, projection='3d') # 224
    xs = [len(w) for w in wordlist]
    ys = [word_fd[w] for w in wordlist]
    zs = [len(word_tag_dict[w]) for w in wordlist]
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('word length (charachters)')
    ax.set_ylabel('word frequency')
    ax.set_zlabel('word ambiguity')

    pylab.subplot(221)
    pylab.yscale('log')
    pylab.ylim(ymin=1, ymax=100000)
    pylab.scatter(xs, ys)
    pylab.title('word length - word freq (log)')
    pylab.xlabel('word length')
    pylab.ylabel('word freq (log)')
    
    pylab.subplot(222)
    pylab.xscale('log')
    pylab.xlim(xmin=1, xmax=100000)
    pylab.scatter(ys, zs)
    pylab.title('word freq (log) - word ambiguity')
    pylab.xlabel('word freq (log)')
    pylab.ylabel('word ambiguity')
    
    pylab.subplot(223)
    pylab.scatter(zs, xs)
    pylab.title('word ambiguity - word length')
    pylab.xlabel('word ambiguity')
    pylab.ylabel('word length')
    
    pylab.show()
    


from nltk.tag.api import TaggerI
import nltk

class MyUnigramTagger(TaggerI):
    def __init__(self, train=None, model=None,
                 backoff=None, cutoff=0, verbose=False):
        if type(train[0]) == tuple:
            pass
        elif type(train[0][0]) == tuple:
            train = flatten(train)            
        self.cfd = nltk.ConditionalFreqDist(train)
        self.default_tag = nltk.FreqDist(tag for (word, tag) in train).max()
        self.wordset = set(word for (word, tag) in train)
                        
    def tag(self, tokens):
        # docs inherited from TaggerI
        return zip(tokens, [self.cfd[word].max()
                            if word in self.wordset
                            else self.default_tag
                            for word in tokens])
            
# This really should have come with either itertools or standard lib..
from itertools import chain
def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)
    
    
from nltk.tag import AffixTagger

class MyAffixTagger(AffixTagger):
    def __init__(self, train=None, model=None, affix_length=-3,
                 min_stem_length=2, backoff=None, cutoff=0, verbose=False,
                 H_param=0):
        self.H_param = H_param
        AffixTagger.__init__(self, train, model, affix_length,
                             min_stem_length, backoff, cutoff, verbose)

                
    def _train(self, tagged_corpus, cutoff=0, verbose=False):
        """
        Initialize this ContextTagger's ``_context_to_tag`` table
        based on the given training data.  In particular, for each
        context ``c`` in the training data, set
        ``_context_to_tag[c]`` to the most frequent tag for that
        context.  However, exclude any contexts that are already
        tagged perfectly by the backoff tagger(s).

        The old value of ``self._context_to_tag`` (if any) is discarded.

        :param tagged_corpus: A tagged corpus.  Each item should be
            a list of (word, tag tuples.
        :param cutoff: If the most likely tag for a context occurs
            fewer than cutoff times, then exclude it from the
            context-to-tag table for the new tagger.
        """

        token_count = hit_count = 0

        # A context is considered 'useful' if it's not already tagged
        # perfectly by the backoff tagger.
        useful_contexts = set()

        # Count how many times each tag occurs in each context.
        fd = nltk.ConditionalFreqDist()
        for sentence in tagged_corpus:
            tokens, tags = zip(*sentence)
            for index, (token, tag) in enumerate(sentence):
                # Record the event.
                token_count += 1
                context = self.context(tokens, index, tags[:index])
                if context is None: continue
                fd[context].inc(tag)
                # If the backoff got it wrong, this context is useful:
                if (self.backoff is None or
                    tag != self.backoff.tag_one(tokens, index, tags[:index])):
                    useful_contexts.add(context)

        # Build the context_to_tag table -- for each context, figure
        # out what the most likely tag is.  Only include contexts that
        # we've seen at least `cutoff` times.
        ### best_preds = []
        for context in useful_contexts:
            best_tag = fd[context].max()
            hits = fd[context][best_tag]
            if hits > cutoff and self.H(fd[context]) > self.H_param:
                self._context_to_tag[context] = best_tag
                hit_count += hits
                
           ###      prediction = fd[context][best_tag] / fd[context].N()
        ###         if prediction > 0.8:
        ###             best_preds.append((prediction, context))
        ### for prediction, context in sorted(best_preds, reverse=True):
        ###     print '{:3} gives {:.3f} percent prediction as {}'.format(context,
        ###                                                               prediction * 100,
        ###                                                               fd[context].max())
            
    def H(self, p):
        # we already used 'from __future__ import division''
        N = p.N()               # total num of items seen by the fd
        sum = 0
        for k,v in p.iteritems():
            Px = v / N          # future division
            logPx = log(Px)
            PxXlogPx = 0 if Px == 0 else Px * logPx 
            sum -= PxXlogPx
        return sum


        
    def optimize_parameter(self):
        corpus = nltk.corpus.brown.tagged_sents(simplify_tags=True)
        corpus_len = len(corpus)
        trainset = corpus[:int(corpus_len * 8 / 10)] # this is ok, it's an int
        develset = corpus[int(corpus_len * 8 / 10):int(corpus_len * 9 / 10)]
        testset =  corpus[int(corpus_len * 9 / 10):]
        options = []            # list of (accuracy, param_val, context_to_tag_table) tuples
        for cutoff_candidate in range(1,5) + range(5,10,2) + range(10,51,10):
            print 'optimizing cutoff parameter: trying {:2}'.format(cutoff_candidate),
            self._context_to_tag.clear() # clear context_to_tag table
            self._train(trainset, cutoff=cutoff_candidate)
            score = self.evaluate(develset)
            print ' ---> it scored {:.5f}'.format(score)
            options.append((score, cutoff_candidate, self._context_to_tag))
            
        _, choosen_cutoff, self._context_to_tag = max(options)
        print ('choosen cutoff value is {}, which scores {} on the '
               'testset.'.format(choosen_cutoff,
                                 self.evaluate(testset)))
                                 
from numpy import linspace

def optimize_H_param():
    corp = nltk.corpus.brown.tagged_sents(simplify_tags=True)
    for h in linspace(0,1,10)[:-1]: # remove the 1.0
        print
        print 'optimize_h_param: H_param={:f}'.format(h)
        affix = MyAffixTagger(corp, H_param=h)
        affix.optimize_parameter()


if __name__ == '__main__':
    pass

    



