from cPickle import dump
import justext
import nltk
import requests
from google import search

def main():
    pages_used = 0
    for url in search('aeron chair', stop=30):
        text = ""
        url_used = False
        html = requests.get(url).content
        paragraphs = justext.justext(html, justext.get_stoplist('English'))
        for paragraph in paragraphs:
            if paragraph['class'] == 'good':
                if not url_used:
                    url_used = True
                    pages_used += 1
                text += (paragraph['text'] + '\n')
        if url_used:
            print 'now analyzing text from: {}'.format(url)
            rawfile = 'rawfile_{}.txt'.format(pages_used) 
            with file(rawfile, 'wb') as f:
                f.write(text)                              # save raw text to file
                
            sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            sents = sent_tokenizer.tokenize(text)
            sentsfile = 'sentsfile_{}.pkl'.format(pages_used)
            dump(sents, file(sentsfile, 'wb'), protocol=2) # pickle the sentences to a file
            
            tokenized_sents = []
            tokenized_sents += [nltk.word_tokenize(sent) for sent in sents]
            tokfile = 'tokfile_{}.pkl'.format(pages_used)
            dump(tokenized_sents, file(tokfile, 'wb'), protocol=2)  # pickle the tokenized_sents to a file
                        
            tagged_sents = [_tag(sent) for sent in tokenized_sents]
            tagged_text = ""
            for sent in tagged_sents:
                tagged_text += ('\n\n\t' + # sentence seperator
                                # create whitespace-separated 'word/tag' sentences
                                ' '.join(['/'.join(word_tag_tuple) for word_tag_tuple in sent]) +
                                '\n') # newline after sentence
                
            corpfile = 'corpfile_{}.txt'.format(pages_used)
            with file(corpfile, 'wb') as f:
                f.write(tagged_text)
                
    return 0

def _tag(sent):
    """
    This is taken from http://goo.gl/TxTyq (short for
    stackoverflow.com/...) with minor changes.
    This function returns the inputed 'sent' as tagged by nltk.pos_tag
    converted to Brown simplified tags.
    """
    from nltk.tag.simplify import simplify_brown_tag
    tagged_sent = nltk.pos_tag(sent) 
    simplified = [(word, simplify_brown_tag(tag)) for word, tag in tagged_sent]
    return simplified

if __name__ == '__main__':
    main()
