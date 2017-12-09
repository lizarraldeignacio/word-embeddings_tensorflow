# word-embeddings_tensorflow

Word embeddings implementation using tensorflow.

## usage

`python main.py DATASET_FOR_TRAINING --out OUTPUT_FILE`

### optional arguments
--batch: batch size  
--esize: dimension of the embedding vector  
--skips: how many words to consider left and right  
--nskips: how many times to reuse an input to generate a label  
--sampled: number of negative examples to sample  
--epochs: number of iterations  

## Training data
The quality of the word vectors increases significantly with amount of the training data. For research purposes, you can consider using data sets that are available on-line:

  * [First billion characters from wikipedia](http://mattmahoney.net/dc/enwik9.zip) (use the pre-processing perl script from the bottom of [Matt Mahoney's page](http://mattmahoney.net/dc/textdata.html)). 
  * [Latest Wikipedia dump](http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2): use the same script as above to obtain clean text. Should be more than 3 billion words. 
  * [WMT11 site](http://www.statmt.org/wmt11/translation-task.html#download): text data for several languages (duplicate sentences should be removed before training the models). 
  * [Dataset from "One Billion Word Language Modeling Benchmark"](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz): almost 1B words, already pre-processed text. 
  * [UMBC webbase corpus](http://ebiquity.umbc.edu/redirect/to/resource/id/351/UMBC-webbase-corpus): around 3 billion words, more info [here](http://ebiquity.umbc.edu/blogger/2013/05/01/umbc-webbase-corpus-of-3b-english-words/). Needs further processing (mainly tokenization). 
  * Text data from more languages can be obtained at [statmt.org](http://statmt.org/) and in the [Polyglot project](https://sites.google.com/site/rmyeid/projects/polyglot#TOC-Download-Wikipedia-Text-Dumps). 
