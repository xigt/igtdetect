# `igt-detect`

This package is used to train and run a classifier to identify IGT instances in PDF-extracted text.

The input for training can be produced by:

* Using one of the two packages to extract text from PDF:
	*  [PDFLib TET](https://www.pdflib.com/products/tet/) (Commercial)
	*  [PDFMiner](https://pypi.python.org/pypi/pdfminer/) (Free) 
*  Reanalyzing the output using our [Freki](https://github.com/xigt/freki) package.

## 0. Dependencies and Setup


### Dependencies

* Python Modules:
	* [scikit-learn](http://scikit-learn.org/)
	* [freki](https://github.com/xigt/freki)
	* [numpy](http://www.numpy.org/)

*N.B.: If for some reason you are unable to install modules into your python installation, you can use the `pythonpath` setting in the config files to add folders that contain the appropriate modules*

### Setup

1. Copy the included `defaults.ini.sample` file as `defaults.ini`
2. Most settings will be fine at their default values, but users may want to adjust settings in the `[paths]` section.
3. More detailed descriptions can be found at the end of the readme.

### Using Config Files

The `defaults.ini` is used to set the defaults for the system, but any value may be overwritten by supplying a config file based on this template using the `-c / --config` option at runtime.

Any value not specified in the config file will use the default value supplied in `defaults.ini`.


## 1. Training

### Usage

The training mode has the following usage:
	
	usage: igtdetect.py train [-h] [-v] [-c CONFIG] [-f]
	                          [--classifier-path CLASSIFIER_PATH]
	                          [--use-bi-labels USE_BI_LABELS]
	                          [--max-features MAX_FEATURES]
	                          [--train-files TRAIN_FILES] [--overwrite-model]
	
	optional arguments:
	  -h, --help                         show this help message and exit
	  -v, --verbose                      Enable verbosity.
	  -c CONFIG, --config CONFIG         Alternate config file.
	  -f, --overwrite-features           Overwrite previously generated feature files.
	  --classifier-path CLASSIFIER_PATH  Path to the saved classifier model.
	  --use-bi-labels USE_BI_LABELS
	  --max-features MAX_FEATURES
	  --train-files TRAIN_FILES          Path to the files for training the classifier.
	  --overwrite-model                  Overwrite previously created models
	                     	                     
	                          
#### Output:
* If `debug_on` is set to `1` or `true`:
	* A file will be created inside a folder in the `debug` folder with the breakdown of feature weights converged upon for the classifier
		* The file will use the same base name as the output classifier
		* With the suffix `_feat_weights.txt`

### Example

To train a classifier using all of the files in the directory `train` and save to the output file `sample_classifier.model`.

    ./igtdetect.py train --train-files "train/*.txt" --classifier-path "sample_classifier.model"
    
Alternatively, if the config file `my_config.ini` contains the lines:

	[paths]
    train_files = train/*.txt
    classifier_path = sample_classifier.model
    
Then the same process could be run using the command:

    ./igtdetect.py train -c "my_config.ini"
    
## 2. Testing

### Usage

The testing mode has the following usage:

	usage: igtdetect.py test [-h] [-v] [-c CONFIG] [-f]
	                         [--classifier-path CLASSIFIER_PATH]
	                         [--test-files TEST_FILES]
	                         [--classified-dir CLASSIFIED_DIR]
	
	optional arguments:
	  -h, --help                          show this help message and exit
	  -v, --verbose                       Enable verbosity.
	  -c CONFIG, --config CONFIG          Alternate config file.
	  -f, --overwrite-features            Overwrite previously generated feature files.
	  --classifier-path CLASSIFIER_PATH   Path to the saved classifier model.
	  --test-files TEST_FILES             Path to the files to be classified.
	  --classified-dir CLASSIFIED_DIR     Directory to output the classified documents.

#### Output:
* Files in [freki](https://github.com/xigt/freki) format will be written to the `classified_dir` directory in the config file, sharing the same filename as their input files, with the suffix `_classified.txt`.
	* the `--classified-dir` can override this option on the commandline.
* If the file being tested has not had its features extracted previously, or the `-f` flag is set:
	* a file will be written containing the svm-lite formatted features for each line
		* in the directory specified by `feat_dir`
		* using the suffix `_feats.txt`
* If `debug_on` is set to `1` or `true`:
	* a line-by-line summary of the classification probabilities will be written for each file
		* in the directory specified by `debug_dir`
		* with the suffix `_classifications.txt`

### Example

To test the saved classifier `sample_classifier.model` on all documents within the `test` directory, and output the classified files to the directory `classified` the following commandline could be used:

    ./igtdetect.py test --classifier-path "sample_classifier.model" \
                        --test-files "./test/*.txt" \
                        --classified-dir "./classified"
                        
Alternatively, if the config file `myconfig.ini` contains the lines:

    [paths]
    classifier_path = sample_classifier.model
    test_files = ./test/*.txt
    classified_dir = ./classified
    
Then the following command could be used to perform the same thing:

    ./igtdetect.py test -c myconfig.ini
    
## 3. Evaluation

The evaluation mode requires a set of gold standard `freki` files placed in a directory. These gold files should have the same base name as the output (`*_classified.txt`) files to be evaluated, without the `_classified` suffix.

The directory containing the gold files must either be specified in the config file or on the commandline.

The evaluation mode has the following usage:

	usage: igtdetect.py eval [-h] [-v] [-c CONFIG] [-f] [-o OUT_PATH] [--csv CSV]
	                         [--eval-files EVAL_FILES] [--gold-dir GOLD_DIR]
	
	optional arguments:
	  -h, --help                        show this help message and exit
	  -v, --verbose                     Enable verbosity.
	  -c CONFIG, --config CONFIG        Alternate config file.
	  -f, --overwrite-features          Overwrite previously generated feature files.
	  -o OUT_PATH, --output OUT_PATH    Output path to write evaluation result. [Default: stdout]
	  --csv CSV                         Format the output as CSV.
	  --eval-files EVAL_FILES           Files to evaluate against
	  --gold-dir GOLD_DIR


### Output

The output of the evaluation mode looks like the following:

		 COLS: Gold --- ROWS: Predicted
		O   	L   	G   	T   	M
	O	1392	  23	  12	   9	  18	0.96
	L	   2	 207	   2	   0	   0	0.98
	G	   4	   2	 196	   0	   0	0.97
	T	   4	   5	   6	 202	   0	0.93
	M	   4	   0	   0	   0	  18	0.82
		0.99	0.87	0.91	0.96	0.50
	
	----- Labels -----
	 Classifiation Acc: 0.96
	       Non-O P/R/F: 0.96	0.89	0.92
	
	----- Spans ------
	  Exact-span P/R/F: 0.22	0.11	0.14
	Partial-span P/R/F: 0.92	0.91	0.92
	
	--- Auto-Spans ---
	  Exact-span P/R/F: 0.61	0.63	0.62
	Partial-span P/R/F: 0.97	0.92	0.95

#### Confusion Matrix
The first section of the output is a confusion matrix comparing the gold standard labels (across the top) with those assigned by the classifier (the labels on the side), as well as precision (rightmost column) and recall (bottom row) for each label.

#### Labels
The **Labels** section gives overall accuracy, as well as precision, recall, and F<sub>1</sub>-score (P/R/F) per-line for **"Non-O"** lines.


#### Spans
The **Spans** section gives P/R/F scores for entire spans of IGT, as defined by unique "span-id" attributes for the lines in the document.

* **Exact-spans** are calculated as spans for which the system gets the line boundaries for the IGT exactly the same as the gold standard. 
* **Partial-spans** are calculated by how many spans the system generates that overlap with a gold span in some way.

#### Auto-Spans

The **"Auto-Spans"** measurements are calculated the same as the **Spans** measurements, with the exception that, rather than using the `span_id` attribute from the lines in the files, spans in both files are automatically generated as any contiguous block of *non-O* lines.

### Example

To evaluate a set of classified files in the directory `classified` against any gold files in the `gold` directory that match the name of the classified files, minus the `*_classified` suffix, the following commandline could be used:

    ./igtdetect.py eval --eval-files "./classified/*.txt"  --gold-dir "./gold"
    
Alternatively, if the config file `myconfig.ini` contains the lines:

    [paths]
    gold_dir = ./gold
    eval_files = ./classified/*.txt
    
Then the following commandline would also achieve the same result:

    ./igtdetect.py eval -c myconfig.ini
    
    
# Features

There are two sets of features used by the classifier; those that have some position and font information, as provided by PDFLib TET's extraction process and reanalyzed by the freki package, and those that are available purely from the unicode text strings on each line.

The following sections provide a list of each of these features, and a brief description.



## Text-based Features
* `words`
	* For each word `(\w+)` in the line, a boolean feature per word.
* `has_langname`
	* Does the line contain a word that also appears in the list of languages in the `lng_names` path in the config file.
* `has_grams`
	* Return `true` if the line contains one of the grams defined in the `gram_list` or `gram_list_cased` files specified in the config file. (Case insensitive and sensitive lists, respectively).
* `has_parenthetical`
	* Return `true` if the line contains some portion wrapped in parenthesis or brackets.
* `has_citation`
	* Return `true` if the line contains something citation-like, like (Author, 1934).
* `has_asterisk`
	* Return `true` if the line contains an asterisk.
* `has_bracketing`
	* Return `true` if there are bracketed (`[ ]`) portions of the line.
* `has_underscore`
	* Return `true` if there is an underscore on the line.
* `has_quotation`
	* Return `true` if some portion of the line is wrapped in single or double quotes.
* `has_numbering`
	* Return `true` if the line contains some form of numbering, such as `1.` or `c)`
* `has_leading_whitespace`
	* Return `true` if the line begins with whitespace.
* `high_oov_rate`
	* Return `true` if the ratio of out-of-vocabulary (OOV) words to total length of the sentence exceeds the `high_oov` setting in the config file, using the English wordlist specified by `en_wordlist`.
* `med_oov_rate`
	* Return `true` if the OOV rate (see above) exceeds the `med_oov` setting in the config file.
* `high_gls_oov`
	* Same as `high_oov_rate`, but uses the `gls_wordlist` file to define the vocabulary.
* `med_gls_oov`
	* Same as `med_oov_rate`, but uses the `gls_wordlist` file to define the vocabulary.
* `has_jpn`
	* Return `true` if this line contains unicode characters in the Japanese character ranges.
* `has_grk`
	* Return `true` if this line contains unicode characters in the Greek character range.
* `has_kor`
	* Return `true` if this line contains unicode characters in the Korean character range.
* `has_cyr`
	* Return `true` if this line contains unicode characters in the Cyrillic character range.
* `has_acc_lat`
	* Return `true` if this line contains accented Latin unicode characters.
* `has_dia`
	* Return `true` if this line contains unicode diacritic characters.
* `has_uni`
	* Return `true` if this line contains **ANY** of the above unicode ranges.
* `has_year`
	* Return `true` if this line contains a valid four-digit date.

## Freki-Based Features

* `is_indented`
	* Returns `true` if the line's leftmost `x` coordinate is greater than the mode of leftmost `x` coordinates over the entire document.
* `is_first_page`
	* Returns `true` if the line is contained within the first page.
* `prev_line_same_block`
	* Returns `true` if the previous line is contained within the same "block" as the current line.
* `next_line_same_block`
	* Returns `true` if the next block is contained within the same "block" as the current line.
* `has_nonstandard_font`
	* Returns `true` if a font is contained on this line aside from the most commonly seen font in the document.
* `has_larger_font`
	* Returns `true` if a font is contained on this line that is larger than the most common font size in the document.
* `has_smaller_font`
	* Returns `true` if a font is contained on this line that is smaller than the most common font size in the document.

## Feature Selection
* `max_features`
	* If not given the value `-1`, the model will gather use the &Chi;<sup>2</sup> test to select the `max_features` number of features to train the model with, saving disk space to store the model, and some small amount of performance at test time.
* `freki_feats_enabled`
	* If set to `1`, use the `freki`-based features.
* `text_feats_enabled`
	* If set to `1`, use the textual features. 
* `use_prev_line`
	* If the current line is *n*, in addition to using every enabled feature on the line *n*, also calculate the features for line *n-1* as features on line *n*.
* `use_prev_prev_line`
	* Same as above, but use the *n-2* line rather than *n-1*.
* `use_next_line`
	* Same as above, but use line *n+1*.