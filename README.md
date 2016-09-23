# `igt-detect`

This package is used to train a classifier to identify IGT instances in PDF-to-Text extracted text, as well as run that classifier on the resulting output.

The input for training and testing can either be:

* The output of [PDFLib TET](https://www.pdflib.com/products/tet/) (Commercial) or [PDFMiner](https://pypi.python.org/pypi/pdfminer/) (Free) that has been reanalyzed by the [Freki](https://github.com/xigt/freki) package.
* Raw text as extracted by a PDF-to-Text conversion package.

## 0. Dependencies and Setup


#### Dependencies
This package requires the [**MALLET**](http://mallet.cs.umass.edu/index.php) toolkit for its implementation of a maximum entropy classifier.

The script is written in Python 3, and requires a current Python 3 interpreter to be installed.

#### Setup

The package comes with an included `config.py` file.

This file must be edited to point toward the installation directory for **MALLET**.

The config file can also be used to configure a number of settings for the classifier, including feature selection and label set to use.

## 1. Training

### Usage

The training mode has the following usage:

	usage: igtdetect.py train [-h] [--type {freki,text}] [-f] -o OUT
	                          files [files ...]
	                          
* `--type` selects which input format to use (`freki` is the default)
* `-o` specifies the output location to save the classifier.  
* `-f` is a flag that will force overwriting the extracted features. If not specified, extracted features will be re-used. This option should be specified if the selected features have been changed in `config.py`.
* `files`, finally, is any amount of input files.

### Example

To train a classifier based on the `freki` file `sample_train.txt`, the following commandline can be used to output a sample classifier `sample_classifier.model`.

    ./igtdetect.py train -f sample/sample_train.txt -o sample/sample_classifier.model
    
## 2. Testing

### Usage

The testing mode has the following usage:

	usage: igtdetect.py test [-h] [--type {freki,text}] [-f] --classifier
	                         CLASSIFIER
	                         files [files ...]

* `--classifier` specifies the path to a previously saved classifier.
* `--type` again selects the input file type. Though a classifier trained on one file type will work on the other, certain features are only available to `freki` files, and thus will not be processed for `text` files, so classifiers should ideally be used with the same filetypes.
* `-f` forces overwriting of saved feature files.
* `--no-eval` is a flag that will prevent test files from being treated as evaluation files. By default, the script expects labels to exist in the target file for comparison.
* `-o` specifies the directory into which the files containing the newly added labels will be saved.

### Example

To test the saved classifier on a new document, the following commandline could be used:

    ./igtdetect.py test --classifier sample/sample_classifier.model sample/sample_test.txt
    
## 3. Evaluation

The evaluation mode requires a set of gold standard `freki` files placed in a folder defined in `config.py`. These gold files should have the same base name as the output `*_classified.txt` files to be evaluated, without the `_classified` suffix.

Although the `gold` files are specified in the config, the evaluation targets are specified as an argument.

The evaluation mode has the following usage:

	usage: igtdetect.py eval [-h] [-o OUTPUT] [--csv CSV] files
	igtdetect.py eval: error: the following arguments are required: files

* `-o` Specifies 
* `--csv` Output the evaluation data in CSV format
* `files` The files to evaluate, wildcards accepted.

### Example

    ./igtdetect.py eval output/classified/sample_test_classified.txt    
    
    