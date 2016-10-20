# `igt-detect`

This package is used to train a classifier to identify IGT instances in PDF-to-Text extracted text, as well as run that classifier on the resulting output.

The input for training and testing can either be:

* The output of [PDFLib TET](https://www.pdflib.com/products/tet/) (Commercial) or [PDFMiner](https://pypi.python.org/pypi/pdfminer/) (Free) that has been reanalyzed by the [Freki](https://github.com/xigt/freki) package.
* Raw text as extracted by a PDF-to-Text conversion package.

## 0. Dependencies and Setup


### Dependencies
This package requires the [**MALLET**](http://mallet.cs.umass.edu/index.php) toolkit for its implementation of a maximum entropy classifier.

The script is written in Python 3, and requires a current Python 3 interpreter to be installed.

### Setup

1. Copy the included `defaults.ini.sample` file as `defaults.ini`
2. Set the value of `mallet_dir` to point to the location of the mallet install on this machine.

### Using Config Files

The `defaults.ini` is used to set the defaults for the system, but any value may be overwritten by supplying a config file based on this template using the `-c / --config` option at runtime.

Any value not specified in the config file will use the default value supplied in `defaults.ini`.

The features used for training are explained more in-depth within the `defaults.ini.sample` file and in the included `FEATURES.md` file.

## 1. Training

### Usage

The training mode has the following usage:
	
	usage: igtdetect.py train [-h] [-f] -o OUT [-c CONFIG]
	                          files [files ...]
		
	positional arguments:
	  files
		
	optional arguments:
	  -h, --help                   show this help message and exit
	  -f, --overwrite              Overwrite text vectors. [Default: Do not overwrite]
	  -o OUT, --out OUT            Output path for the classifier.
	  -c CONFIG, --config CONFIG   Specifies a config file which may 
	                               override settings in defaults.ini.
	                          
#### Output:
* If `debug_on` is set to `1` or `true`:
	* A file will be created inside a folder in the `debug` folder with the breakdown of feature weights converged upon for the classifier
		* The file will use the same base name as the output classifier
		* With the suffix `_feat_weights.txt`

### Example

To train a classifier based on the `freki` file `sample_train.txt`, the following commandline can be used to output a sample classifier `sample_classifier.model`.

    ./igtdetect.py train -f sample/sample_train.txt -o sample/sample_classifier.model
    
## 2. Testing

### Usage

The testing mode has the following usage:

	usage: igtdetect.py test [-h] [--type {freki,text}] [-f] --classifier
	                         CLASSIFIER [-c CONFIG]
	                         files [files ...]
	
	positional arguments:
	  files                        Files to apply classifier against
	
	optional arguments:
	  -h, --help                   show this help message and exit
	  -f, --overwrite              Overwrite text vectors. [Default: do not overwrite]
	  --classifier CLASSIFIER      Specifies the path for the saved classifier to use.
	  -c CONFIG, --config CONFIG   Specifies a config file which may 
	                               override settings in defaults.ini.

#### Output:
* Files in [freki](https://github.com/xigt/freki) format will be written to the `out_path` directory in the config file, sharing the same filename as their input files, with the suffix `_classified.txt`
* If the file being tested has not had its features extracted previously, or the `-f` flag is set:
	* a file will be written containing the svm-lite formatted features for each line
		* in the directory specified by `feat_dir`
		* using the suffix `_feats.txt`
* If `debug_on` is set to `1` or `true`:
	* a line-by-line summary of the classification probabilities will be written for each file
		* in the directory specified by `debug_dir`
		* with the suffix `_classifications.txt`

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
    
    