# HW2 - TF-IDF vector construction

## Requirement
Write a program to convert a set of documents into tf-idf vectors.
- Construct a dictionary based on the terms extracted from the given documents.
- Transfer each document into a tf-idf unit vector.
- Write a function cosine($Doc_{x}$, $Doc_{y}$) which loads the tf-idf vectors of documents x and y and returns their cosine similarity.

For detailed instruction, please refer to `PA-2.pptx`
For detailed explanation of my algorithm, please refer to `report.pdf`.


## Environment
- Python: 3.6.8
- Packages: nltk, numpy

## Reproduce
```bash
python3 preprocess.py
python3 pa2.py
```