# Analysis of 10K Report 

# SEC Filing Text Extraction and Analysis

This project involves the extraction of text data from SEC (U.S. Securities and Exchange Commission) filings, particularly 10-K reports, for a given company ticker. The extracted text is then utilized for analysis and further processing.

## Overview

The provided scripts enable the following functionalities:

1. Retrieving SEC filing details such as accession numbers, filing dates, and report dates for a given company ticker.
2. Filtering 10-K reports from the obtained filings.
3. Extracting text data from the filtered 10-K reports.
4. Cleaning and preprocessing the extracted text for analysis.

## Dependencies

To run the scripts, ensure you have the following dependencies installed:

- `sec_api`: A Python client for the SEC API to retrieve filing data.
- `sec_edgar_downloader`: A Python package to download SEC filings.
- `requests`: A library for making HTTP requests.
- `pandas`: A powerful data manipulation library.
- `bs4` (Beautiful Soup): A library for web scraping.
- `re` (Regular Expression): A module for working with regular expressions.

You can install these dependencies using `pip`:



# Q&A using BERT
This document outlines the process of analyzing 10K reports using BERT (Bidirectional Encoder Representations from Transformers) models. The analysis involves fine-tuning BERT models for sequence classification and question-answering tasks.

## Code Overview
The provided Python script conducts the following tasks:
1. Preprocesses the text data.
2. Fine-tunes a BERT model for sequence classification on each 10K report.
3. Fine-tunes a BERT model for question answering.
4. Performs question answering on each 10K report.

## Code Implementation

```python
import os
from transformers import BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering
import torch

# Define functions for preprocessing, fine-tuning, and question answering
# (Code for preprocess_text, fine_tune_bert, fine_tune_bert_qa, and perform_qa_on_chunks functions)

# Fine-tune BERT models for classification on each file
# (Code for fine-tuning BERT models for classification)

# Fine-tune BERT model for question answering
# (Code for fine-tuning BERT model for question answering)

# Define your question and perform question answering on each file
# (Code for defining question and performing question answering)
```

## Usage

1. Set up your API key for the SEC API by registering on the SEC website.
2. Replace the placeholder API key in the script with your actual API key.
3. Run the script with the desired company ticker to extract and analyze the 10-K reports.

## Scripts

- `sec_filing_extraction.py`: Contains functions to extract filing data from the SEC API and filter 10-K reports.
- `sec_text_extraction.py`: Contains functions to extract text data from the filtered 10-K reports and clean the text.
- `main.py`: Main script to execute the entire process, from data extraction to text analysis.

## Outputs

- Text files: Extracted and cleaned text data from 10-K reports, saved in the specified directory.
- Dataframes: Processed dataframes containing filing details, accession numbers, and report dates.

## Additional Notes

- Ensure proper directory paths are set up for saving text files and other outputs.
- Check for any rate limits or restrictions on API requests to avoid errors during data retrieval.



# SEC Filing Text Topic Modeling

This project involves extracting text data from SEC (U.S. Securities and Exchange Commission) filings for companies like American Express (AXP), Mastercard (MA), and Visa (V). The extracted text is then used for topic modeling using Latent Dirichlet Allocation (LDA) and visualized using pyLDAvis.

## Overview

The provided scripts enable the following functionalities:

1. **Text Data Extraction**: Text data is extracted from SEC filings for the specified tickers and years.
2. **Text Preprocessing**: The extracted text is preprocessed, which includes tokenization, stop word removal, and lemmatization.
3. **Topic Modeling**: LDA (Latent Dirichlet Allocation) is applied to the preprocessed text data to identify topics within the documents.
4. **Visualization**: The topics identified by LDA are visualized using pyLDAvis to aid interpretation.

## Dependencies

Ensure you have the following dependencies installed:

- `pandas`
- `scikit-learn`
- `nltk`
- `wordcloud`
- `matplotlib`
- `seaborn`
- `gensim`
- `pyLDAvis`
- `corpora`
- `os`

You can install these dependencies using `pip`:

## Usage

1. Clone this repository to your local machine.
2. Make sure you have the required dependencies installed.
3. Modify the script with your desired tickers and directory paths.
4. Run the script to extract text data, preprocess it, and perform topic modeling.
5. Visualize the topics using the generated pyLDAvis visualizations.

## Scripts

- `sec_text_extraction.py`: Contains functions to extract text data from SEC filings and preprocess the text.
- `sec_topic_modeling.py`: Contains functions to apply LDA for topic modeling and visualize the topics.
- `main.py`: Main script to execute the entire process, from text extraction to topic modeling.

## Outputs

- **Visualizations**: Visualizations of topics identified by LDA for each ticker and year.
- **Processed Text**: Preprocessed text data ready for topic modeling.

## Additional Notes

- Ensure you have proper directory paths set up for saving outputs and accessing SEC filings.
- Modify the script according to your requirements, such as specifying different tickers or time periods.
- Experiment with different parameters for LDA to achieve optimal topic modeling results.


# Comparion between company

## Introduction
This repository contains code for comparing the similarity of TF-IDF matrices generated from financial reports (10-K reports) of different companies. The code processes text data from these reports, calculates TF-IDF matrices, compares the matrices using cosine similarity, and visualizes the results using a heatmap.

## Setup
1. Clone this repository to your local machine.
2. Ensure you have Python installed (version 3.6 or later).
3. Install the required libraries by running:
   ```
   pip install numpy scikit-learn seaborn matplotlib nltk
   ```
4. Download NLTK data by running the following Python script:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

## Usage
1. Ensure your financial reports (10-K reports) are organized in directories according to tickers. Each ticker should have its own directory containing the reports.
2. Update the `base_directory` variable in the code to specify the path to the directory containing the ticker directories.
3. Update the `tickers` list in the code to include the tickers for which you want to compare the reports.
4. Run the Python script.
5. The script will preprocess the text data, calculate TF-IDF matrices, compare the matrices using cosine similarity, and generate a heatmap to visualize the results.

## Files
- `heatmap_comparison.py`: Python script for comparing TF-IDF matrices and generating a heatmap.
- Sample text files (10-K reports) for demonstration purposes.

## Results
After running the script, the following results will be displayed:
- Minimum shape of TF-IDF matrices for each ticker.
- Overall minimum shape across all tickers.
- TF-IDF matrices for each ticker and report.
- Similarity values between TF-IDF matrices of different tickers.

## Visualization
A heatmap will be generated to visualize the cosine similarity between TF-IDF matrices of different tickers.

## Notes
- Ensure the text files (10-K reports) are in plain text format and stored in the specified directory structure.
- This code assumes English language text data.
- The preprocessing steps include converting text to lowercase, removing special characters, tokenization, removing stopwords, and lemmatization.





