# Fine-tuning-sentiment-analysis
### Sentiment Analysis using Fine-Tuned DistilBERT
This notebook fine-tunes the DistilBERT model on the IMDB movie review dataset for sentiment analysis.

#### Usage
The main steps are:

* Load and preprocess the IMDB dataset using TensorFlow Datasets
* Tokenize the text using the DistilBERT tokenizer
* Fine-tune the pre-trained DistilBERT model on the dataset
* Evaluate the fine-tuned model on the test set
* Make predictions on new examples
#### The key aspects are:

* Leveraging a pre-trained model - DistilBERT uncased base model
* Fine-tuning on the downstream sentiment analysis task
* Achieving 90% test accuracy after fine-tuning for 1 epoch on IMDB data
* The trained model can be used to classify if a given text contains positive or negative sentiment.

#### Requirements
The main libraries used are:

* TensorFlow 2.x
* Transformers
* Datasets
* Pandas, Numpy
It runs on a GPU runtime for faster training.

#### References
> The pretrained DistilBERT model and tokenizer are from HuggingFace.

> The IMDB dataset is loaded using TensorFlow Datasets.
