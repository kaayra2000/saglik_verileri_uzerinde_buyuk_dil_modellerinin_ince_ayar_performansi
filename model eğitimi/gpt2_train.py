from sabitler import *
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer
df = pd.read_csv(data_filepath)
dataset = df[metin].tolist()
dataset = Dataset.from_pandas(df)
dataset_split = dataset.train_test_split(test_size=0.2)
dataset_split = dataset_split.flatten()

# Load the GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a preprocessing function that tokenizes the text using the tokenizer
# and joins the tokens back together into a string
def preprocess_function(examples):
    return tokenizer(["".join(x) for x in examples], truncation=True)

# Apply the preprocessing function to the Yelp dataset using the `map` method
# `batched=True` allows the dataset to be processed in batches, reducing memory usage
# `num_proc=12` parallelizes the preprocessing across 12 processes for faster execution
# `remove_columns=dataset_split["train"].column_names` removes the columns from the dataset that are not needed for further processing
tokenized_yelp = dataset_split.map(
    preprocess_function,
    batched=True,
    num_proc=12,
    remove_columns=dataset_split["train"].column_names,
)


# Set the maximum block size for the text data
block_size = 128

# Define a function to group input examples into blocks of size block_size
# The function concatenates the input examples into a single list and then
# divides the list into blocks of size block_size
def group_texts(examples):
    # Concatenate the examples into a single list
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    
    # Compute the total length of the concatenated examples
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # Round the total length down to the nearest multiple of block_size
    total_length = (total_length // block_size) * block_size
    
    # Divide the concatenated examples into blocks of size block_size
    # for each key in the input examples
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    # Return the resulting blocks
    return result

lm_dataset = tokenized_yelp.map(group_texts, batched=True, num_proc=2)