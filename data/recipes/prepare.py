import os
import requests
import tiktoken
import numpy as np
import pandas as pd
import random

import spacy
from spacy.lang.en.stop_words import STOP_WORDS


# Define a function to perform stemming and stop word removal
def preprocess(text):
    doc = nlp(text)
    # Remove stop words
    tokens = [token for token in doc if not token.is_stop]
    # Perform stemming
    stemmer = Stemmer(nlp.vocab)
    stemmed_tokens = [stemmer(token.text).lower() for token in tokens]
    return stemmed_tokens


def row2recipe(title, ing, steps):
    recipe = f"""Title : {title}
Ingredients: {ing}
Steps: {steps}
----------------\n"""
    return recipe


def row2recipeNoSteps(title, ing, steps):
    recipe = f"""Title : {title}
Ingredients: {ing}
----------------\n"""
    return recipe


# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(input_file_path):
    print("INPUT.TXT NOT FOUND")
    print("# ---------------------------------")
    # data_url = "https://pspipelines.blob.core.windows.net/recipe-nanogpt/recipes_willko_fix_v1.csv"
    df = pd.read_csv("data.csv")
    df["steps"] = df["steps"].fillna(" ")

    # Load the English language model in Spacy

    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp.pipe(list(df["steps"]))
    # cleaned_steps = []

    # i = 0
    # for d in doc:
    #     tokens = [
    #         token.lemma_
    #         for token in d
    #         if not token.is_stop and not token.is_digit and not token.is_bracket
    #     ]
    #     cleaned_steps.append(" ".join(tokens))
    #     i += 1
    #     if i % 1000 == 0:
    #         print(f"CLEANING STEPS -{i}/{len(df)}")

    # df["cleaned_steps"] = cleaned_steps
    # df.to_csv("data.csv", index=False)
    print("DOWNLOADED CSV")
    print("# ---------------------------------")

    with open("input.txt", "w") as f:
        for index, row in df.iterrows():
            recipe = row2recipeNoSteps(
                row["title"].lower(),
                row["ingredient"].replace("diced", ""),
                row["steps"].replace("\n", ""),
            )
            f.write(recipe)

            ings = row["ingredient"].replace("diced", "").split(", ")
            for n in range(10):
                random.shuffle(ings)
                recipe = row2recipeNoSteps(
                    row["title"].lower(),
                    ", ".join(ings),
                    row["steps"].replace("\n", ""),
                )
                f.write(recipe)

    print("INPUT.TXT CREATED")
    print("# ---------------------------------")

print("DATA PREP START")
print("# ---------------------------------")

with open(input_file_path, "r") as f:
    data = f.read()
n = len(data.split("----------------\n"))
recipes = data.split("----------------\n")
number_of_tokens = len(data)
train_data = "<|endoftext|>".join(recipes[: int(n * 0.9)])
val_data = "<|endoftext|>".join(recipes[int(n * 0.9) :])

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
print(f"END OF TEXT TOKEN NUMBER : {tiktoken.encoding_for_model('gpt2').eot_token}")
print(
    f"END OF TEXT TOKEN : {enc.decode([tiktoken.encoding_for_model('gpt2').eot_token])}"
)
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"train has {len(train_ids):,} recipe tokens")
print(f"val has {len(val_ids):,} recipe tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

print("DATA PREP DONE")
print("# ---------------------------------")

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
