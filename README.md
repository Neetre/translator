# english_translator

## Description

Decoder-Encoder Transformer Model for Ensligh to N translation.
Basically at first I tried to create a model from scratch, but I don't have enough computational power to train it, so I decided to use a pre-trained model and fine-tune it.
During finetune, I tested with 2 different models, the first one is the mT5 model from google, the second one is the Llama3.1-8B model from Meta.
Unfortunately, the mT5 model was a flop, it gave me problems with the tokenization of data, and in the end, I've lost hours trying to fix errors and training the model.
The Llama3.1-8B model gave better results, but it's still not perfect, I think that the model needs more training and more data to get better results (If more money comes in, I'll try to train the model with more data and more epochs).

My hope is to manage to train the from scratch model, but for now, I'll stick with the Llama3.1-8B model.

For the whole finetune part, I've created a new repo, this one was becomming too big.
Here is the [repo](https://github.com/Neetre/mT5)

Look at [Note](Note.md) ti get more infos about the whole process.

## Installation

### Requirements

- Python > 3.9

Run the .sh script to setup all the environment:

   ```bash
   cd bin
   ./setup.sh
   ```

Or follow the steps below:

### Environment setup

1. Create and activate a virtual environment:

   **Linux/macOS:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows:**

    ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Licence

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Look at [CITATIONS.md](CITATIONS.md) to get more infos about the sources used in this project.
Or look at [Note](Note.md) to get more infos about the whole process.

## Author

[Neetre](https://github.com/Neetre)
