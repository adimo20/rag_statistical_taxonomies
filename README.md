# Project Overview

The underlying project is the implementation of the experiments conducted during my master's thesis. It is structured as follows:

- **baseline** - training scripts for supervised baseline approaches bert-base-german-uncased, gelectra-large and fastText.
- **hcs4os** - a stripped-down version of the decoupled Python [hcs4os](https://github.com/adimo20/hcs4os) package, which I am releasing alongside my thesis. Both repositories should be regarded as decoupled. The stripped-down version contains the base class for the linear RAG and the agentic RAG approaches used to conduct the experiments. The base classes build on the **dspy** framework. The base classes contain the relevant dspy configurations (temperature, max tokens, max iters, ...). They build on an abstract implementation of a ClassificationSystem, which enables operations like loading, looking up codes, and getting children and parent codes within a given classification system. It serves as a standardized interface for interacting with the taxonomy.
- **py_scripts** - contains all scripts used to generate the training data, putting it into format, mining hard negatives, and evaluating the results.
- **r_scripts** - only filters the dataset for all unique free text + label pairs.
- **RQ1** - contains the training script for the DPR model training, as well as an evaluation script that takes the fine-tuned model and evaluates it on the test set.
- **RQ_2_3** - contains the scripts used to conduct the linear RAG and agentic RAG experiments. The vector-databases/collections regarding the different configurations of retriever and context were embedded once and loaded from disk within the experiments.

## Order of execution

1. r_scripts/drop_duplicated_rows.R
2. py_scripts/train_test_split.py
3. py_scripts/mine_hard_negatives.py
4. py_scripts/create_rag_subsample.py
5. RQ1/DPR.py
6. RQ1/evaluate/evalDPR.py
7. RQ_2_3/linear_RAG_focus_description.py
8. RQ_2_3/linear_RAG_focus_includes.py
9. RQ_2_3/agentic_RAG.py
10. baseline/BERT.py
11. baseline/train_ft.py

## Additional Information

All experiments, regarding RQ1, RQ2, and RQ3, that tested more than one configuration were executed on the same script, using the defined CLI to change the input configuration.
Experiments regarding BERT, fastText, and the RAG-based approaches were all conducted using separate virtual environments, as there are certain package conflicts between the underlying packages. For fastText, the [fasttext-numpy2 wheel](https://pypi.org/project/fasttext-numpy2/) has been used, as it supports more recent Python versions than the original fastText, where support has been stopped, and it therefore remains bound to NumPy 1.XX.

Example CLI execution:

```bash
# --model_name pull from hugging face or local path
python baseline/BERT.py \
    --model_name "dbmdz/bert-base-german-uncased" \
    --output_dir "data/bert_base_german_uncased_finetuned" \
    --path_train "data/training_data/train.json" \
    --path_test "data/training_data/test.json"
```

All other experiments were conducted accordingly via CLI. The BERT script has been used to train both the BERT and ELECTRA-based models.

## Note for practitioners:

To try out the implemented approaches, it is possible to use this repository, but I strongly advise using the [hcs4os](https://github.com/adimo20/hcs4os) package, as it has built-in modules for several classification systems and provides an interface to add custom classification systems that are not pre-implemented. Implementations and usage are described in detail there.

## Note to data and used API

Data used to conduct the experiments is propiotary to Destatis and can therefore not be disclosed.
The API used to conduct the RAG experiments is an iternal API of Destatis. It follows the openai compatible standards, and therefore the model-ids one can see in the RQ_2_3 scripts has the prefix openai.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full text.
