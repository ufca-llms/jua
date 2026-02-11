import json, os
from tqdm import tqdm
from scipy import stats
from scipy.stats import shapiro
import numpy as np
import matplotlib.pyplot as plt
from beir.datasets.data_loader import GenericDataLoader

class TrainDataset:
    """
    Class to create training dataset based on results and statistical cutoff
    """
    def __init__(self, results_path, dataset_path="./jua-dataset", output_path="./data/train_datase.jsonl", alpha=0.01, max_samples=100):
        """Initialize the TrainDataset class.
        Args:
            results_path (str): Path to the results file.
            dataset_path (str): Path to the dataset directory.
            output_path (str): Path to the output file.     
            alpha (float): Alpha value for cutoff calculation.
            max_samples (int): Maximum number of samples to consider for cutoff.
        
        """
        self.results = json.load(open(results_path))
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.alpha = alpha
        self.max_samples = max_samples

        self.corpus, self.queries, self.qrels = self.__load_dataset()

    def __cutoff(self, items):
        """Calculate cutoff based on statistical analysis.
        Args:
            items (dict): Dictionary of items with their counts.
        Returns:
            list: List of selected item names above the cutoff.
        """
        names = list(items.keys())[:self.max_samples]
        counts = list(items.values())[:self.max_samples]
        a = np.array(counts)
        mean = a.mean()
        std = a.std()

        z = stats.norm.ppf(1-self.alpha)
        x = (z*std)+mean

        idxs = np.where(a > x)[0]

        selected_names = np.array(names)[idxs]

        return selected_names
    
    def __load_dataset(self):
        """Load dataset using GenericDataLoader.
        Returns:
            tuple: corpus, queries, and qrels.
        """
        corpus_path = os.path.join(self.dataset_path, "corpus.jsonl")
        query_path = os.path.join(self.dataset_path, "queries.jsonl")
        qrels_path = os.path.join(self.dataset_path,'qrels', "train.tsv")
        print(f"Loading dataset from {corpus_path}, {query_path}, {qrels_path}")
        corpus, queries, qrels = GenericDataLoader(
            corpus_file=corpus_path, 
            query_file=query_path, 
            qrels_file=qrels_path).load_custom()
        
        return corpus, queries, qrels

    def create(self):
        for k,v in tqdm(self.results.items()):
            gold_id = k[:-2]
            # skip impossible cases where gold_id is not in the top-1000 retrieved items
            if gold_id not in v:
                continue
            if k not in self.qrels:
                continue
            selected_items = list(self.__cutoff(v))
            
            if gold_id in selected_items:  
                selected_items.remove(gold_id)
            
            query_text = self.queries[k]
            gold_text = self.corpus[gold_id]['text']

            negative_texts = [self.corpus[item]['text'] for item in selected_items]

            # {"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "sentence2"}]], "negative_messages": [[{"role": "user", "content": "sentence3"}], [{"role": "user", "content": "sentence4"}]]}

            output_data = {
                "messages": [{"role": "user", "content": query_text}],
                "positive_messages": [[{"role": "user", "content": gold_text}]],
                "negative_messages": [[{"role": "user", "content": text}] for text in negative_texts]
            }

            with open(self.output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
        print(f"Training dataset created at {self.output_path}")