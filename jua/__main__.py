from jua.dataset import Dataset
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "filepath",
        nargs="?",
        default="data/jurisprudencia-selecionada.csv",
        help="CSV file path to load",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="jua-dataset",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--sample_size",
        nargs="?",
        default=None,
        type=int,
        help="Sample size to load",
    )
    args = parser.parse_args()
    ds = Dataset(args.filepath, args.sample_size)
    ds.save_dataset(args.directory)
    print(f"Dataset saved to {args.directory}")