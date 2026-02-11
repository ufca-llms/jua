from argparse import ArgumentParser
from jua.train.train_dataset import TrainDataset


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to the results file",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./jua-dataset",
        help="Path to the dataset directory",
    )
    parser.add_argument(
       "--alpha",
        type=float,
        default=0.01,
        help="Alpha value for cutoff calculation",
    )

    args = parser.parse_args()

    trainer = TrainDataset(
        results_path=args.results_path,
        dataset_path=args.dataset_path,
        alpha=args.alpha,
    )
    trainer.create()

if __name__ == "__main__":
    main()