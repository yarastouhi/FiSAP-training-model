from src.train_occurences import train_occurrence
from src.train_severity import train_severity
from src.train_time import train_time
from src.predict import run_inference


def main():
    print("Training wildfire occurrence model...")
    # train_occurrence()

    print("Training wildfire severity model...")
    train_severity()

    print("Training fire time prediction model...")
    train_time()

    print("Running inference on future data...")
    run_inference()


if __name__ == "__main__":
    main()
