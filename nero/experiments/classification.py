import pathlib

import numpy as np
import pandas as pd
import sklearn.model_selection as skl_model_selection
from tqdm import trange

import nero.constants as constants
import nero.converters.tudataset as tudataset
import nero.embedding.pipelines as pipelines
import nero.tools.logging as logging

logger = logging.get_configured_logger()


def tu_datasets_demo() -> None:
    samples, classes, description = tudataset.tudataset2persisted('MUTAG')
    pipeline = pipelines.create_pipeline(description, 'AV0', (20, 20, None))

    csv_dir_path = pathlib.Path(constants.CSV_DIR)
    csv_dir_path.mkdir(parents=True, exist_ok=True)
    csv_file = f"{logging.formatted_today()}.csv"
    with open(csv_dir_path / csv_file, 'a') as file:
        file.write("Benchmark,Repetition,Split,Accuracy")

    for repetition in trange(10, desc="Assessing accuracy for benchmark MUTAG"):
        k_fold = skl_model_selection.StratifiedKFold(n_splits=10, shuffle=True)
        for split, (train_indices, test_indices) in enumerate(k_fold.split(list(range(len(classes))), classes)):
            train_samples = [samples[i] for i in train_indices]
            train_classes = [classes[i] for i in train_indices]
            test_samples = [samples[i] for i in test_indices]
            test_classes = [classes[i] for i in test_indices]
            pipeline.fit(train_samples, train_classes)
            accuracy_score = pipeline.score(test_samples, test_classes)
            with open(csv_dir_path / csv_file, 'a') as file:
                file.write(f"\nMUTAG,{repetition},{split},{accuracy_score}")

    df = pd.read_csv(csv_dir_path / csv_file)
    df = df.groupby(
        ['Benchmark', 'Repetition'], as_index=False,
    ).aggregate(
        {
            'Accuracy': [np.mean, np.std],
        }
    ).groupby(
        ['Benchmark'], as_index=False,
    ).aggregate(
        {
            ('Accuracy', 'mean'): [np.mean, np.std],
            ('Accuracy', 'std'): [np.mean, np.std],
        }
    )
    print(df)
