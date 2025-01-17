import wandb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from catboost import CatBoostClassifier
import yaml
import json


def train_cat(config_defaults=None):
    with wandb.init(config=config_defaults, name="cat_tuned"):
        config = wandb.config

        train_x = pd.read_csv("vis/data/cat_train_x.csv")
        valid_x = pd.read_csv("vis/data/cat_valid_x.csv")
        test_x = pd.read_csv("vis/data/cat_test_ft.csv")
        train_y = pd.read_csv("vis/data/train_y.csv")
        valid_y = pd.read_csv("vis/data/valid_y.csv")
        test_y = pd.read_csv("vis/data/test_target.csv")
        with open("output/cat_features.json", "r") as f:
            cat_features = json.load(f)

        model = CatBoostClassifier(
            iterations=config.iterations,
            learning_rate=config.learning_rate,
            depth=config.depth,
            l2_leaf_reg=config.l2_leaf_reg,
            min_child_samples=config.min_child_samples,
            max_bin=config.max_bin,
            cat_features=cat_features,
            verbose=100,
            random_state=42,
        )

        model.fit(train_x, train_y)

        valid_pred = model.predict(valid_x)
        valid_acc = accuracy_score(valid_y, valid_pred)
        valid_roc_auc = roc_auc_score(valid_y, model.predict_proba(valid_x)[:, 1])

        train_ft = pd.concat([train_x, valid_x])
        train_target = pd.concat([train_y, valid_y])

        model.fit(train_ft, train_target)

        test_pred = model.predict(test_x)
        test_acc = accuracy_score(test_y, test_pred)
        test_roc_auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])

        wandb.log(
            {
                "valid_accuracy": valid_acc,
                "valid_roc_auc": valid_roc_auc,
                "test_accuracy": test_acc,
                "test_roc_auc": test_roc_auc,
            }
        )


if __name__ == "__main__":
    with open("vis/sweep_cat.yaml", "r") as file:
        sweep_config = yaml.safe_load(file)
    sweep_id = wandb.sweep(sweep_config, project="my_model_experiments")
    wandb.agent(sweep_id, function=train_cat, count=10)
