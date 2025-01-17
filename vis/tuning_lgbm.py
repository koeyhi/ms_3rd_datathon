import wandb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
import yaml


def train_lgbm(config_defaults=None):
    with wandb.init(config=config_defaults):
        config = wandb.config

        train_x = pd.read_csv("vis/data/train_x.csv")
        valid_x = pd.read_csv("vis/data/valid_x.csv")
        test_x = pd.read_csv("vis/data/test_ft.csv")
        train_y = pd.read_csv("vis/data/train_y.csv")
        valid_y = pd.read_csv("vis/data/valid_y.csv")
        test_y = pd.read_csv("vis/data/test_target.csv")

        model = LGBMClassifier(
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            max_depth=config.max_depth,
            min_child_samples=config.min_child_samples,
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
    with open("vis/sweep_lgbm.yaml", "r") as file:
        sweep_config = yaml.safe_load(file)
    sweep_id = wandb.sweep(sweep_config, project="my_model_experiments")
    wandb.agent(sweep_id, function=train_lgbm, count=10)
