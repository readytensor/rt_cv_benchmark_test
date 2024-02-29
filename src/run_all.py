import os
import torch
import pandas as pd
from config import paths
from torch_utils.dataloader import CustomDataLoader
from models.custom_trainer import CustomTrainer
from utils import read_json_as_dict, set_seeds, get_model_parameters
from score import (
    calculate_confusion_matrix,
    evaluate_metrics,
    plot_and_save_confusion_matrix,
    save_metrics_to_csv,
)

from utils import track_resources
from logger import get_logger

logger = get_logger(task_name="run_all")


def main():
    config = read_json_as_dict(paths.CONFIG_FILE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = config.get("num_epochs")
    loss_choice = config.get("loss_function")
    num_workers = config.get("num_workers")
    validation_size = config.get("validation_size")
    loss_function = (
        torch.nn.CrossEntropyLoss()
        if loss_choice == "crossentropy"
        else torch.nn.MultiMarginLoss()
    )
    logger.info("Setting seeds to:", config["seed"])
    set_seeds(config["seed"])

    predictions_folder = paths.RUN_ALL_PREDICTIONS_DIR
    artifacts_folder = paths.RUN_ALL_ARTIFACTS_DIR
    model_names = config.get("run_all_model_names")

    for model_name in model_names:
        params = get_model_parameters(
            model_name=model_name,
            hyperparameters_file_path=paths.HYPERPARAMETERS_FILE,
            hyperparameter_tuning=config["hyperparameter_tuning"],
        )

        batch_size = params["batch_size"]
        image_size = params["image_size"]
        optimizer = params.get("optimizer")
        lr = params.get("lr")

        custom_data_loader = CustomDataLoader(
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            validation_size=validation_size,
        )
        logger.info(f"\nWorking on model: {model_name}")

        model_predictions_folder = os.path.join(predictions_folder, model_name)
        model_artifacts_folder = os.path.join(artifacts_folder, model_name)
        os.makedirs(model_artifacts_folder, exist_ok=True)
        os.makedirs(model_predictions_folder, exist_ok=True)

        train_loader, test_loader, validation_loader = (
            custom_data_loader.train_loader,
            custom_data_loader.test_loader,
            custom_data_loader.validation_loader,
        )

        num_classes = len(train_loader.dataset.classes)

        trainer = CustomTrainer(
            train_loader=train_loader,
            test_loader=test_loader,
            validation_loader=validation_loader,
            num_classes=num_classes,
            model_name=model_name,
            output_folder=model_artifacts_folder,
            optimizer=optimizer,
            lr=lr,
        )
        trainer.set_device(device)
        trainer.set_loss_function(loss_function)

        checkpoint_dir_path = os.path.join(model_artifacts_folder, "checkpoints")
        os.makedirs(checkpoint_dir_path, exist_ok=True)

        logger.info("Training model...")
        metrics_history = trainer.train(
            num_epochs=num_epochs, checkpoint_dir_path=checkpoint_dir_path
        )

        predictor_path = os.path.join(model_artifacts_folder, "predictor")

        logger.info("Saving model...")
        trainer.save_model(predictor_path=predictor_path)

        logger.info("Saving metrics to csv...")
        save_metrics_to_csv(
            metrics_history,
            output_folder=model_artifacts_folder,
            file_name="train_validation_metrics.csv",
        )

        logger.info("Predicting train labels...")
        train_labels, train_pred, _ = trainer.predict(train_loader)

        logger.info("Saving train confusion matrix...")
        train_cm = calculate_confusion_matrix(
            all_labels=train_labels, all_predictions=train_pred
        )

        logger.info("Saving train confusion matrix plot...")
        plot_and_save_confusion_matrix(
            cm=train_cm,
            phase="train",
            model_name=trainer.__class__.__name__,
            output_folder=model_artifacts_folder,
            class_names=trainer.train_loader.dataset.classes,
        )

        if validation_loader:
            logger.info("Predicting validation labels...")
            validiation_labels, validation_pred, _ = trainer.predict(validation_loader)

            logger.info("Saving validation confusion matrix...")
            validation_cm = calculate_confusion_matrix(
                all_labels=validiation_labels, all_predictions=validation_pred
            )

            logger.info("Saving validation confusion matrix plot...")
            plot_and_save_confusion_matrix(
                cm=validation_cm,
                phase="validation",
                model_name=model_name,
                output_folder=model_artifacts_folder,
                class_names=trainer.train_loader.dataset.classes,
            )

        labels, predictions, probs = trainer.predict(test_loader)

        test_metrics = evaluate_metrics(
            labels=labels,
            predictions=predictions,
            probabilities=probs,
            loss_function=trainer.loss_function,
            top_k=[5],
            n_classes=trainer.num_classes,
        )

        test_metrics_df = pd.DataFrame(
            {
                "test loss": [test_metrics["loss"]],
                "test accuracy": [test_metrics["accuracy"]],
                "test macro_f1": [test_metrics["macro_f1"]],
                "test weighted_f1": [test_metrics["weighted_f1"]],
                "test macro_recall": [test_metrics["macro_recall"]],
                "test weighted_recall": [test_metrics["weighted_recall"]],
                "test macro_precision": [test_metrics["macro_precision"]],
                "test weighted_precision": [test_metrics["weighted_precision"]],
                "test top_k_accuracy": [test_metrics["top_k_accuracy"]],
            }
        )
        logger.info("Saving test metrics to csv...")
        save_metrics_to_csv(
            test_metrics_df,
            output_folder=model_predictions_folder,
            file_name="test_metrics.csv",
        )

        logger.info("Saving confusion matrix...")
        test_cm = calculate_confusion_matrix(
            all_labels=labels, all_predictions=predictions
        )
        plot_and_save_confusion_matrix(
            cm=test_cm,
            phase="test",
            model_name=trainer.__class__.__name__,
            output_folder=model_predictions_folder,
            class_names=trainer.train_loader.dataset.classes,
        )

        logger.info(
            f"Training Accuracy (Last Epoch): {metrics_history['train accuracy'][-1]}"
        )

        logger.info(f"Training and evaluation for model {model_name} completed.\n")

    logger.info("All models have been processed.")


if __name__ == "__main__":
    track_resources(main, logger=logger.info)
