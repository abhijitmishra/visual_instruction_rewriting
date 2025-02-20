# args
import argparse


def get_args_pretraining():
    parser = argparse.ArgumentParser(description="Training Arguments Parser")

    parser.add_argument(
        "--num_train_epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument(
        "--remove_unused_columns",
        action="store_true",
        default=False,
        help="Remove unused columns",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size per device during training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100, help="Number of warmup steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 parameter for Adam optimizer",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Log every X updates steps"
    )
    parser.add_argument(
        "--optim", type=str, default="adamw_hf", help="Optimizer to use"
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        help="The checkpoint save strategy",
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Limit the total amount of checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./revision_v1",
        help="The output directory where the model checkpoints will be written",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Whether to use bf16 precision",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        nargs="+",
        default=["tensorboard"],
        help="The list of integrations to report the results and logs to",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        action="store_true",
        default=False,
        help="Whether to pin memory in data loaders",
    )

    return parser.parse_args()


def get_args_fine_tuning():
    parser = argparse.ArgumentParser(description="Training Arguments Parser")

    parser.add_argument(
        "--num_train_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--remove_unused_columns",
        action="store_true",
        default=False,
        help="Remove unused columns",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size per device during training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=10, help="Number of warmup steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 parameter for Adam optimizer",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Log every X updates steps"
    )
    parser.add_argument(
        "--optim", type=str, default="adamw_hf", help="Optimizer to use"
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        help="The checkpoint save strategy",
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Limit the total amount of checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="revision_v1",
        help="The output directory where the model checkpoints will be written",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Whether to use bf16 precision",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        nargs="+",
        default=["tensorboard"],
        help="The list of integrations to report the results and logs to",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        action="store_true",
        default=False,
        help="Whether to pin memory in data loaders",
    )

    return parser.parse_args()


# Example usage:
if __name__ == "__main__":
    args = get_args_pretraining()
    print(args)
