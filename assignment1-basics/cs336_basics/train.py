import os
import json
from cs336_basics.model import Transformer
from cs336_basics.optm.Optimizer import AdamW
from cs336_basics.modules.utility import get_device, seed_everything
from cs336_basics.modules.train_engine import train
from torchinfo import summary
from types import SimpleNamespace


def main(train_config_path):
    with open(train_config_path, "r") as f:
        train_config = json.load(f)

    train_config = SimpleNamespace(**train_config)
    save_path = os.path.join(
        train_config.save_checkpoint_dir,
        train_config.model_name
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_config.device = get_device()

    seed_everything(train_config.seed)

    # model = Transformer(vocab_size=10000, context_length=256, d_model=512, num_layers=4, num_heads=16, d_ff=1344, theta=10000.0)
    model = Transformer(vocab_size=10000, context_length=256, d_model=512, num_layers=4, num_heads=16, d_ff=1344, theta=10000.0, use_moe=True) 
    model = model.to(train_config.device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=train_config.min_lr, betas=tuple(train_config.betas), weight_decay=train_config.weight_decay)

    print("Starting training...")
    train(model, optimizer, train_config)

    print("Training completed.")


if __name__ == "__main__":
    train_config_path = "cs336_basics/config.json"
    main(train_config_path)
    # with open(train_config_path, "r") as f:
    #     train_config = json.load(f)
    # train_config = SimpleNamespace(**train_config)
    # print(len(train_config.betas))