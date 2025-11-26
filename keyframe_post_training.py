import yaml
from app.vjepa_droid.train import main


if __name__ == "__main__":
    args = yaml.safe_load(open("configs/train/vitg16/droid-256px-8f.yaml"))
    main(args)