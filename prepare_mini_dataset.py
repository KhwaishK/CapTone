import os
import random
import shutil
import yaml

def create_mini_val_dataset(val_dir, mini_val_dir, n_val=50):
    os.makedirs(mini_val_dir, exist_ok=True)
    val_imgs = random.sample(os.listdir(val_dir), n_val)
    for img in val_imgs:
        shutil.copy(os.path.join(val_dir, img), os.path.join(mini_val_dir, img))
    print(f"Mini dataset created: {len(val_imgs)} images at {mini_val_dir}")


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    val_dir = params["dataset"]["val_dir"]
    mini_val_dir = params["dataset"]["mini_val_dir"]
    n_val = params["dataset"]["n_val"]                       

    create_mini_val_dataset(val_dir, mini_val_dir, n_val)


if __name__ == "__main__":
    main()

