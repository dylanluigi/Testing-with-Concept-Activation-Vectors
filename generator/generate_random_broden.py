import os, random, shutil, glob


source_dir = "imagenet_dump"     
target_root = "data/concepts"
os.makedirs(target_root, exist_ok=True)


n_sets = 4
n_images_per_set = 120
excluded_keywords = ["zebra", "striped", "zigzagged", "dotted", "braided", "nautilus"]  


all_images = glob.glob(os.path.join(source_dir, "*.jpg")) + \
             glob.glob(os.path.join(source_dir, "*.jpeg")) + \
             glob.glob(os.path.join(source_dir, "*.JPEG"))


filtered_images = [img for img in all_images
                   if not any(k.lower() in img.lower() for k in excluded_keywords)]

print(f"Total available for random sampling: {len(filtered_images)}")


for i in range(n_sets):
    subset = random.sample(filtered_images, n_images_per_set)
    target_dir = os.path.join(target_root, f"random_{i}")
    os.makedirs(target_dir, exist_ok=True)

    for src in subset:
        shutil.copy(src, target_dir)

    print(f"Created {target_dir} with {len(os.listdir(target_dir))} images")
