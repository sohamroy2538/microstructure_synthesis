import os
import argparse
import torch
from train import create_img_dendritic,create_img_normal
import matplotlib.pyplot as plt


MODEL_LIST = os.listdir("./models")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to evaluate")
    parser.add_argument("--image_type", type=str, default = "non_dendritic", required=True, help="Path to the image to evaluate")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model of the corresponding image")
    parser.add_argument("--use_bcos", type=bool, default=True, help="Whether to use BCOS (default: True)")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--image_size", type=int, required=False, default=400, help="Size of the generated image (e.g., 256 for 256x256)")
    parser.add_argument("--output_dir", type=str, required=False,  default="./output", help="Directory to save the generated images")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = args.image_path
    image_type = args.image_type
    model_path = args.model_path
    use_bcos = args.use_bcos
    num_images = args.num_images
    image_size = args.image_size
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if model for the image exists
    '''model_name = os.path.basename(image_path)
    model_path = f"./models/{model_name}.pth"
    if model_name not in MODEL_LIST:
        print(f"Model for the image '{model_name}' not found in ./models. Please train a model for this image.")
        exit(1)'''

    # Create and load the model
    if image_type == "dendritic":
        create_img_model = create_img_dendritic(image_path, img_size=image_size, use_bcos=use_bcos)

    elif image_type == "non_dendritic":
        create_img_model = create_img_normal(image_path, img_size=image_size, use_bcos=use_bcos, task = "train_from_scratch")
        
    create_img_model.model.load_state_dict(torch.load(model_path))
    create_img_model.eval()

    # Generate and save images
    for i in range(num_images):
        with torch.no_grad():
            create_img_model()
            tensor_squeezed = (create_img_model.img).squeeze().permute(1, 2, 0).cpu().detach()
            # Save and plot the generated image
            output_path = os.path.join(output_dir, f"{os.path.basename(image_path[: -4])}_output_{i + 1}.png")
            plt.imshow(tensor_squeezed)
            plt.axis("off")
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
            print(f"Saved image to {output_path}")
            #plt.show()