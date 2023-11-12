import argparse
from transformers import pipeline
import re
from diffusers import StableDiffusionPipeline
import torch
import os
from pathlib import Path
import random
import pandas as pd
import shutil


DEBUG = False

list_of_products = ["Dress", "Sweater", "T-Shirt", "Trousers", "Top", "Skirt", "Earrings", "Blouse", "Jeans", "Shirt", "Handbag", "Sandals", "Jacket", "Jumpsuit", "Coat", "Belt", "Cardigan", "Blazer", "Shorts", "Shoes", "Necklace", "Shoulder bag", "Crossbody bag", "Ankle Boots", "Sunglasses", "Puffer coat", "Kerchief", "Scarf", "Wallet", "Sweatshirt", "Glasses", "Beanie", "Totes bag", "Boots", "Cosmetic bag", "Vest", "Ring", "Trenchcoat", "Leggings", "Card holder", "Bracelet", "Bodysuit", "Gloves", "Bikini top", "Case", "Foulard", "Trainers", "Purse", "Pyjama Trousers", "Socks", "Leather Jacket", "Hat", "Pyjama", "Cap", "Bras", "Bermudas", "Overshirt", "Outer vest", "Bikini pantie", "Sweater Vest", "Pyjama Shirt", "Bucket bag", "Citybag", "Tights", "Overall", "Poloshirt", "Hairclip", "Parka", "Swimsuit", "Poncho", "Cardigan Vest", "Nightgown", "Clutch and Pochettes", "Shape", "Kaftan", "Pyjama Shorts", "Cape", "Pyjama T-Shirt", "Belt bag", "Hairband", "Pyjama Sweatshirt", "Dressing Gown (Bata)", "Pyjama Top", "Tie", "Glasses case", "Mini Bag", "Backpack", "Joggers", "Headband", "Sock (Pack)", "Jacket (Cazadora)", "Knicker", "Skort", "Turban", "Pichi", "Braces", "Slippers", "Pyjama Cardigan", "Clogs"]
list_of_colors = ["Grey", "White", "Blue", "Brown", "Pink", "Red", "Green", "Yellow", "Orange", "Purple"]
list_of_materials = ["Flat", "Complements", "Circular", "Tricot", "Jeans", "Faux Leather", "Leather"]

dict_of_lists = {"product": list_of_products, "color": list_of_colors, "material": list_of_materials}

def posprocess_product(prod):

    try:
        p = re.findall(r"Product: (.*), C", prod)[0]
        c = re.findall(r"Color: (.*), M", prod)[0]
        m = re.findall(r"Material: (.*)", prod)[0]
        
        return_dict = {"product": p, "color": c, "material": m}
        
        for key in return_dict.keys():
            if return_dict[key] not in dict_of_lists[key]:
                # Revisar que estes todos los 3 atributos
                return None
        if DEBUG:
            print(prod)
    
        return return_dict
    except Exception:
        # Caso de que el material este partido a la mitad
        return None
    return None

def posprocess_output(output):
    outfit = []
    products = output.split("\n")[1:]
    for product in products:
        res = posprocess_product(product)
        if res is not None:
            outfit.append(res)
    return outfit

def generate_prompt(product):
    text = f'Give me a outfit that combine with the following piece:\n- Product: {product["product"]}, Color: {product["color"]}, Material: {product["material"]}'
    if DEBUG:
        print(text)
    return text

def generate_outfits(generator, product, n_outfits = 10, max_length = 100):
    prompt = generate_prompt(product)
    outfits = generator(prompt, max_length=max_length, num_return_sequences=n_outfits)
    outfits = [outfit["generated_text"] for outfit in outfits]
    outfits_list = []
    for outfit in outfits:
        if DEBUG:
            print(f"\n{prompt}")
        outfits_list.append(posprocess_output(outfit))
    return outfits_list

def gen_sd_image(outfit_list, pipe):
    text = "modelshoot style a woman wearing: "
    for outfit_dict in outfit_list[:-1]:
        text += f"{outfit_dict['product']} {outfit_dict['color']}, "
    text += f"and {outfit_list[-1]['product']} {outfit_list[-1]['color']}."
    if DEBUG:
        print(text)
    return pipe(text, guidance_scale=10).images[0]

def find_product_in_dataset(des_product_type_get, des_agrup_color_eng_get, des_fabric_get, product_ds):
    productv1 = product_ds.loc[(product_ds['des_product_type'] == des_product_type_get) & (product_ds['des_agrup_color_eng'] == des_agrup_color_eng_get) & (product_ds['des_fabric'] == des_fabric_get), 'cod_modelo_color'].tolist()
    if not productv1:
        productv2 = product_ds.loc[(product_ds['des_product_type'] == des_product_type_get) & (product_ds['des_agrup_color_eng'] == des_agrup_color_eng_get), 'cod_modelo_color'].tolist()
        if not productv2:
            productv3 = product_ds.loc[product_ds['des_product_type'] == des_product_type_get, 'cod_modelo_color'].tolist()
            if not productv3:
                return None
            return productv3, 1
        return productv2, 2
    return productv1, 3

def create_outfit(outfit_model, product_csv):
    new_outfit = []
    for product_dict in outfit_model:
        products_cand, num_atributes = find_product_in_dataset(product_dict["product"], product_dict["color"], product_dict["material"], product_csv)
        products_cand_random = random.choice(products_cand)
        new_outfit.append(products_cand_random)
    return new_outfit

def inference(args):
    model_path = "tj-solergibert/Mango-DA-GPT2" #os.path.join(os.getcwd(), "juan")
    img_output = os.path.join(os.getcwd(), "Output")
    
    product_ds_path = os.path.join(os.getcwd(), "datathon", "dataset", "filtered_product_data.csv")
    product_csv = pd.read_csv(product_ds_path)

    generator = pipeline('text-generation', model=model_path, device = "cuda")
    prod = {'product': args.product, 'color': args.color, 'material': args.material}
    outfits_processed = generate_outfits(generator, prod, n_outfits = args.n_outfits)

    for outfit in outfits_processed:
        
        outfit_product_ids = create_outfit(outfit, product_csv)

        if outfit_product_ids is not None:
            folder_name = "Outfit"
            for outfit_dict in outfit:
                folder_name += f"_{outfit_dict['product']}{outfit_dict['color']}"
            
            outfit_output_folder = os.path.join(img_output, folder_name)
            Path(outfit_output_folder).mkdir(parents=True, exist_ok=True)

            outfit_product_ids = create_outfit(outfit, product_csv)
            for product_id in outfit_product_ids:
                src_filename = product_csv[product_csv["cod_modelo_color"] == product_id]["des_filename"].item()
                dst_filename = os.path.join(outfit_output_folder, f"{product_id}.jpg")
                shutil.copyfile(src_filename, dst_filename)

            if args.generate_sd:
                model_id = "wavymulder/modelshoot"
                pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
                image = gen_sd_image(outfit, pipe)
                image.save(os.path.join(outfit_output_folder, "SD.png"))

def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--product",
        type=str,
        default="Skirt",
        help="Location on where to store experiment tracking logs` and relevent project information",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="Blue",
        help="Location on where to store experiment tracking logs` and relevent project information",
    )
    parser.add_argument(
        "--material",
        type=str,
        default="Jeans",
        help="Location on where to store experiment tracking logs` and relevent project information",
    )
    parser.add_argument(
        "--generate_sd",
        action="store_true",
        help="Whether to generate images for the outfit with StableDiffusion",
    )
    parser.add_argument(
        "--n_outfits",
        type=int,
        help="Number of outfits to generate",
    )
    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()