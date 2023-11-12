import pandas as pd
import os
import csv

directorio_cwd = os.getcwd()
directorio_product = os.path.join(directorio_cwd, "datathon", "dataset", "product_data.csv")

# Read the original dataset
product_ds = pd.read_csv(directorio_product)

def filter_columns(dataset, columns_to_remove):
    # Filter out specified columns
    filtered_dataset = dataset.drop(columns=columns_to_remove, errors='ignore')
    return filtered_dataset

def replace_values(dataset, column, old_value, new_value):
    # Replace specified values in a column
    dataset[column] = dataset[column].replace(old_value, new_value)
    return dataset

def filter_and_save_dataset(output_file):
    # Filter the dataset to remove rows with des_product_category = 'Home' and des_product_family = 'Gadgets'
    filtered_dataset = product_ds[
        (~product_ds['des_product_category'].isin(['Home', 'Beauty'])) &
        (product_ds['des_product_family'] != 'Gadgets')
    ]

    # Filter columns from the dataset
    columns_to_remove = ['cod_color_code', 'des_color_specification_esp', 'des_sex', 'des_age', 'des_line', 'des_product_category', 'des_product_aggregated_family','des_product_family']
    filtered_dataset = filter_columns(filtered_dataset, columns_to_remove)

    # Replace Materials in the dataset
    filtered_dataset = replace_values(filtered_dataset, 'des_fabric', 'P-PLANA', 'Flat')
    filtered_dataset = replace_values(filtered_dataset, 'des_fabric', 'C-COMPLEMENTOS', 'Complements')
    filtered_dataset = replace_values(filtered_dataset, 'des_fabric', 'K-CIRCULAR', 'Circular')
    filtered_dataset = replace_values(filtered_dataset, 'des_fabric', 'T-TRICOT', 'Tricot')
    filtered_dataset = replace_values(filtered_dataset, 'des_fabric', 'J-JEANS', 'Jeans')
    filtered_dataset = replace_values(filtered_dataset, 'des_fabric', 'O-POLIPIEL', 'Faux Leather')
    filtered_dataset = replace_values(filtered_dataset, 'des_fabric', 'L-PIEL', 'Leather')

    # Replace Colors in the dataset

    filtered_dataset = replace_values(filtered_dataset, 'des_agrup_color_eng', 'GREY', 'Grey')
    filtered_dataset = replace_values(filtered_dataset, 'des_agrup_color_eng', 'WHITE', 'White')
    filtered_dataset = replace_values(filtered_dataset, 'des_agrup_color_eng', 'BLUE', 'Blue')
    filtered_dataset = replace_values(filtered_dataset, 'des_agrup_color_eng', 'BROWN', 'Brown')
    filtered_dataset = replace_values(filtered_dataset, 'des_agrup_color_eng', 'PINK', 'Pink')
    filtered_dataset = replace_values(filtered_dataset, 'des_agrup_color_eng', 'RED', 'Red')
    filtered_dataset = replace_values(filtered_dataset, 'des_agrup_color_eng', 'GREEN', 'Green')
    filtered_dataset = replace_values(filtered_dataset, 'des_agrup_color_eng', 'YELLOW', 'Yellow')
    filtered_dataset = replace_values(filtered_dataset, 'des_agrup_color_eng', 'ORANGE', 'Orange')
    filtered_dataset = replace_values(filtered_dataset, 'des_agrup_color_eng', 'PURPLE', 'Purple')

    # Save the filtered dataset to a new CSV file
    filtered_dataset.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)

    print(f"Filtered dataset saved to {output_file}")

if __name__ == "__main__":
    output_file_path = os.path.join(directorio_cwd, "datathon", "dataset", "filtered_product_data.csv")
    filter_and_save_dataset(output_file_path)
