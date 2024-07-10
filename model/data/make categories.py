import os


# Function to create the directory structure
def create_directory_structure(data_file, output_dir,level):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read data from the file
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Process each line
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split('"')
            if len(parts) >= 3:
                title = parts[1].strip()
                text = parts[3].strip()

                category_dir = os.path.join(output_dir, level)
                os.makedirs(category_dir, exist_ok=True)

                # Assume file names are generated or use some ID
                file_name = f"file_{len(os.listdir(category_dir)) + 1}.txt"

                # Write to file
                with open(os.path.join(category_dir, file_name), 'w', encoding='utf-8') as fw:
                    fw.write(f'{title}\n{text}')


# Example usage
if __name__ == "__main__":
    output_dir = "categories"  # Replace with your desired output directory

    data_file = "level1.txt"  # Replace with your actual data file name
    create_directory_structure(data_file, output_dir, 'category1')

    data_file = "level2.txt"  # Replace with your actual data file name
    create_directory_structure(data_file, output_dir, 'category2')

    data_file = "level3.txt"  # Replace with your actual data file name
    create_directory_structure(data_file, output_dir, 'category3')

    data_file = "level4.txt"  # Replace with your actual data file name
    create_directory_structure(data_file, output_dir,'category4')