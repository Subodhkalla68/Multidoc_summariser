import os

def load_and_merge_documents(data_dir, separator="<region-sep>"):
    """
    Load all text files from a directory and merge them into a single string, 
    with each document prefixed by a region-specific tag.
    """
    documents = []
    for file_name in sorted(os.listdir(data_dir)):
        if file_name.endswith(".txt"):
            region = f"<region-{len(documents)+1}>"  # Add a region tag
            with open(os.path.join(data_dir, file_name), 'r') as file:
                documents.append(f"{region} {file.read().strip()}")
    return f" {separator} ".join(documents)

# Example usage
if __name__ == "__main__":
    merged_text = load_and_merge_documents("data")
    print("Merged Text:")
    print(merged_text)