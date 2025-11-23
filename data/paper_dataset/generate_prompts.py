import json
import os

def create_prompts():
    """
    Reads a JSON file with paper data and a prompt template to generate
    individual prompt files for each paper.
    """
    # --- Configuration ---
    # Define the paths to your data file, template, and output directory.
    data_file_path = os.path.join('paper_dataset', 'paperQA.json')
    template_file_path = os.path.join('paper_dataset', 'QAprompt.txt')
    output_dir = os.path.join('paper_dataset', 'generated_prompts')

    # --- Pre-run Checks ---
    # Check if the necessary files and folders exist.
    if not os.path.exists('paper_dataset'):
        print("Error: The 'paper_dataset' directory was not found.")
        print("Please make sure your 'paperQA.json' and 'QAprompt.txt' are inside a folder named 'paper_dataset'.")
        return
        
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at '{data_file_path}'")
        return

    if not os.path.exists(template_file_path):
        print(f"Error: Prompt template not found at '{template_file_path}'")
        return

    # --- Main Logic ---
    # Create the output directory if it doesn't already exist.
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Read the main prompt template from the text file.
        with open(template_file_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        # Read the paper data from the JSON file.
        with open(data_file_path, 'r', encoding='utf-8') as f:
            papers_data = json.load(f)

        # Loop through each paper entry in your dataset.
        for paper in papers_data:
            # Safely get data, providing defaults if a key is missing.
            paper_id = paper.get('id', 'unknown_id')
            title = paper.get('title', 'No Title Provided')
            introduction = paper.get('introduction', 'No Introduction Provided')

            # Format the text that will replace the placeholder.
            # Including the title provides more context to the LLM.
            text_to_analyze = f"Title: {title}\n\n{introduction}"

            # Replace the placeholder in the template with the paper's specific text.
            final_prompt = prompt_template.replace('[INSERT INTRODUCTION TEXT HERE]', text_to_analyze)

            # Create a unique, descriptive file name for each prompt.
            output_file_name = f'prompt_for_paper_{paper_id}.txt'
            output_file_path = os.path.join(output_dir, output_file_name)

            # Save the completed prompt to its new file.
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(final_prompt)
            
            print(f"Successfully created prompt for paper ID {paper_id}: {output_file_path}")

        print(f"\nProcess complete. All {len(papers_data)} prompts have been generated in the '{output_dir}' folder.")

    except json.JSONDecodeError:
        print(f"Error: The file '{data_file_path}' is not a valid JSON file. Please check its format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    create_prompts()
