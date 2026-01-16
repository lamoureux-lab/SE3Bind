import os


def create_project_structure():
    """
    set up directories for project
    """
    # Define the directory structure
    directories = ['src', 'tests', 'data', 'models']
    files = ['environment.yml', 'requirements.txt', 'README.md', '.gitignore']

    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Create files
    for file in files:
        with open(file, 'w') as f:
            f.write('')


if __name__ == "__main__":
    create_project_structure()
    print("Project structure created successfully.")
