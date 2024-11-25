import os

# Liste des fichiers et dossiers à ignorer
ignored_files = ['tree1.py', 'output.txt', 'LICENSE', '.gitignore', 'package-lock.json']
ignored_dirs = ['__pycache__', 'front', 'api', 'icons', '.git', 'node_modules']


def generate_directory_tree(directory, f_out, prefix=""):
    """
    Fonction pour générer un plan d'arborescence en respectant les exclusions.
    """
    items = sorted(os.listdir(directory))
    items = [item for item in items if item not in ignored_files and item not in ignored_dirs]

    for i, item in enumerate(items):
        path = os.path.join(directory, item)
        is_last = i == len(items) - 1
        connector = "└───" if is_last else "├───"
        f_out.write(f"{prefix}{connector} {item}\n")
        
        if os.path.isdir(path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            generate_directory_tree(path, f_out, new_prefix)


def list_files_recursively(directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Génère le plan d'architecture au début
        f_out.write("Architecture du projet :\n\n")
        generate_directory_tree(directory, f_out)
        f_out.write("\n\nContenu des fichiers :\n\n")
        
        for root, dirs, files in os.walk(directory):
            # Ignore les dossiers spécifiés
            dirs[:] = [d for d in dirs if d not in ignored_dirs]

            for file in files:
                # Ignore les fichiers spécifiés
                if file not in ignored_files:
                    relative_path = os.path.relpath(os.path.join(root, file), directory)
                    f_out.write(f"fichier {relative_path}: \n")

                    # Lire et écrire le contenu du fichier
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f_in:
                            content = f_in.read()
                            f_out.write(f"{content}\n\n")
                    except Exception as e:
                        f_out.write(f"Erreur de lecture du fichier: {e}\n\n")


# Exécution du script
if __name__ == "__main__":
    current_dir = os.getcwd()  
    output_filename = "output.txt" 
    list_files_recursively(current_dir, output_filename)
