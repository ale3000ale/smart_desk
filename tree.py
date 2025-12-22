import os
import fnmatch

def parse_gitignore(gitignore_path):
   # Legge il file .gitignore e ritorna una lista di pattern da ignorare
    ignore_patterns = {'.git', '__pycache__', 'generate_tree.py', 'structure.txt', 'tree.py' } # Default obbligatori
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Salta commenti e linee vuote
                if not line or line.startswith('#'):
                    continue
                # Rimuovi slash iniziali/finali per match più generico
                clean_pattern = line.strip('/')
                ignore_patterns.add(clean_pattern)
    return list(ignore_patterns)

def should_ignore(name, patterns):
    """Controlla se un file/cartella corrisponde a uno dei pattern."""
    for pattern in patterns:
        # Match esatto o wildcard
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name, pattern + '*'):
            return True
    return False

def generate_tree(startpath, output_file):
    ignore_list = parse_gitignore('.gitignore')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Intestazione opzionale
        f.write(f"Project Structure (Ignored from .gitignore)\n")
        f.write(f"===========================================\n\n")
        
        for root, dirs, files in os.walk(startpath):
            # Filtra le cartelle IN-PLACE per non attraversarle
            dirs[:] = [d for d in dirs if not should_ignore(d, ignore_list)]
            
            level = root.replace(startpath, '').count(os.sep)
            indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
            
            if level == 0:
                f.write(f'{os.path.basename(os.getcwd())}/\n')
            else:
                f.write(f'{indent}{os.path.basename(root)}/\n')
            
            subindent = '│   ' * level + '├── '
            for filename in files:
                if not should_ignore(filename, ignore_list):
                    f.write(f'{subindent}{filename}\n')

if __name__ == "__main__":
    generate_tree('.', 'structure.txt')
    print("✅ File 'structure.txt' creato usando le regole del .gitignore!")
