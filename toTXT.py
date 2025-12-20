import os
from pathlib import Path
from typing import List, Optional

def merge_python_files(
    project_root: str,
    exclude_files: List[str],
    output_file: str = "merged_output.txt"
) -> None:
    """
    Raccoglie tutti i file .py nel progetto (escludendo quelli nella lista),
    e crea un unico file .txt con il contenuto di tutti i file separati da trattini.
    
    Args:
        project_root: Percorso radice del progetto
        exclude_files: Lista di nomi file/cartelle da escludere (es: ['test.py', 'tests/', '__pycache__'])
        output_file: Nome del file di output (default: 'merged_output.txt')
    
    Example:
        merge_python_files(
            project_root='.',
            exclude_files=['test.py', 'tests/', 'migrations/', '.venv', '__pycache__'],
            output_file='project_files.txt'
        )
    """
    
    python_files = []
    project_path = Path(project_root).resolve()
    
    # Normalizzare i pattern di esclusione
    exclude_patterns = []
    for pattern in exclude_files:
        # Rimuovere slash finali per normalizzazione
        normalized = pattern.rstrip('/')
        exclude_patterns.append(normalized)
    
    # Raccogliere tutti i file .py
    for py_file in project_path.rglob('*.py'):
        relative_path = py_file.relative_to(project_path)
        relative_path_str = str(relative_path).replace('\\', '/')  # Normalizzare per Windows
        
        # Controllare se il file deve essere escluso
        should_exclude = False
        for exclude_pattern in exclude_patterns:
            # Escludere se il pattern matcha il nome file o una cartella nel percorso
            if (exclude_pattern == relative_path.name or  # Nome file esatto
                exclude_pattern in relative_path_str or    # Cartella nel percorso
                any(part == exclude_pattern for part in relative_path.parts)):  # Parte del percorso
                should_exclude = True
                break
        
        if not should_exclude:
            python_files.append((py_file, relative_path))
    
    # Ordinare i file per path
    python_files.sort(key=lambda x: str(x[1]))
    
    # Creare il file di output
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for py_file, relative_path in python_files:
            # Scrivere il path del file
            outfile.write(f"\n{'='*80}\n")
            outfile.write(f"FILE: {relative_path}\n")
            outfile.write(f"{'='*80}\n\n")
            
            # Scrivere il contenuto del file
            try:
                with open(py_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write(f"\n\n{'-'*80}\n\n")
            except Exception as e:
                outfile.write(f"[ERRORE: Impossibile leggere il file - {e}]\n\n")
    
    print(f"✓ Merge completato!")
    print(f"✓ File creato: {output_file}")
    print(f"✓ File processati: {len(python_files)}")


# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

if __name__ == "__main__":
    # Definisci i file/cartelle da escludere
    exclude_list = [
        'test.py',           # Esclude il file test.py
        'tests',             # Esclude l'intera cartella tests/
        'migrations',        # Esclude l'intera cartella migrations/
        '__pycache__',       # Esclude l'intera cartella __pycache__/
        '.venv',             # Esclude l'intera cartella .venv/
        'venv',              # Esclude l'intera cartella venv/
        'node_modules',      # Esclude l'intera cartella node_modules/
        'tree.py',          # Esclude il file setup.py
        'toTXT.py',         # Esclude questo file di script
        '.vscode',            # Esclude l'intera cartella .vscode/
		'.vs',				# Esclude l'intera cartella .vs/
        'env',               # Esclude l'intera cartella env/
        
    ]
    
    # Esegui la funzione
    merge_python_files(
        project_root='.',                    # Cartella radice del progetto
        exclude_files=exclude_list,
        output_file='project_code.txt'       # Nome file output
    )
