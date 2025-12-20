#!/bin/bash

# Setup script per macOS e Linux - Crea env, attiva, aggiorna pip e installa dipendenze

echo "========================================"
echo "Setup Progetto Smart Desk - macOS/Linux"
echo "========================================"
echo ""

# Verifica il sistema operativo
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PYTHON_CMD="python3.11"
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PYTHON_CMD="python3.11"
    OS="macOS"
else
    echo "Sistema operativo non supportato."
    exit 1
fi

echo "Sistema rilevato: $OS"
echo ""

# Verifica se requirements.txt esiste
if [ ! -f requirements.txt ]; then
    echo "ERRORE: requirements.txt non trovato!"
    echo "Assicurati di essere nella cartella giusta e che il file esista."
    exit 1
fi

# Crea ambiente virtuale con Python 3.11
echo "[1/4] Creazione ambiente virtuale con Python 3.11..."
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "ERRORE: Python 3.11 non trovato nel PATH."
    echo "Prova: which $PYTHON_CMD"
    exit 1
fi

$PYTHON_CMD -m venv env
if [ $? -ne 0 ]; then
    echo "ERRORE: Creazione env fallita."
    exit 1
fi
echo "✓ Ambiente virtuale creato"
echo ""

# Attiva ambiente virtuale
echo "[2/4] Attivazione ambiente virtuale..."
source env/bin/activate
if [ $? -ne 0 ]; then
    echo "ERRORE: Attivazione fallita."
    exit 1
fi
echo "✓ Ambiente attivato"
echo ""

# Aggiorna pip
echo "[3/4] Aggiornamento pip..."
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "AVVISO: Aggiornamento pip non riuscito, continuo comunque..."
fi
echo "✓ Pip aggiornato"
echo ""

# Installa dipendenze
echo "[4/4] Installazione dipendenze da requirements.txt..."
python -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERRORE: Installazione dipendenze fallita."
    exit 1
fi
echo "✓ Dipendenze installate"
echo ""

echo "========================================"
echo "✓ Setup completato con successo!"
echo "========================================"
echo ""
echo "Ambiente attivato. Digita 'deactivate' per disattivare."
echo ""
