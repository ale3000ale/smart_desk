@echo off
REM Setup script per Windows - Crea env, attiva, aggiorna pip e installa dipendenze

echo ========================================
echo Setup Progetto Smart Desk - Windows
echo ========================================
echo.

REM Verifica se requirements.txt esiste
if not exist requirements.txt (
    echo ERRORE: requirements.txt non trovato!
    echo Assicurati di essere nella cartella giusta e che il file esista.
    pause
    exit /b 1
)

REM Crea ambiente virtuale con Python 3.11
echo [1/4] Creazione ambiente virtuale con Python 3.11...
py -3.11 -m venv env
if errorlevel 1 (
    echo ERRORE: Creazione env fallita. Verifica che Python 3.11 sia installato.
    echo Prova con: py -0
    pause
    exit /b 1
)
echo ✓ Ambiente virtuale creato
echo.

REM Attiva ambiente virtuale
echo [2/4] Attivazione ambiente virtuale...
call env\Scripts\activate.bat
if errorlevel 1 (
    echo ERRORE: Attivazione fallita.
    pause
    exit /b 1
)
echo ✓ Ambiente attivato
echo.

REM Aggiorna pip
echo [3/4] Aggiornamento pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo AVVISO: Aggiornamento pip non riuscito, continuo comunque...
)
echo ✓ Pip aggiornato
echo.

REM Installa dipendenze
echo [4/4] Installazione dipendenze da requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERRORE: Installazione dipendenze fallita.
    pause
    exit /b 1
)
echo ✓ Dipendenze installate
echo.

echo ========================================
echo ✓ Setup completato con successo!
echo ========================================
echo.
echo Ambiente attivato. Digita 'deactivate' per disattivare.
echo.
pause
