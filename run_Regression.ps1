Write-Host "Tworzenie srodowiska wirtualnego..."
python -m venv venv

Write-Host "Instalacja zaleznosci..."
.\venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Uruchamianie analizy modelu..."
.\venv\Scripts\python.exe run_Regression_analysis.py


function Clean-Artifacts {
    Write-Host "Rozpoczynanie czyszczenia..."
    
    if (Test-Path ".\venv") {
        Write-Host "Usuwam folder venv..."
        Remove-Item -Recurse -Force venv
    } else {
        Write-Host "Folder venv nie istnieje."
    }
    
    Write-Host "Czyszczenie zakonczone."
}

Clean-Artifacts

Write-Host "Zakonczono."


# ZMIANA PONIŻEJ: Dodano znak # aby wyłączyć czyszczenie
# Clean-Artifacts 

Write-Host "Zakonczono. Model .pkl oraz srodowisko venv zostaly zachowane."
# Opcjonalnie: Zatrzymaj okno, żebyś widział wyniki
Read-Host "Nacisnij Enter, aby zamknac..."
