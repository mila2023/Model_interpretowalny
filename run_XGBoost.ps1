Write-Host "Tworzenie srodowiska wirtualnego..."
python -m venv venv

Write-Host "Instalacja zaleznosci..."
.\venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Uruchamianie analizy modelu..."
.\venv\Scripts\python.exe run_XGBoost_analysis.py


function Clean-Artifacts {
    Write-Host "Rozpoczynanie czyszczenia..."
    
    if (Test-Path ".\venv") {
        Write-Host "Usuwam folder venv..."
        Remove-Item -Recurse -Force venv
    } else {
        Write-Host "Folder venv nie istnieje."
    }

    if (Test-Path ".\XGBoost_model_pipeline.pkl") {
        Write-Host "Usuwam zapisany model (.pkl)..."
        Remove-Item ".\XGBoost_model_pipeline.pkl"
    }
    
    Write-Host "Czyszczenie zakonczone."
}

Clean-Artifacts

Write-Host "Zakonczono."