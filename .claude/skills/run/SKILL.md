# run

description: Launch the Appetite Engineering Streamlit app on localhost:8501

## Steps

1. Kill any existing process on port 8501:
```powershell
$proc = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($proc) { Stop-Process -Id $proc -Force -ErrorAction SilentlyContinue; Start-Sleep -Milliseconds 500 }
```

2. Delete stale pyc cache (prevents ImportError after code changes):
```powershell
Remove-Item -Recurse -Force "C:\Users\Eliad\OneDrive - Technion\Documents\GitHub\eliad-gideon\src\__pycache__" -ErrorAction SilentlyContinue
```

3. Start Streamlit in the background:
```powershell
Start-Process -FilePath "python" -ArgumentList "-m","streamlit","run","app.py","--server.port","8501","--server.headless","true" -WorkingDirectory "C:\Users\Eliad\OneDrive - Technion\Documents\GitHub\eliad-gideon" -WindowStyle Hidden
```

4. Wait 4 seconds, then verify the server is up:
```powershell
Start-Sleep -Seconds 4
$r = Invoke-WebRequest -Uri "http://localhost:8501" -TimeoutSec 10 -UseBasicParsing
Write-Host "HTTP $($r.StatusCode)"
```

5. Tell the user the app is running at **http://localhost:8501** and they can open it in their browser.