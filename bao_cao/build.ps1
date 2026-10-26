# Biên dịch báo cáo: pdflatex -> biber -> pdflatex x2 (đủ để cập nhật mục lục, số bảng/hình, trích dẫn)
# Cách chạy (PowerShell, trong thư mục bao_cao):  .\build.ps1
Set-Location $PSScriptRoot
pdflatex -interaction=nonstopmode main.tex | Out-Null
biber main | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null

$errors = Select-String -Path main.log -Pattern '^! ' -Context 0,3
if ($errors) {
    Write-Host "CO LOI - xem chi tiet trong main.log:" -ForegroundColor Red
    $errors | Select-Object -First 5 | ForEach-Object { Write-Host $_ }
} else {
    Write-Host "Xong: main.pdf" -ForegroundColor Green
}
