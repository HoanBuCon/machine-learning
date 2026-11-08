# Biên dịch báo cáo: xelatex -> biber -> xelatex x2 (cập nhật mục lục, số hình/bảng, trích dẫn)
# Cách chạy (PowerShell, trong thư mục bao_cao):  .\build.ps1
Set-Location $PSScriptRoot
xelatex -interaction=nonstopmode main.tex | Out-Null
biber main | Out-Null
xelatex -interaction=nonstopmode main.tex | Out-Null
xelatex -interaction=nonstopmode main.tex | Out-Null
$errors = Select-String -Path main.log -Pattern '^! ' -Context 0,3
if ($errors) {
    Write-Host "CO LOI - xem main.log:" -ForegroundColor Red
    $errors | Select-Object -First 5 | ForEach-Object { Write-Host $_ }
} else {
    Write-Host "Xong: main.pdf" -ForegroundColor Green
}
