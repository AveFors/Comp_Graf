# ========================================================================
# Сборка проекта; -Run также запускает готовую программу
# ========================================================================
param([switch]$Run)
$ErrorActionPreference = 'Stop'

# ========================================================================
# Поиск MSVC, CMake и Ninja в Visual Studio Build Tools
# ========================================================================
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (!(Test-Path -LiteralPath $vswhere)) { throw 'Visual Studio Build Tools is required.' }
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$installation) { throw 'Install the Desktop development with C++ workload in Build Tools.' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$cmake = Join-Path $installation 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ninja = Join-Path $installation 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
foreach ($tool in @($vcvars, $cmake, $ninja)) {
    if (!(Test-Path -LiteralPath $tool)) { throw "Required tool not found: $tool" }
}

# ========================================================================
# Подготовка окружения MSVC и сборка через CMake + Ninja
# ========================================================================
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force -Path 'build-debug' | Out-Null
    # После переноса папки CMake должен заново настроить прежний каталог сборки.
    $fresh = ''
    $cache = Join-Path $PSScriptRoot 'build-debug\CMakeCache.txt'
    if (Test-Path -LiteralPath $cache) {
        $homeEntry = Get-Content -LiteralPath $cache | Where-Object { $_ -like 'CMAKE_HOME_DIRECTORY:INTERNAL=*' }
        if ($homeEntry -and $homeEntry.Substring($homeEntry.IndexOf('=') + 1) -ne ($PSScriptRoot -replace '\\','/')) { $fresh = '--fresh' }
    }
    $commandFile = Join-Path $PSScriptRoot 'build-debug\build-local.cmd'
    @(
        '@echo off'
        "call `"$vcvars`""
        'if errorlevel 1 exit /b 1'
        "`"$cmake`" $fresh -S . -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=`"$ninja`""
        'if errorlevel 1 exit /b 1'
        "`"$cmake`" --build build-debug --parallel 4"
        'if errorlevel 1 exit /b 1'
    ) | Set-Content -LiteralPath $commandFile -Encoding oem
    & $env:ComSpec /d /c $commandFile
    if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }

    # --- Запуск после успешной сборки ---
    if ($Run) { & (Join-Path $PSScriptRoot 'build-debug\vulkan-starter-app.exe') }
} finally { Pop-Location }
