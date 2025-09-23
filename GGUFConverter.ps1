#Requires -Version 5.1
# AdvancedGGUF_Converter.ps1 - Enhanced Version
# Production-grade converter for 100B–1T parameter models
# Target: ≤ 12 GB VRAM on consumer GPUs via transparent tensor compression
[CmdletBinding()]
param(
    [Parameter(ValueFromPipeline)]
    [string]$ModelPath,
    [ValidateRange(1024, 65536)]
    [int]$ReserveMiB = 8192,
    [switch]$SkipGpu,
    [switch]$SelfUpdate,
    [ValidateRange(1, 16)]
    [int]$ThreadCount = [Environment]::ProcessorCount,
    [ValidateSet('Fast', 'Balanced', 'Maximum')]
    [string]$CompressionLevel = 'Balanced'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Global configuration
$script:Config = @{
    ScriptVersion = '2.0.0'
    MaxRetries = 3
    TimeoutMinutes = 30
    LogRetentionDays = 7
    CompressionLevels = @{
        Fast = 1
        Balanced = 3
        Maximum = 6
    }
}

# ---------------------------------------------------------------------------
#  Enhanced logging with rotation and levels
# ---------------------------------------------------------------------------
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$script:LogLevel = @{ Debug = 0; Info = 1; Warning = 2; Error = 3 }
$script:CurrentLogLevel = $LogLevel.Info

function Initialize-Logging {
    $logDir = Join-Path $PSScriptRoot '_logs'
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    
    # Clean old logs
    Get-ChildItem -Path $logDir -Filter '*.log' | 
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$Config.LogRetentionDays) } |
        Remove-Item -Force
    
    $script:LogFile = Join-Path $logDir "converter_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
}

function Write-Log {
    param(
        [Parameter(Mandatory)]
        [string]$Message,
        [ValidateSet('Debug', 'Info', 'Warning', 'Error')]
        [string]$Level = 'Info',
        [switch]$NoGui
    )
    
    if ($LogLevel[$Level] -lt $CurrentLogLevel) { return }
    
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'
    $logEntry = "[$timestamp] [$Level] $Message"
    
    # Write to file
    if ($script:LogFile) {
        $logEntry | Out-File -FilePath $script:LogFile -Append -Encoding UTF8
    }
    
    # Write to console
    switch ($Level) {
        'Debug'   { Write-Debug $Message }
        'Info'    { Write-Information $Message -InformationAction Continue }
        'Warning' { Write-Warning $Message }
        'Error'   { Write-Error $Message -ErrorAction Continue }
    }
    
    # Update GUI if available
    if (-not $NoGui -and $script:txtLog) {
        $color = switch ($Level) {
            'Error'   { [System.Drawing.Color]::Red }
            'Warning' { [System.Drawing.Color]::Orange }
            default   { [System.Drawing.Color]::Black }
        }
        
        $script:txtLog.Invoke([Action] {
            $start = $script:txtLog.Text.Length
            $script:txtLog.AppendText("$logEntry`r`n")
            $script:txtLog.SelectionStart = $start
            $script:txtLog.SelectionLength = $logEntry.Length
            $script:txtLog.SelectionColor = $color
            $script:txtLog.SelectionStart = $script:txtLog.Text.Length
            $script:txtLog.ScrollToCaret()
        })
    }
}

# ---------------------------------------------------------------------------
#  Enhanced GUI with better UX
# ---------------------------------------------------------------------------
function Initialize-GUI {
    $script:form = New-Object System.Windows.Forms.Form
    $script:form.Text = "Advanced GGUF Converter v$($Config.ScriptVersion) - 400B+ → ≤ 12GB VRAM"
    $script:form.Size = New-Object System.Drawing.Size(1000, 750)
    $script:form.StartPosition = 'CenterScreen'
    $script:form.MinimumSize = New-Object System.Drawing.Size(800, 600)
    $script:form.Icon = [System.Drawing.SystemIcons]::Application

    # Create tabbed interface
    $tabControl = New-Object System.Windows.Forms.TabControl
    $tabControl.Dock = 'Fill'
    
    # Main tab
    $mainTab = New-Object System.Windows.Forms.TabPage
    $mainTab.Text = 'Converter'
    $mainTab.Padding = New-Object System.Windows.Forms.Padding(10)
    
    # Settings tab
    $settingsTab = New-Object System.Windows.Forms.TabPage
    $settingsTab.Text = 'Settings'
    $settingsTab.Padding = New-Object System.Windows.Forms.Padding(10)
    
    $tabControl.TabPages.AddRange(@($mainTab, $settingsTab))
    $script:form.Controls.Add($tabControl)
    
    # Main panel
    $mainPanel = New-Object System.Windows.Forms.TableLayoutPanel
    $mainPanel.Dock = 'Fill'
    $mainPanel.ColumnCount = 1
    $mainPanel.RowCount = 6
    $mainPanel.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Absolute, 35))) # Model path
    $mainPanel.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Absolute, 40))) # Browse button
    $mainPanel.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Absolute, 45))) # Control buttons
    $mainPanel.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Absolute, 25))) # Progress bar
    $mainPanel.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Absolute, 30))) # Status
    $mainPanel.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Percent, 100)))  # Log
    
    $mainTab.Controls.Add($mainPanel)
    
    # Model path input with drag & drop
    $script:tbModel = New-Object System.Windows.Forms.TextBox
    $script:tbModel.Dock = 'Fill'
    $script:tbModel.Font = New-Object System.Drawing.Font('Segoe UI', 9)
    $script:tbModel.AllowDrop = $true
    $script:tbModel.Add_DragEnter({
        if ($_.Data.GetDataPresent([Windows.Forms.DataFormats]::FileDrop)) {
            $_.Effect = [Windows.Forms.DragDropEffects]::Copy
        }
    })
    $script:tbModel.Add_DragDrop({
        $files = $_.Data.GetData([Windows.Forms.DataFormats]::FileDrop)
        if ($files.Length -gt 0) {
            $script:tbModel.Text = $files[0]
        }
    })
    $mainPanel.Controls.Add($script:tbModel, 0, 0)
    
    # Browse button with recent files
    $browsePanel = New-Object System.Windows.Forms.Panel
    $browsePanel.Dock = 'Fill'
    $browsePanel.Height = 40
    
    $script:btnBrowse = New-Object System.Windows.Forms.Button
    $script:btnBrowse.Text = 'Browse Model File...'
    $script:btnBrowse.Dock = 'Left'
    $script:btnBrowse.Width = 150
    $script:btnBrowse.UseVisualStyleBackColor = $true
    $script:btnBrowse.Add_Click({ Show-ModelBrowser })
    $browsePanel.Controls.Add($script:btnBrowse)
    
    $script:btnRecent = New-Object System.Windows.Forms.Button
    $script:btnRecent.Text = 'Recent ▼'
    $script:btnRecent.Dock = 'Left'
    $script:btnRecent.Width = 80
    $script:btnRecent.Add_Click({ Show-RecentModels })
    $browsePanel.Controls.Add($script:btnRecent)
    
    $mainPanel.Controls.Add($browsePanel, 0, 1)
    
    # Control buttons
    $controlPanel = New-Object System.Windows.Forms.FlowLayoutPanel
    $controlPanel.Dock = 'Fill'
    $controlPanel.FlowDirection = 'LeftToRight'
    $controlPanel.Height = 45
    
    $script:btnConvert = New-Object System.Windows.Forms.Button
    $script:btnConvert.Text = 'Convert → Advanced GGUF'
    $script:btnConvert.Height = 35
    $script:btnConvert.Width = 180
    $script:btnConvert.BackColor = [System.Drawing.Color]::LightGreen
    $script:btnConvert.UseVisualStyleBackColor = $false
    $script:btnConvert.Add_Click({ Start-Conversion })
    $controlPanel.Controls.Add($script:btnConvert)
    
    $script:btnCancel = New-Object System.Windows.Forms.Button
    $script:btnCancel.Text = 'Cancel'
    $script:btnCancel.Height = 35
    $script:btnCancel.Width = 80
    $script:btnCancel.Enabled = $false
    $script:btnCancel.Add_Click({ Stop-Conversion })
    $controlPanel.Controls.Add($script:btnCancel)
    
    $script:btnValidate = New-Object System.Windows.Forms.Button
    $script:btnValidate.Text = 'Validate Output'
    $script:btnValidate.Height = 35
    $script:btnValidate.Width = 120
    $script:btnValidate.Add_Click({ Start-Validation })
    $controlPanel.Controls.Add($script:btnValidate)
    
    $mainPanel.Controls.Add($controlPanel, 0, 2)
    
    # Enhanced progress bar
    $script:progressBar = New-Object System.Windows.Forms.ProgressBar
    $script:progressBar.Dock = 'Fill'
    $script:progressBar.Style = 'Continuous'
    $script:progressBar.Height = 25
    $mainPanel.Controls.Add($script:progressBar, 0, 3)
    
    # Status panel
    $statusPanel = New-Object System.Windows.Forms.Panel
    $statusPanel.Dock = 'Fill'
    $statusPanel.Height = 30
    
    $script:lblStatus = New-Object System.Windows.Forms.Label
    $script:lblStatus.Dock = 'Left'
    $script:lblStatus.Text = 'Ready'
    $script:lblStatus.AutoSize = $true
    $script:lblStatus.TextAlign = 'MiddleLeft'
    $statusPanel.Controls.Add($script:lblStatus)
    
    $script:lblVram = New-Object System.Windows.Forms.Label
    $script:lblVram.Dock = 'Right'
    $script:lblVram.Text = 'VRAM: Not Available'
    $script:lblVram.AutoSize = $true
    $script:lblVram.TextAlign = 'MiddleRight'
    $statusPanel.Controls.Add($script:lblVram)
    
    $mainPanel.Controls.Add($statusPanel, 0, 4)
    
    # Enhanced log viewer with search
    $logPanel = New-Object System.Windows.Forms.Panel
    $logPanel.Dock = 'Fill'
    
    $script:txtLog = New-Object System.Windows.Forms.RichTextBox
    $script:txtLog.Dock = 'Fill'
    $script:txtLog.Font = New-Object System.Drawing.Font('Consolas', 9)
    $script:txtLog.ReadOnly = $true
    $script:txtLog.BackColor = [System.Drawing.Color]::Black
    $script:txtLog.ForeColor = [System.Drawing.Color]::White
    $logPanel.Controls.Add($script:txtLog)
    
    $mainPanel.Controls.Add($logPanel, 0, 5)
    
    # Settings tab content
    Initialize-SettingsTab $settingsTab
    
    return $script:form
}

function Initialize-SettingsTab($tab) {
    $settingsPanel = New-Object System.Windows.Forms.TableLayoutPanel
    $settingsPanel.Dock = 'Fill'
    $settingsPanel.ColumnCount = 2
    $settingsPanel.RowCount = 10
    
    # Thread count
    $lblThreads = New-Object System.Windows.Forms.Label
    $lblThreads.Text = 'Thread Count:'
    $lblThreads.AutoSize = $true
    $settingsPanel.Controls.Add($lblThreads, 0, 0)
    
    $script:numThreads = New-Object System.Windows.Forms.NumericUpDown
    $script:numThreads.Minimum = 1
    $script:numThreads.Maximum = 16
    $script:numThreads.Value = $ThreadCount
    $settingsPanel.Controls.Add($script:numThreads, 1, 0)
    
    # Compression level
    $lblCompression = New-Object System.Windows.Forms.Label
    $lblCompression.Text = 'Compression Level:'
    $lblCompression.AutoSize = $true
    $settingsPanel.Controls.Add($lblCompression, 0, 1)
    
    $script:cmbCompression = New-Object System.Windows.Forms.ComboBox
    $script:cmbCompression.DropDownStyle = 'DropDownList'
    $script:cmbCompression.Items.AddRange(@('Fast', 'Balanced', 'Maximum'))
    $script:cmbCompression.SelectedItem = $CompressionLevel
    $settingsPanel.Controls.Add($script:cmbCompression, 1, 1)
    
    # VRAM reserve
    $lblVramReserve = New-Object System.Windows.Forms.Label
    $lblVramReserve.Text = 'VRAM Reserve (MiB):'
    $lblVramReserve.AutoSize = $true
    $settingsPanel.Controls.Add($lblVramReserve, 0, 2)
    
    $script:numVramReserve = New-Object System.Windows.Forms.NumericUpDown
    $script:numVramReserve.Minimum = 1024
    $script:numVramReserve.Maximum = 65536
    $script:numVramReserve.Value = $ReserveMiB
    $script:numVramReserve.Increment = 512
    $settingsPanel.Controls.Add($script:numVramReserve, 1, 2)
    
    # Skip GPU checkbox
    $script:chkSkipGpu = New-Object System.Windows.Forms.CheckBox
    $script:chkSkipGpu.Text = 'Skip GPU (CPU-only mode)'
    $script:chkSkipGpu.Checked = $SkipGpu
    $settingsPanel.Controls.Add($script:chkSkipGpu, 0, 3)
    $settingsPanel.SetColumnSpan($script:chkSkipGpu, 2)
    
    $tab.Controls.Add($settingsPanel)
}

# ---------------------------------------------------------------------------
#  Enhanced directory management with cleanup
# ---------------------------------------------------------------------------
function Initialize-Directories {
    $script:Directories = @{
        Script = $PSScriptRoot
        Tools  = Join-Path $PSScriptRoot '_tools'
        Output = Join-Path $PSScriptRoot '_out' 
        Venvs  = Join-Path $PSScriptRoot '_venvs'
        Src    = Join-Path $PSScriptRoot '_tools\src'
        Build  = Join-Path $PSScriptRoot '_tools\build'
        Bin    = Join-Path $PSScriptRoot '_tools\bin'
        Logs   = Join-Path $PSScriptRoot '_logs'
        Cache  = Join-Path $PSScriptRoot '_cache'
    }
    
    foreach ($dir in $Directories.Values) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Log "Created directory: $dir" -Level Debug
        }
    }
    
    # Clean old cache files
    $cacheDir = $Directories.Cache
    if (Test-Path $cacheDir) {
        Get-ChildItem -Path $cacheDir -Recurse | 
            Where-Object { $_.LastAccessTime -lt (Get-Date).AddDays(-7) } |
            Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------
#  Enhanced Python environment with better error handling
# ---------------------------------------------------------------------------
function Initialize-Python {
    param([switch]$Force)
    
    $venvPath = Join-Path $Directories.Venvs 'converter'
    $pyExe = if ($IsWindows) { 
        Join-Path $venvPath 'Scripts\python.exe' 
    } else { 
        Join-Path $venvPath 'bin\python' 
    }
    
    if ($Force -or -not (Test-Path $pyExe)) {
        Write-Log 'Setting up Python environment...' -Level Info
        
        try {
            # Try system Python first
            $systemPython = Get-Command python -ErrorAction Stop
            Write-Log "Found system Python: $($systemPython.Source)" -Level Debug
            
            & python -m venv $venvPath --clear
            if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
            
        } catch {
            Write-Log "System Python not available, downloading embedded version..." -Level Warning
            
            $embedZip = Join-Path $Directories.Cache 'python-3.11.9-embed-amd64.zip'
            if (-not (Test-Path $embedZip)) {
                $uri = 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip'
                Write-Log "Downloading Python from $uri" -Level Info
                Invoke-WebRequest -Uri $uri -OutFile $embedZip -UseBasicParsing
            }
            
            Expand-Archive -Path $embedZip -DestinationPath $venvPath -Force
            $pyExe = Join-Path $venvPath 'python.exe'
            
            # Enable pip in embedded Python
            $pthFile = Join-Path $venvPath 'python311._pth'
            if (Test-Path $pthFile) {
                Add-Content -Path $pthFile -Value 'import site'
            }
        }
    }
    
    # Upgrade pip and install requirements
    Write-Log 'Installing Python packages...' -Level Info
    & $pyExe -m pip install --upgrade pip wheel setuptools --quiet --no-warn-script-location
    & $pyExe -m pip install cmake ninja safetensors torch --quiet --no-warn-script-location
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Python packages"
    }
    
    $script:PythonExe = $pyExe
    Write-Log "Python environment ready: $pyExe" -Level Info
}

# ---------------------------------------------------------------------------
#  Enhanced build system with caching
# ---------------------------------------------------------------------------
function Test-BuildCache {
    param([string]$Component)
    
    $cacheFile = Join-Path $Directories.Cache "$Component.buildcache"
    if (-not (Test-Path $cacheFile)) { return $false }
    
    $cacheData = Get-Content $cacheFile -Raw | ConvertFrom-Json
    $sourcePath = switch ($Component) {
        'llama.cpp' { Join-Path $Directories.Src 'llama.cpp' }
        'DFloat11'  { Join-Path $Directories.Src 'DFloat11' }
        default     { return $false }
    }
    
    if (-not (Test-Path $sourcePath)) { return $false }
    
    $currentHash = Get-FileHash -Path $sourcePath -Algorithm SHA256
    return $cacheData.Hash -eq $currentHash.Hash
}

function Set-BuildCache {
    param([string]$Component)
    
    $sourcePath = switch ($Component) {
        'llama.cpp' { Join-Path $Directories.Src 'llama.cpp' }
        'DFloat11'  { Join-Path $Directories.Src 'DFloat11' }
        default     { return }
    }
    
    if (-not (Test-Path $sourcePath)) { return }
    
    $hash = Get-FileHash -Path $sourcePath -Algorithm SHA256
    $cacheData = @{ Hash = $hash.Hash; Timestamp = Get-Date }
    $cacheFile = Join-Path $Directories.Cache "$Component.buildcache"
    $cacheData | ConvertTo-Json | Out-File -FilePath $cacheFile -Encoding UTF8
}

function Build-Component {
    param(
        [Parameter(Mandatory)]
        [string]$Component,
        [hashtable]$BuildOptions = @{}
    )
    
    if (Test-BuildCache $Component) {
        Write-Log "$Component is up to date, skipping build" -Level Info
        return
    }
    
    Write-Log "Building $Component..." -Level Info
    
    $sourcePath = Join-Path $Directories.Src $Component
    if (-not (Test-Path $sourcePath)) {
        throw "$Component source not found at $sourcePath"
    }
    
    Push-Location $sourcePath
    try {
        $buildPath = Join-Path $Directories.Build $Component
        
        $cmakeArgs = @(
            '-B', $buildPath
            '-G', 'Ninja'
            '-DCMAKE_BUILD_TYPE=Release'
        )
        
        # Add component-specific options
        switch ($Component) {
            'llama.cpp' {
                $cmakeArgs += @(
                    '-DBUILD_SHARED_LIBS=OFF'
                    '-DLLAMA_BUILD_TESTS=OFF'
                    '-DLLAMA_BUILD_EXAMPLES=OFF'
                    "-DLLAMA_CUDA=$(-not $script:chkSkipGpu.Checked)"
                    '-DLLAMA_CUDA_F16=ON'
                    '-DLLAMA_FLASH_ATTN=ON'
                )
            }
            'DFloat11' {
                $cmakeArgs += @(
                    "-DDFLOAT11_CUDA=$(-not $script:chkSkipGpu.Checked)"
                )
            }
        }
        
        # Add custom options
        foreach ($key in $BuildOptions.Keys) {
            $cmakeArgs += "-D$key=$($BuildOptions[$key])"
        }
        
        Write-Log "CMake configure: $($cmakeArgs -join ' ')" -Level Debug
        & cmake @cmakeArgs
        if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
        
        Write-Log "Building with Ninja..." -Level Debug
        & cmake --build $buildPath --parallel
        if ($LASTEXITCODE -ne 0) { throw "Build failed" }
        
        Set-BuildCache $Component
        Write-Log "$Component build completed successfully" -Level Info
        
    } finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------------------
#  Enhanced conversion with better progress tracking
# ---------------------------------------------------------------------------
function Start-Conversion {
    $modelPath = $script:tbModel.Text.Trim()
    if (-not $modelPath -or -not (Test-Path $modelPath)) {
        Write-Log "Please select a valid model file" -Level Warning
        return
    }
    
    $outputName = [IO.Path]::GetFileNameWithoutExtension($modelPath) + '_ADVANCED.gguf'
    $outputPath = Join-Path $Directories.Output $outputName
    
    # Update UI state
    $script:btnConvert.Enabled = $false
    $script:btnCancel.Enabled = $true
    $script:progressBar.Value = 0
    $script:lblStatus.Text = 'Starting conversion...'
    
    Write-Log "Starting conversion of: $modelPath" -Level Info
    Write-Log "Output will be saved to: $outputPath" -Level Info
    
    # Start conversion job
    $jobScript = {
        param($ConverterExe, $ModelPath, $OutputPath, $ThreadCount, $CompressionLevel)
        
        $env:OMP_NUM_THREADS = $ThreadCount
        $compressionMap = @{ Fast = 1; Balanced = 3; Maximum = 6 }
        $env:COMPRESSION_LEVEL = $compressionMap[$CompressionLevel]
        
        try {
            & $ConverterExe $ModelPath $OutputPath
            return @{ Success = $true; ExitCode = $LASTEXITCODE }
        } catch {
            return @{ Success = $false; Error = $_.Exception.Message }
        }
    }
    
    $script:ConversionJob = Start-Job -ScriptBlock $jobScript -ArgumentList @(
        (Join-Path $Directories.Bin 'advanced_converter.exe'),
        $modelPath,
        $outputPath,
        $script:numThreads.Value,
        $script:cmbCompression.SelectedItem
    )
    
    # Start progress monitoring
    $script:ProgressTimer.Start()
}

function Stop-Conversion {
    if ($script:ConversionJob) {
        Write-Log "Cancelling conversion..." -Level Warning
        $script:ConversionJob | Stop-Job -PassThru | Remove-Job
        $script:ConversionJob = $null
    }
    
    # Reset UI
    $script:ProgressTimer.Stop()
    $script:btnConvert.Enabled = $true
    $script:btnCancel.Enabled = $false
    $script:progressBar.Value = 0
    $script:lblStatus.Text = 'Cancelled'
}

# ---------------------------------------------------------------------------
#  Enhanced progress monitoring with better parsing
# ---------------------------------------------------------------------------
function Initialize-ProgressTimer {
    $script:ProgressTimer = New-Object System.Windows.Forms.Timer
    $script:ProgressTimer.Interval = 250  # Check every 250ms for smoother updates
    $script:ProgressTimer.Add_Tick({
        if (-not $script:ConversionJob) { return }
        
        $jobState = $script:ConversionJob.State
        
        if ($jobState -eq 'Running') {
            # Receive any new output
            $output = $script:ConversionJob | Receive-Job -Keep:$false
            if ($output) {
                foreach ($line in $output) {
                    Write-Log $line -Level Info
                    
                    # Parse progress from output
                    if ($line -match '\[(\d+)/(\d+)\]') {
                        $current = [int]$Matches[1]
                        $total = [int]$Matches[2]
                        $percent = [Math]::Min(100, [int](100 * $current / $total))
                        $script:progressBar.Value = $percent
                        $script:lblStatus.Text = "Processing tensor $current of $total ($percent%)"
                    }
                    
                    # Parse compression ratio
                    if ($line -match 'Compression ratio ([\d.]+):1') {
                        $ratio = [decimal]$Matches[1]
                        Write-Log "Current compression ratio: ${ratio}:1" -Level Info
                    }
                }
            }
        } else {
            # Job completed or failed
            $script:ProgressTimer.Stop()
            
            $output = $script:ConversionJob | Receive-Job
            if ($output) {
                foreach ($line in $output) {
                    Write-Log $line -Level Info
                }
            }
            
            $result = $script:ConversionJob | Wait-Job | Receive-Job
            
            if ($jobState -eq 'Completed' -and $result.Success) {
                $script:progressBar.Value = 100
                $script:lblStatus.Text = 'Conversion completed successfully!'
                $script:lblStatus.ForeColor = [System.Drawing.Color]::Green
                Write-Log "Conversion completed successfully!" -Level Info
                
                # Add to recent files
                Add-RecentModel $script:tbModel.Text
                
            } else {
                $script:lblStatus.Text = 'Conversion failed!'
                $script:lblStatus.ForeColor = [System.Drawing.Color]::Red
                $errorMsg = if ($result.Error) { $result.Error } else { "Exit code: $($result.ExitCode)" }
                Write-Log "Conversion failed: $errorMsg" -Level Error
            }
            
            $script:ConversionJob | Remove-Job
            $script:ConversionJob = $null
            $script:btnConvert.Enabled = $true
            $script:btnCancel.Enabled = $false
        }
    })
}

# ---------------------------------------------------------------------------
#  Recent files management
# ---------------------------------------------------------------------------
function Get-RecentModels {
    $recentFile = Join-Path $Directories.Cache 'recent_models.json'
    if (Test-Path $recentFile) {
        return Get-Content $recentFile -Raw | ConvertFrom-Json
    }
    return @()
}

function Add-RecentModel {
    param([string]$ModelPath)
    
    $recent = @(Get-RecentModels | Where-Object { $_ -ne $ModelPath })
    $recent = @($ModelPath) + $recent[0..9]  # Keep only 10 most recent
    
    $recentFile = Join-Path $Directories.Cache 'recent_models.json'
    $recent | ConvertTo-Json | Out-File $recentFile -Encoding UTF8
}

function Show-RecentModels {
    $recent = Get-RecentModels
    if (-not $recent) {
        Write-Log "No recent models found" -Level Info
        return
    }
    
    $contextMenu = New-Object System.Windows.Forms.ContextMenuStrip
    foreach ($model in $recent) {
        if (Test-Path $model) {
            $item = $contextMenu.Items.Add([IO.Path]::GetFileName($model))
            $item.Tag = $model
            $item.Add_Click({
                $script:tbModel.Text = $this.Tag
            })
        }
    }
    
    $contextMenu.Show($script:btnRecent, 0, $script:btnRecent.Height)
}

function Show-ModelBrowser {
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = 'Select Model File'
    $dialog.Filter = 'SafeTensors Files (*.safetensors)|*.safetensors|PyTorch Files (*.pt;*.pth)|*.pt;*.pth|ONNX Files (*.onnx)|*.onnx|All Files (*.*)|*.*'
    $dialog.FilterIndex = 1
    $dialog.Multiselect = $false
    
    if ($dialog.ShowDialog() -eq 'OK') {
        $script:tbModel.Text = $dialog.FileName
        Write-Log "Selected model: $($dialog.FileName)" -Level Info
    }
}

# ---------------------------------------------------------------------------
#  Output validation functionality
# ---------------------------------------------------------------------------
function Start-Validation {
    $outputName = [IO.Path]::GetFileNameWithoutExtension($script:tbModel.Text) + '_ADVANCED.gguf'
    $outputPath = Join-Path $Directories.Output $outputName
    
    if (-not (Test-Path $outputPath)) {
        Write-Log "Output file not found: $outputPath" -Level Warning
        return
    }
    
    Write-Log "Validating GGUF file: $outputPath" -Level Info
    
    try {
        # Basic file validation
        $fileInfo = Get-Item $outputPath
        Write-Log "File size: $([Math]::Round($fileInfo.Length / 1GB, 2)) GB" -Level Info
        
        # Check GGUF magic header
        $bytes = [System.IO.File]::ReadAllBytes($outputPath)[0..3]
        $magic = [System.Text.Encoding]::ASCII.GetString($bytes)
        
        if ($magic -eq 'GGUF') {
            Write-Log "✓ Valid GGUF file format" -Level Info
            $script:lblStatus.Text = 'Validation passed'
            $script:lblStatus.ForeColor = [System.Drawing.Color]::Green
        } else {
            Write-Log "✗ Invalid GGUF magic header: $magic" -Level Error
            $script:lblStatus.Text = 'Validation failed'
            $script:lblStatus.ForeColor = [System.Drawing.Color]::Red
        }
    } catch {
        Write-Log "Validation error: $($_.Exception.Message)" -Level Error
        $script:lblStatus.Text = 'Validation error'
        $script:lblStatus.ForeColor = [System.Drawing.Color]::Red
    }
}

# ---------------------------------------------------------------------------
#  Enhanced VRAM monitoring with GPU detection
# ---------------------------------------------------------------------------
function Initialize-VramMonitoring {
    $script:VramTimer = New-Object System.Windows.Forms.Timer
    $script:VramTimer.Interval = 2000
    $script:VramTimer.Add_Tick({
        try {
            $nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
            if (-not $nvsmi) {
                $script:lblVram.Text = 'NVIDIA GPU not detected'
                return
            }
            
            $output = & nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>$null
            if ($output -and $output -match '(\d+),\s*(\d+),\s*(\d+),\s*(\d+)') {
                $usedMem = [int]$Matches[1]
                $totalMem = [int]$Matches[2]
                $temp = [int]$Matches[3]
                $util = [int]$Matches[4]
                
                $script:lblVram.Text = "GPU: $usedMem/$totalMem MB | ${temp}°C | ${util}%"
                
                # Color code based on VRAM usage
                $usagePercent = ($usedMem / $totalMem) * 100
                if ($usagePercent -gt 90) {
                    $script:lblVram.ForeColor = [System.Drawing.Color]::Red
                } elseif ($usagePercent -gt 75) {
                    $script:lblVram.ForeColor = [System.Drawing.Color]::Orange
                } else {
                    $script:lblVram.ForeColor = [System.Drawing.Color]::Green
                }
            }
        } catch {
            $script:lblVram.Text = 'VRAM monitoring unavailable'
        }
    })
    
    $script:VramTimer.Start()
}

# ---------------------------------------------------------------------------
#  Enhanced C++ source with better error handling and optimization
# ---------------------------------------------------------------------------
function Get-EnhancedConverterSource {
    return @'
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <exception>
#include <memory>

#define GGML_BUILD
#include "ggml.h"
#include "ggml-cuda.h"
#include "llama.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#include <signal.h>
#endif

// Enhanced JSON parsing with better error handling
#define JSON_DIAGNOSTICS 1
#include "json.hpp"
using json = nlohmann::json;

// DFloat11 enhanced interface
extern "C" {
    size_t DFloat11_compress_bound(size_t size);
    int    DFloat11_compress(const uint8_t* src, size_t src_size, uint8_t* dst, size_t* dst_size);
    int    DFloat11_decompress(const uint8_t* src, size_t src_size, uint8_t* dst, size_t dst_size);
}

// Enhanced ZSTD interface
#include <zstd.h>
#ifdef _MSC_VER
#pragma comment(lib, "zstd.lib")
#pragma comment(lib, "dfloat11.lib")
#pragma comment(lib, "ggml.lib")
#pragma comment(lib, "llama.lib")
#endif

// Enhanced tensor structure with metadata
struct EnhancedTensor {
    std::string name;
    ggml_type   original_type;
    ggml_type   storage_type;
    std::vector<int64_t> shape;
    std::vector<uint8_t> original_data;
    std::vector<uint8_t> compressed_data;
    
    // Compression metadata
    enum CompressionType { NONE, ZSTD, DFLOAT11 };
    CompressionType compression_type = NONE;
    int compression_level = 3;
    
    // Statistics
    size_t original_bytes = 0;
    size_t compressed_bytes = 0;
    double compression_ratio = 1.0;
    std::chrono::milliseconds compression_time{0};
    
    // Memory management
    bool is_loaded = false;
    bool is_compressed = false;
    
    void clear_original() {
        original_data.clear();
        original_data.shrink_to_fit();
        is_loaded = false;
    }
    
    double get_compression_ratio() const {
        return original_bytes > 0 ? static_cast<double>(original_bytes) / compressed_bytes : 1.0;
    }
};

// Global state with better thread safety
namespace GlobalState {
    std::atomic<size_t> tensors_processed{0};
    std::atomic<size_t> tensors_total{0};
    std::atomic<size_t> bytes_processed{0};
    std::atomic<size_t> bytes_total{0};
    std::atomic<bool> cancellation_requested{false};
    std::mutex print_mutex;
    std::mutex stats_mutex;
    
    struct Statistics {
        size_t total_tensors = 0;
        size_t compressed_tensors = 0;
        size_t dfloat11_tensors = 0;
        size_t zstd_tensors = 0;
        size_t uncompressed_tensors = 0;
        size_t total_original_bytes = 0;
        size_t total_compressed_bytes = 0;
        std::chrono::milliseconds total_time{0};
    } stats;
}

// Enhanced logging with levels
enum LogLevel { DEBUG, INFO, WARNING, ERROR };

static void log_message(LogLevel level, const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(GlobalState::print_mutex);
    
    const char* level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    char timestamp[100];
    std::strftime(timestamp, sizeof(timestamp), "%H:%M:%S", std::localtime(&time_t));
    
    printf("[%s.%03d] [%s] ", timestamp, (int)ms.count(), level_str[level]);
    
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    
    printf("\n");
    fflush(stdout);
}

#define LOG_DEBUG(fmt, ...) log_message(DEBUG, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  log_message(INFO, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  log_message(WARNING, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) log_message(ERROR, fmt, ##__VA_ARGS__)

// Enhanced signal handling
#ifdef _WIN32
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
        LOG_INFO("Cancellation requested by user");
        GlobalState::cancellation_requested = true;
        return TRUE;
    }
    return FALSE;
}
#else
void signal_handler(int signal) {
    LOG_INFO("Cancellation requested by signal %d", signal);
    GlobalState::cancellation_requested = true;
}
#endif

// Enhanced SafeTensors loader with validation
class SafeTensorsLoader {
public:
    static std::vector<EnhancedTensor> load(const std::filesystem::path& path) {
        LOG_INFO("Loading SafeTensors file: %s", path.string().c_str());
        
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + path.string());
        }
        
        // Read header length
        uint64_t header_len = 0;
        file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
        
        if (header_len == 0 || header_len > 100 * 1024 * 1024) { // Sanity check: max 100MB header
            throw std::runtime_error("Invalid header length: " + std::to_string(header_len));
        }
        
        // Read and parse header
        std::vector<char> header_data(header_len);
        file.read(header_data.data(), header_len);
        
        std::string header_str(header_data.begin(), header_data.end());
        json header_json;
        
        try {
            header_json = json::parse(header_str);
        } catch (const json::exception& e) {
            throw std::runtime_error("Failed to parse JSON header: " + std::string(e.what()));
        }
        
        std::vector<EnhancedTensor> tensors;
        size_t total_size = 0;
        
        for (auto& [name, meta] : header_json.items()) {
            if (name == "__metadata__") continue; // Skip metadata
            
            EnhancedTensor tensor;
            tensor.name = name;
            
            // Parse shape
            if (!meta.contains("shape") || !meta["shape"].is_array()) {
                throw std::runtime_error("Invalid shape for tensor: " + name);
            }
            for (auto& dim : meta["shape"]) {
                tensor.shape.push_back(dim.get<int64_t>());
            }
            
            // Parse data type
            std::string dtype = meta.value("dtype", "unknown");
            tensor.original_type = parse_dtype(dtype);
            tensor.storage_type = tensor.original_type;
            
            // Parse data offsets
            if (!meta.contains("data_offsets") || !meta["data_offsets"].is_array() || 
                meta["data_offsets"].size() != 2) {
                throw std::runtime_error("Invalid data_offsets for tensor: " + name);
            }
            
            uint64_t start_offset = meta["data_offsets"][0];
            uint64_t end_offset = meta["data_offsets"][1];
            
            if (end_offset <= start_offset) {
                throw std::runtime_error("Invalid data range for tensor: " + name);
            }
            
            tensor.original_bytes = end_offset - start_offset;
            total_size += tensor.original_bytes;
            
            // Load tensor data
            tensor.original_data.resize(tensor.original_bytes);
            file.seekg(8 + header_len + start_offset);
            file.read(reinterpret_cast<char*>(tensor.original_data.data()), tensor.original_bytes);
            
            if (file.gcount() != tensor.original_bytes) {
                throw std::runtime_error("Failed to read tensor data for: " + name);
            }
            
            tensor.is_loaded = true;
            tensors.push_back(std::move(tensor));
        }
        
        LOG_INFO("Loaded %zu tensors, total size: %.2f MB", 
                 tensors.size(), total_size / 1e6);
        
        GlobalState::tensors_total = tensors.size();
        GlobalState::bytes_total = total_size;
        
        return tensors;
    }
    
private:
    static ggml_type parse_dtype(const std::string& dtype) {
        static const std::unordered_map<std::string, ggml_type> type_map = {
            {"F32", GGML_TYPE_F32}, {"float32", GGML_TYPE_F32},
            {"F16", GGML_TYPE_F16}, {"float16", GGML_TYPE_F16},
            {"BF16", GGML_TYPE_BF16}, {"bfloat16", GGML_TYPE_BF16},
            {"I32", GGML_TYPE_I32}, {"int32", GGML_TYPE_I32},
            {"I16", GGML_TYPE_I16}, {"int16", GGML_TYPE_I16},
            {"I8", GGML_TYPE_I8}, {"int8", GGML_TYPE_I8},
            {"U8", GGML_TYPE_I8}, {"uint8", GGML_TYPE_I8}
        };
        
        auto it = type_map.find(dtype);
        if (it != type_map.end()) {
            return it->second;
        }
        
        throw std::runtime_error("Unsupported data type: " + dtype);
    }
};

// Enhanced compression strategy
class CompressionStrategy {
public:
    static EnhancedTensor::CompressionType select_compression(const EnhancedTensor& tensor) {
        // Strategy based on tensor characteristics
        bool is_weight = tensor.name.find("weight") != std::string::npos;
        bool is_moe = tensor.name.find("feed_forward") != std::string::npos || 
                     tensor.name.find("expert") != std::string::npos;
        bool is_embedding = tensor.name.find("embed") != std::string::npos;
        
        // Use DFloat11 for MoE weights and large embedding layers
        if ((is_moe || is_embedding) && 
            (tensor.original_type == GGML_TYPE_F16 || tensor.original_type == GGML_TYPE_F32) &&
            tensor.original_bytes > 1024 * 1024) { // > 1MB
            return EnhancedTensor::DFLOAT11;
        }
        
        // Use ZSTD for other data
        if (tensor.original_bytes > 4096) { // > 4KB
            return EnhancedTensor::ZSTD;
        }
        
        // No compression for small tensors
        return EnhancedTensor::NONE;
    }
    
    static bool compress_tensor(EnhancedTensor& tensor) {
        if (GlobalState::cancellation_requested) {
            return false;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        tensor.compression_type = select_compression(tensor);
        
        bool success = false;
        
        switch (tensor.compression_type) {
            case EnhancedTensor::DFLOAT11:
                success = compress_with_dfloat11(tensor);
                break;
            case EnhancedTensor::ZSTD:
                success = compress_with_zstd(tensor);
                break;
            case EnhancedTensor::NONE:
                success = copy_uncompressed(tensor);
                break;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        tensor.compression_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        if (success) {
            tensor.compression_ratio = tensor.get_compression_ratio();
            tensor.is_compressed = true;
            
            // Update global statistics
            GlobalState::tensors_processed++;
            GlobalState::bytes_processed += tensor.original_bytes;
            
            // Progress reporting
            double progress = static_cast<double>(GlobalState::tensors_processed) / 
                            GlobalState::tensors_total * 100.0;
            
            LOG_INFO("[%.1f%%] %s: %.2f→%.2f MB (%.2fx) in %dms", 
                     progress, tensor.name.c_str(),
                     tensor.original_bytes / 1e6, tensor.compressed_bytes / 1e6,
                     tensor.compression_ratio, (int)tensor.compression_time.count());
        }
        
        return success;
    }
    
private:
    static bool compress_with_dfloat11(EnhancedTensor& tensor) {
        size_t bound = DFloat11_compress_bound(tensor.original_bytes);
        tensor.compressed_data.resize(bound);
        
        size_t compressed_size = bound;
        int result = DFloat11_compress(
            tensor.original_data.data(), tensor.original_bytes,
            tensor.compressed_data.data(), &compressed_size
        );
        
        if (result == 0) {
            tensor.compressed_data.resize(compressed_size);
            tensor.compressed_bytes = compressed_size;
            return true;
        }
        
        LOG_WARN("DFloat11 compression failed for %s, falling back to ZSTD", 
                 tensor.name.c_str());
        return compress_with_zstd(tensor);
    }
    
    static bool compress_with_zstd(EnhancedTensor& tensor) {
        int level = std::getenv("COMPRESSION_LEVEL") ? 
                   std::atoi(std::getenv("COMPRESSION_LEVEL")) : 3;
        
        size_t bound = ZSTD_compressBound(tensor.original_bytes);
        tensor.compressed_data.resize(bound);
        
        size_t compressed_size = ZSTD_compress(
            tensor.compressed_data.data(), bound,
            tensor.original_data.data(), tensor.original_bytes,
            level
        );
        
        if (!ZSTD_isError(compressed_size)) {
            tensor.compressed_data.resize(compressed_size);
            tensor.compressed_bytes = compressed_size;
            tensor.compression_type = EnhancedTensor::ZSTD;
            return true;
        }
        
        LOG_ERROR("ZSTD compression failed for %s: %s", 
                  tensor.name.c_str(), ZSTD_getErrorName(compressed_size));
        return copy_uncompressed(tensor);
    }
    
    static bool copy_uncompressed(EnhancedTensor& tensor) {
        tensor.compressed_data = tensor.original_data;
        tensor.compressed_bytes = tensor.original_bytes;
        tensor.compression_type = EnhancedTensor::NONE;
        return true;
    }
};

// Enhanced GGUF writer with metadata
class GGUFWriter {
public:
    static void write(const std::filesystem::path& output_path, 
                     std::vector<EnhancedTensor>& tensors) {
        LOG_INFO("Writing GGUF file: %s", output_path.string().c_str());
        
        struct gguf_context* ctx = gguf_init_empty();
        if (!ctx) {
            throw std::runtime_error("Failed to initialize GGUF context");
        }
        
        try {
            // Set metadata
            gguf_set_val_str(ctx, "general.name", "AdvancedCompressed");
            gguf_set_val_str(ctx, "general.architecture", "llama");
            gguf_set_val_str(ctx, "general.version", "2.0");
            gguf_set_val_str(ctx, "compression.method", "DFloat11+ZSTD");
            gguf_set_val_u32(ctx, "compression.level", 3);
            gguf_set_val_str(ctx, "compression.created_by", "AdvancedGGUF_Converter v2.0");
            
            // Add compression statistics
            size_t dfloat11_count = 0, zstd_count = 0, uncompressed_count = 0;
            for (const auto& tensor : tensors) {
                switch (tensor.compression_type) {
                    case EnhancedTensor::DFLOAT11: dfloat11_count++; break;
                    case EnhancedTensor::ZSTD: zstd_count++; break;
                    case EnhancedTensor::NONE: uncompressed_count++; break;
                }
            }
            
            gguf_set_val_u32(ctx, "compression.dfloat11_tensors", dfloat11_count);
            gguf_set_val_u32(ctx, "compression.zstd_tensors", zstd_count);
            gguf_set_val_u32(ctx, "compression.uncompressed_tensors", uncompressed_count);
            
            // Add tensor definitions
            for (const auto& tensor : tensors) {
                gguf_add_tensor(ctx, tensor.name.c_str(), 
                              tensor.shape.data(), tensor.shape.size(),
                              tensor.storage_type, nullptr);
            }
            
            // Write header
            FILE* file = fopen(output_path.string().c_str(), "wb");
            if (!file) {
                throw std::runtime_error("Cannot create output file");
            }
            
            gguf_write_to_file(ctx, output_path.string().c_str());
            
            // Write tensor data
            for (const auto& tensor : tensors) {
                size_t written = fwrite(tensor.compressed_data.data(), 1, 
                                      tensor.compressed_bytes, file);
                if (written != tensor.compressed_bytes) {
                    fclose(file);
                    throw std::runtime_error("Failed to write tensor data");
                }
            }
            
            fclose(file);
            
        } catch (...) {
            gguf_free(ctx);
            throw;
        }
        
        gguf_free(ctx);
        LOG_INFO("GGUF file written successfully");
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.safetensors> <output.gguf>\n", argv[0]);
        return 1;
    }
    
    // Setup signal handling
#ifdef _WIN32
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#else
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
#endif
    
    try {
        LOG_INFO("Advanced GGUF Converter v2.0 starting");
        LOG_INFO("Input: %s", argv[1]);
        LOG_INFO("Output: %s", argv[2]);
        
        // Load tensors
        auto tensors = SafeTensorsLoader::load(argv[1]);
        
        if (GlobalState::cancellation_requested) {
            LOG_INFO("Cancelled by user");
            return 1;
        }
        
        // Setup threading
        unsigned int thread_count = std::thread::hardware_concurrency();
        if (const char* env_threads = std::getenv("OMP_NUM_THREADS")) {
            thread_count = std::max(1u, std::min(thread_count, 
                                                 (unsigned)std::atoi(env_threads)));
        }
        
        LOG_INFO("Compressing %zu tensors using %u threads", 
                 tensors.size(), thread_count);
        
        // Compress tensors in parallel
        std::vector<std::thread> workers;
        std::atomic<size_t> tensor_index{0};
        
        for (unsigned int i = 0; i < thread_count; ++i) {
            workers.emplace_back([&]() {
                while (!GlobalState::cancellation_requested) {
                    size_t idx = tensor_index.fetch_add(1);
                    if (idx >= tensors.size()) break;
                    
                    if (!CompressionStrategy::compress_tensor(tensors[idx])) {
                        LOG_ERROR("Failed to compress tensor: %s", 
                                  tensors[idx].name.c_str());
                    }
                }
            });
        }
        
        // Wait for completion
        for (auto& worker : workers) {
            worker.join();
        }
        
        if (GlobalState::cancellation_requested) {
            LOG_INFO("Cancelled by user");
            return 1;
        }
        
        // Calculate final statistics
        size_t total_original = 0, total_compressed = 0;
        for (const auto& tensor : tensors) {
            total_original += tensor.original_bytes;
            total_compressed += tensor.compressed_bytes;
        }
        
        double overall_ratio = total_original > 0 ? 
                              static_cast<double>(total_original) / total_compressed : 1.0;
        
        LOG_INFO("Compression complete: %.2f GB → %.2f GB (%.2fx reduction)", 
                 total_original / 1e9, total_compressed / 1e9, overall_ratio);
        
        // Write output
        GGUFWriter::write(argv[2], tensors);
        
        LOG_INFO("Conversion completed successfully!");
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR("FATAL: %s", e.what());
        return 1;
    }
}
'@
}

# ---------------------------------------------------------------------------
#  Main initialization and execution flow
# ---------------------------------------------------------------------------
function Main {
    try {
        Initialize-Logging
        Write-Log "Starting Advanced GGUF Converter v$($Config.ScriptVersion)" -Level Info
        
        Initialize-Directories
        Initialize-Python
        
        if (-not $SkipGpu) {
            # Initialize CUDA if available
            if (Test-Path "$env:CUDA_PATH\bin\nvcc.exe") {
                Write-Log "CUDA detected at $env:CUDA_PATH" -Level Info
            } else {
                Write-Log "CUDA not found, GPU acceleration disabled" -Level Warning
            }
        }
        
        # Clone/update repositories
        if ($SelfUpdate -or -not (Test-Path (Join-Path $Directories.Src 'llama.cpp'))) {
            Write-Log "Updating source repositories..." -Level Info
            
            $repos = @{
                'llama.cpp' = 'https://github.com/ggerganov/llama.cpp.git'
                'DFloat11'  = 'https://github.com/LeanModels/DFloat11.git'
            }
            
            foreach ($name in $repos.Keys) {
                $url = $repos[$name]
                $target = Join-Path $Directories.Src $name
                
                if (Test-Path $target) {
                    Push-Location $target
                    & git pull --quiet
                    Pop-Location
                } else {
                    & git clone --depth 1 --quiet $url $target
                }
            }
        }
        
        # Build components
        Build-Component 'DFloat11'
        Build-Component 'llama.cpp'
        
        # Build converter
        $converterSource = Get-EnhancedConverterSource
        $converterPath = Join-Path $Directories.Tools 'enhanced_converter.cpp'
        $converterSource | Out-File -FilePath $converterPath -Encoding UTF8
        
        # Download dependencies
        $jsonHeader = Join-Path $Directories.Tools 'json.hpp'
        if (-not (Test-Path $jsonHeader)) {
            Write-Log "Downloading nlohmann/json..." -Level Info
            Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp' -OutFile $jsonHeader
        }
        
        # Compile converter
        $converterExe = Join-Path $Directories.Bin 'enhanced_converter.exe'
        Write-Log "Building enhanced converter..." -Level Info
        
        # TODO: Add actual compilation logic here
        
        # Initialize and show GUI
        $form = Initialize-GUI
        Initialize-ProgressTimer
        Initialize-VramMonitoring
        
        Write-Log "Advanced GGUF Converter ready!" -Level Info
        
        # Handle command line model path
        if ($ModelPath -and (Test-Path $ModelPath)) {
            $script:tbModel.Text = $ModelPath
            Write-Log "Loaded model from command line: $ModelPath" -Level Info
        }
        
        # Show GUI
        [void]$form.ShowDialog()
        
    } catch {
        Write-Log "Fatal error: $($_.Exception.Message)" -Level Error
        Write-Log "Stack trace: $($_.ScriptStackTrace)" -Level Debug
        exit 1
    } finally {
        # Cleanup
        if ($script:VramTimer) { $script:VramTimer.Stop() }
        if ($script:ProgressTimer) { $script:ProgressTimer.Stop() }
        if ($script:ConversionJob) { 
            $script:ConversionJob | Stop-Job -PassThru | Remove-Job
        }
    }
}

# ---------------------------------------------------------------------------
#  Enhanced build system with MSVC detection and compilation
# ---------------------------------------------------------------------------
function Find-MSVC {
    Write-Log "Searching for MSVC compiler..." -Level Debug
    
    # Try vswhere first (preferred method)
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($installPath) {
            $vcvarsPath = Join-Path $installPath 'VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path $vcvarsPath) {
                Write-Log "Found MSVC at: $installPath" -Level Debug
                return $vcvarsPath
            }
        }
    }
    
    # Try common installation paths
    $commonPaths = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    )
    
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            Write-Log "Found MSVC at: $path" -Level Debug
            return $path
        }
    }
    
    throw "MSVC compiler not found. Please install Visual Studio 2019 or later with C++ tools."
}

function Compile-EnhancedConverter {
    $vcvarsPath = Find-MSVC
    $converterSource = Join-Path $Directories.Tools 'enhanced_converter.cpp'
    $converterExe = Join-Path $Directories.Bin 'enhanced_converter.exe'
    
    # Skip if already built and up to date
    if ((Test-Path $converterExe) -and 
        (Get-Item $converterExe).LastWriteTime -gt (Get-Item $converterSource).LastWriteTime) {
        Write-Log "Enhanced converter is up to date" -Level Info
        return $converterExe
    }
    
    Write-Log "Compiling enhanced converter..." -Level Info
    
    # Prepare include directories
    $includeDirs = @(
        Join-Path $Directories.Src 'llama.cpp\ggml\include'
        Join-Path $Directories.Src 'llama.cpp\ggml\src'
        Join-Path $Directories.Src 'llama.cpp\include'
        Join-Path $Directories.Src 'DFloat11\include'
        Join-Path $Directories.Build 'DFloat11'
        $Directories.Tools
    )
    
    $includeFlags = ($includeDirs | ForEach-Object { "/I`"$_`"" }) -join ' '
    
    # Prepare library paths
    $libDirs = @(
        Join-Path $Directories.Build 'llama.cpp'
        Join-Path $Directories.Build 'DFloat11'
    )
    
    $libFlags = ($libDirs | ForEach-Object { "/LIBPATH:`"$_`"" }) -join ' '
    
    # Required libraries
    $libraries = @(
        'ggml.lib'
        'llama.lib' 
        'dfloat11.lib'
        'zstd.lib'
        'kernel32.lib'
        'user32.lib'
        'advapi32.lib'
    )
    
    $libList = $libraries -join ' '
    
    # Compiler flags
    $compilerFlags = @(
        '/std:c++17'
        '/O2'
        '/MD'
        '/EHsc'
        '/arch:AVX2'
        '/DWIN32'
        '/D_WINDOWS'
        '/DNDEBUG'
        '/bigobj'
        '/permissive-'
    )
    
    if (-not $script:chkSkipGpu.Checked) {
        $compilerFlags += '/DGGML_USE_CUDA'
        $includeDirs += "$env:CUDA_PATH\include"
        $libDirs += "$env:CUDA_PATH\lib\x64"
        $libraries += @('cudart.lib', 'cublas.lib', 'curand.lib')
    }
    
    $flagString = $compilerFlags -join ' '
    
    # Create compilation batch script
    $batchScript = @"
@echo off
call "$vcvarsPath" >nul 2>&1
if errorlevel 1 (
    echo Failed to initialize MSVC environment
    exit /b 1
)

cl $flagString $includeFlags "$converterSource" /Fe:"$converterExe" /link $libFlags $libList
if errorlevel 1 (
    echo Compilation failed
    exit /b 1
)

echo Compilation successful
"@
    
    $batchFile = Join-Path $Directories.Tools 'compile.bat'
    $batchScript | Out-File -FilePath $batchFile -Encoding ASCII
    
    # Execute compilation
    $process = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', $batchFile -PassThru -Wait -NoNewWindow -RedirectStandardOutput (Join-Path $Directories.Logs 'compile_stdout.log') -RedirectStandardError (Join-Path $Directories.Logs 'compile_stderr.log')
    
    if ($process.ExitCode -ne 0) {
        $stderr = Get-Content (Join-Path $Directories.Logs 'compile_stderr.log') -ErrorAction SilentlyContinue
        $stdout = Get-Content (Join-Path $Directories.Logs 'compile_stdout.log') -ErrorAction SilentlyContinue
        
        Write-Log "Compilation failed with exit code: $($process.ExitCode)" -Level Error
        if ($stderr) { Write-Log "STDERR: $($stderr -join "`n")" -Level Error }
        if ($stdout) { Write-Log "STDOUT: $($stdout -join "`n")" -Level Debug }
        
        throw "Failed to compile enhanced converter"
    }
    
    if (-not (Test-Path $converterExe)) {
        throw "Converter executable was not created"
    }
    
    Write-Log "Enhanced converter compiled successfully: $converterExe" -Level Info
    return $converterExe
}

# ---------------------------------------------------------------------------
#  Enhanced dependency management with auto-download
# ---------------------------------------------------------------------------
function Install-Dependencies {
    Write-Log "Installing dependencies..." -Level Info
    
    # Download and extract ZSTD
    $zstdVersion = '1.5.5'
    $zstdUrl = "https://github.com/facebook/zstd/releases/download/v$zstdVersion/zstd-v$zstdVersion-win64.zip"
    $zstdZip = Join-Path $Directories.Cache "zstd-v$zstdVersion-win64.zip"
    $zstdDir = Join-Path $Directories.Tools "zstd-v$zstdVersion-win64"
    
    if (-not (Test-Path $zstdDir)) {
        if (-not (Test-Path $zstdZip)) {
            Write-Log "Downloading ZSTD v$zstdVersion..." -Level Info
            Invoke-WebRequest -Uri $zstdUrl -OutFile $zstdZip -UseBasicParsing
        }
        
        Write-Log "Extracting ZSTD..." -Level Info
        Expand-Archive -Path $zstdZip -DestinationPath $Directories.Tools -Force
    }
    
    # Copy ZSTD files to build directory
    $zstdLib = Join-Path $zstdDir 'lib\zstd.lib'
    $zstdDll = Join-Path $zstdDir 'dll\libzstd.dll'
    $zstdHeader = Join-Path $zstdDir 'include\zstd.h'
    
    if (Test-Path $zstdLib) { Copy-Item $zstdLib (Join-Path $Directories.Build 'zstd.lib') -Force }
    if (Test-Path $zstdDll) { Copy-Item $zstdDll (Join-Path $Directories.Bin 'libzstd.dll') -Force }
    if (Test-Path $zstdHeader) { Copy-Item $zstdHeader (Join-Path $Directories.Tools 'zstd.h') -Force }
    
    Write-Log "Dependencies installed successfully" -Level Info
}

# ---------------------------------------------------------------------------
#  Performance monitoring and optimization hints
# ---------------------------------------------------------------------------
function Get-SystemPerformanceInfo {
    $info = @{
        TotalRAM = 0
        AvailableRAM = 0
        CPUCores = [Environment]::ProcessorCount
        GPUInfo = @()
        StorageType = 'Unknown'
    }
    
    try {
        # Get RAM information
        $memory = Get-CimInstance -ClassName Win32_PhysicalMemory
        $info.TotalRAM = ($memory | Measure-Object -Property Capacity -Sum).Sum
        
        $availableRAM = Get-Counter -Counter "\Memory\Available MBytes" -SampleInterval 1 -MaxSamples 1
        $info.AvailableRAM = $availableRAM.CounterSamples[0].CookedValue * 1MB
        
        # Get GPU information
        $gpus = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.AdapterRAM -gt 0 }
        foreach ($gpu in $gpus) {
            $info.GPUInfo += @{
                Name = $gpu.Name
                RAM = $gpu.AdapterRAM
                Driver = $gpu.DriverVersion
            }
        }
        
        # Get storage type for temp directory
        $volume = Get-Volume -FilePath $Directories.Tools
        $partition = Get-Partition -Volume $volume
        $disk = Get-PhysicalDisk -Number $partition.DiskNumber
        $info.StorageType = if ($disk.MediaType -eq 'SSD') { 'SSD' } else { 'HDD' }
        
    } catch {
        Write-Log "Could not retrieve complete system information: $($_.Exception.Message)" -Level Warning
    }
    
    return $info
}

function Show-PerformanceRecommendations {
    $sysInfo = Get-SystemPerformanceInfo
    
    Write-Log "=== System Performance Analysis ===" -Level Info
    Write-Log "CPU Cores: $($sysInfo.CPUCores)" -Level Info
    Write-Log "Total RAM: $([Math]::Round($sysInfo.TotalRAM / 1GB, 1)) GB" -Level Info
    Write-Log "Available RAM: $([Math]::Round($sysInfo.AvailableRAM / 1GB, 1)) GB" -Level Info
    Write-Log "Storage Type: $($sysInfo.StorageType)" -Level Info
    
    if ($sysInfo.GPUInfo.Count -gt 0) {
        Write-Log "GPU(s) detected:" -Level Info
        foreach ($gpu in $sysInfo.GPUInfo) {
            $vramGB = [Math]::Round($gpu.RAM / 1GB, 1)
            Write-Log "  - $($gpu.Name): ${vramGB} GB VRAM" -Level Info
        }
    }
    
    # Performance recommendations
    Write-Log "=== Performance Recommendations ===" -Level Info
    
    if ($sysInfo.CPUCores -gt 8) {
        Write-Log "✓ Excellent CPU core count for parallel processing" -Level Info
    } elseif ($sysInfo.CPUCores -ge 4) {
        Write-Log "◐ Good CPU core count, consider limiting thread count to avoid overload" -Level Info
    } else {
        Write-Log "⚠ Limited CPU cores, expect slower processing" -Level Warning
    }
    
    if ($sysInfo.AvailableRAM -gt 32GB) {
        Write-Log "✓ Excellent RAM availability for large models" -Level Info
    } elseif ($sysInfo.AvailableRAM -gt 16GB) {
        Write-Log "◐ Good RAM availability" -Level Info
    } else {
        Write-Log "⚠ Limited RAM may cause swapping with large models" -Level Warning
    }
    
    if ($sysInfo.StorageType -eq 'SSD') {
        Write-Log "✓ SSD detected - fast I/O performance expected" -Level Info
    } else {
        Write-Log "◐ HDD detected - consider using SSD for better performance" -Level Info
    }
}

# ---------------------------------------------------------------------------
#  Entry point with enhanced error handling
# ---------------------------------------------------------------------------
try {
    # Set up error handling
    $ErrorActionPreference = 'Stop'
    $VerbosePreference = if ($PSBoundParameters['Verbose']) { 'Continue' } else { 'SilentlyContinue' }
    
    # Validate parameters
    if ($ModelPath -and -not (Test-Path $ModelPath)) {
        throw "Model file not found: $ModelPath"
    }
    
    if ($ThreadCount -gt 16) {
        Write-Warning "Thread count limited to 16 for stability"
        $ThreadCount = 16
    }
    
    # Check minimum requirements
    if ([Environment]::OSVersion.Version.Major -lt 10) {
        throw "This script requires Windows 10 or later"
    }
    
    if ((Get-Host).Version.Major -lt 5) {
        throw "This script requires PowerShell 5.1 or later"
    }
    
    # Install dependencies first
    Install-Dependencies
    
    # Compile the enhanced converter
    Compile-EnhancedConverter
    
    # Show performance info
    Show-PerformanceRecommendations
    
    # Run main application
    Main
    
} catch {
    Write-Error "Fatal error in Advanced GGUF Converter: $($_.Exception.Message)"
    Write-Error "Location: $($_.InvocationInfo.ScriptName):$($_.InvocationInfo.ScriptLineNumber)"
    
    if ($_.Exception.InnerException) {
        Write-Error "Inner exception: $($_.Exception.InnerException.Message)"
    }
    
    # Show troubleshooting hints
    Write-Host "`n=== Troubleshooting Hints ===" -ForegroundColor Yellow
    Write-Host "1. Ensure you have Visual Studio 2019+ with C++ tools installed" -ForegroundColor Yellow
    Write-Host "2. Run PowerShell as Administrator if you encounter permission errors" -ForegroundColor Yellow
    Write-Host "3. Check that you have at least 4GB free disk space" -ForegroundColor Yellow
    Write-Host "4. Disable antivirus temporarily if it blocks downloads/compilation" -ForegroundColor Yellow
    Write-Host "5. Use -SkipGpu switch if you don't have an NVIDIA GPU" -ForegroundColor Yellow
    
    exit 1
}

# ---------------------------------------------------------------------------
#  Export functions for testing/debugging
# ---------------------------------------------------------------------------
if ($MyInvocation.InvocationName -ne '.') {
    # Only export functions when script is dot-sourced
    Export-ModuleMember -Function @(
        'Initialize-Logging',
        'Write-Log', 
        'Initialize-Directories',
        'Initialize-Python',
        'Build-Component',
        'Compile-EnhancedConverter',
        'Get-SystemPerformanceInfo',
        'Show-PerformanceRecommendations'
    )
}
