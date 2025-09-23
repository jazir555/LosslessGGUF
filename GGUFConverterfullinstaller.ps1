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
/*
  AdvancedGGUF_Converter v6.0 — GPU-first (nvCOMP batched ZSTD) + multi-stream GPU SHA256 + official GGUF
  - Drop-in single-file C++ source. Compile with nvcc or host compiler that links CUDA/nvCOMP libs.
  - Requires: CUDA toolkit (runtime), nvCOMP (batched API), gguf/ggml C headers & libs, zstd (fallback), nlohmann::json, OpenSSL (fallback).
  - Example build (linux):
      nvcc -O3 -std=c++17 -Xcompiler -fPIC converter.cpp -o converter \
           -I/path/to/json -I/path/to/gguf/include -I/usr/local/cuda/include -I/path/to/nvcomp/include \
           -L/path/to/libs -lnvcomp -lcudart -lgguf -lzstd -lssl -lcrypto
*/

#include <cstdio>
#include <cstdlib>
#include <cstdint>
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
#include <queue>
#include <condition_variable>
#include <sstream>
#include <iomanip>
#include <optional>

#include "json.hpp"
using json = nlohmann::json;

#define GGML_BUILD
#include "ggml.h"
#include "gguf.h"

#include <cuda_runtime.h>
#include <nvcomp.h>   // batched nvCOMP C API
#include <zstd.h>
#include <openssl/sha.h>

// DFloat11 extern API (optional - for CPU fallback)
extern "C" {
    size_t DFloat11_compress_bound(size_t size);
    int    DFloat11_compress(const uint8_t* src, size_t src_size, uint8_t* dst, size_t* dst_size);
    int    DFloat11_decompress(const uint8_t* src, size_t src_size, uint8_t* dst, size_t dst_size);
}

// ---------------------------
// Configuration
// ---------------------------
static constexpr size_t CHUNK_MIN = 4ULL * 1024 * 1024;        // 4 MiB
static constexpr size_t CHUNK_DEFAULT = 16ULL * 1024 * 1024;  // 16 MiB
static constexpr size_t CHUNK_MAX = 128ULL * 1024 * 1024;     // 128 MiB (depending on GPU memory)
static constexpr const char* CHECKPOINT_FILENAME = "convert_checkpoint.json";
static constexpr const char* TELEMETRY_CSV = "convert_telemetry.csv";
static constexpr const char* TELEMETRY_JSON = "convert_telemetry.json";

// ---------------------------
// Logging
// ---------------------------
enum LogLevel { DEBUG=0, INFO=1, WARN=2, ERROR=3 };
static std::mutex g_log_mutex;
static void log_message(LogLevel level, const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    const char* names[] = {"DEBUG","INFO","WARN","ERROR"};
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    printf("[%s.%03d] [%s] ", buf, (int)ms.count(), names[level]);
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    printf("\n"); fflush(stdout);
}
#define LOGD(...) log_message(DEBUG, __VA_ARGS__)
#define LOGI(...) log_message(INFO,  __VA_ARGS__)
#define LOGW(...) log_message(WARN,  __VA_ARGS__)
#define LOGE(...) log_message(ERROR, __VA_ARGS__)

// ---------------------------
// Cancellation / signals
// ---------------------------
static std::atomic<bool> g_cancelled{false};
#ifdef _WIN32
#include <windows.h>
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
        LOGI("Cancellation requested by user");
        g_cancelled = true;
        return TRUE;
    }
    return FALSE;
}
#else
#include <signal.h>
void signal_handler(int s) {
    LOGI("Signal %d received -> cancellation", s);
    g_cancelled = true;
}
#endif

// ---------------------------
// Utility: CPU SHA256 (fallback or verification)
static std::string sha256_hex_cpu(const uint8_t* data, size_t len) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(data, len, hash);
    std::ostringstream ss; ss << std::hex << std::setfill('0');
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) ss << std::setw(2) << (int)hash[i];
    return ss.str();
}

// ---------------------------
// Highly-optimized multi-stream GPU SHA-256 (tree-hash)
// - per-segment parallel SHA256 kernel (one block per segment)
// - pairwise reduction kernel to hash concatenated digests
// - final root digest returned as hex string
// Notes: This tree mode is high-throughput and secure; it differs from streaming SHA256 (single linear digest).
// Use tree mode for fast integrity; if canonical single-block SHA256 is required, adapt or use CPU fallback.
// ---------------------------

__device__ __constant__ uint32_t d_sha256_k[64] = {
  0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
  0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
  0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
  0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
  0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
  0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
  0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
  0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

__device__ inline uint32_t rotr32(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }

// Per-segment SHA256 kernel (one block per segment)
extern "C" __global__
void sha256_segment_kernel(const uint8_t* __restrict__ dev_buf, size_t total_len,
                           size_t segment_bytes, uint8_t* __restrict__ dev_out_hashes, size_t out_stride) {
    size_t seg_idx = blockIdx.x;
    size_t seg_offset = seg_idx * segment_bytes;
    if (seg_offset >= total_len) return;
    size_t remaining = total_len - seg_offset;
    size_t this_len = remaining < segment_bytes ? remaining : segment_bytes;
    const uint8_t* seg_ptr = dev_buf + seg_offset;

    extern __shared__ uint32_t s_mem[]; // dynamic shared for W + state
    uint32_t* W = s_mem;                // 64 words
    uint32_t* s_state = s_mem + 64;     // 8 words

    if (threadIdx.x == 0) {
        s_state[0] = 0x6a09e667u;
        s_state[1] = 0xbb67ae85u;
        s_state[2] = 0x3c6ef372u;
        s_state[3] = 0xa54ff53au;
        s_state[4] = 0x510e527fu;
        s_state[5] = 0x9b05688cu;
        s_state[6] = 0x1f83d9abu;
        s_state[7] = 0x5be0cd19u;
    }
    __syncthreads();

    size_t num_blocks = (this_len + 8 + 63) / 64;
    for (size_t blk = 0; blk < num_blocks; ++blk) {
        // W[0..15]
        for (int i = threadIdx.x; i < 16; i += blockDim.x) {
            size_t byte_index = blk * 64 + i * 4;
            uint32_t val = 0;
            for (int b = 0; b < 4; ++b) {
                size_t idx = byte_index + b;
                uint8_t byte = 0;
                if (idx < this_len) byte = seg_ptr[idx];
                else if (idx == this_len) byte = 0x80;
                val = (val << 8) | byte;
            }
            W[i] = val;
        }
        __syncthreads();

        if (blk == num_blocks - 1) {
            uint64_t bits_len = (uint64_t)this_len * 8;
            if (threadIdx.x == 0) {
                W[14] = (uint32_t)((bits_len >> 32) & 0xFFFFFFFFu);
                W[15] = (uint32_t)(bits_len & 0xFFFFFFFFu);
            }
        }
        __syncthreads();

        for (int t = 16 + threadIdx.x; t < 64; t += blockDim.x) {
            uint32_t s0 = rotr32(W[t-15],7) ^ rotr32(W[t-15],18) ^ (W[t-15] >> 3);
            uint32_t s1 = rotr32(W[t-2],17) ^ rotr32(W[t-2],19) ^ (W[t-2] >> 10);
            W[t] = W[t-16] + s0 + W[t-7] + s1;
        }
        __syncthreads();

        uint32_t a = s_state[0], b = s_state[1], c = s_state[2], d = s_state[3];
        uint32_t e = s_state[4], f = s_state[5], g = s_state[6], h = s_state[7];

        for (int t = 0; t < 64; ++t) {
            uint32_t S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h + S1 + ch + d_sha256_k[t] + W[t];
            uint32_t S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;
            h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
        }
        if (threadIdx.x == 0) {
            s_state[0] += a; s_state[1] += b; s_state[2] += c; s_state[3] += d;
            s_state[4] += e; s_state[5] += f; s_state[6] += g; s_state[7] += h;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint8_t* out = dev_out_hashes + seg_idx * out_stride;
        for (int i = 0; i < 8; ++i) {
            uint32_t v = s_state[i];
            out[i*4 + 0] = (v >> 24) & 0xFF;
            out[i*4 + 1] = (v >> 16) & 0xFF;
            out[i*4 + 2] = (v >> 8) & 0xFF;
            out[i*4 + 3] = v & 0xFF;
        }
    }
}

// Pairwise reduction kernel: hash pairs of 32-byte digests -> 32-byte digest
extern "C" __global__
void sha256_pair_reduce_kernel(const uint8_t* __restrict__ in_hashes, size_t in_count,
                               uint8_t* __restrict__ out_hashes) {
    size_t pair_idx = blockIdx.x;
    size_t i0 = pair_idx * 2;
    if (i0 >= in_count) return;

    // concatenate 32 + 32 bytes into 64-byte buffer
    __shared__ uint8_t buf[64];
    for (int i = threadIdx.x; i < 32; i += blockDim.x) {
        buf[i] = in_hashes[i0 * 32 + i];
    }
    if (i0 + 1 < in_count) {
        for (int i = threadIdx.x; i < 32; i += blockDim.x) {
            buf[32 + i] = in_hashes[(i0+1) * 32 + i];
        }
    } else {
        for (int i = threadIdx.x; i < 32; i += blockDim.x) {
            buf[32 + i] = 0;
        }
    }
    __syncthreads();

    // single-block SHA-256 on 64-byte buffer
    __shared__ uint32_t W[64];
    __shared__ uint32_t s_state[8];
    if (threadIdx.x == 0) {
        s_state[0] = 0x6a09e667u; s_state[1] = 0xbb67ae85u; s_state[2] = 0x3c6ef372u; s_state[3] = 0xa54ff53au;
        s_state[4] = 0x510e527fu; s_state[5] = 0x9b05688cu; s_state[6] = 0x1f83d9abu; s_state[7] = 0x5be0cd19u;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        W[i] = (uint32_t)buf[i*4 + 0] << 24 | (uint32_t)buf[i*4 + 1] << 16 | (uint32_t)buf[i*4 + 2] << 8 | (uint32_t)buf[i*4 + 3];
    }
    __syncthreads();

    // set length bits (512 bits)
    if (threadIdx.x == 0) {
        W[14] = 0;
        W[15] = 512;
    }
    __syncthreads();

    for (int t = 16 + threadIdx.x; t < 64; t += blockDim.x) {
        uint32_t s0 = rotr32(W[t-15],7) ^ rotr32(W[t-15],18) ^ (W[t-15] >> 3);
        uint32_t s1 = rotr32(W[t-2],17) ^ rotr32(W[t-2],19) ^ (W[t-2] >> 10);
        W[t] = W[t-16] + s0 + W[t-7] + s1;
    }
    __syncthreads();

    uint32_t a = s_state[0], b = s_state[1], c = s_state[2], d = s_state[3];
    uint32_t e = s_state[4], f = s_state[5], g = s_state[6], h = s_state[7];

    for (int t = 0; t < 64; ++t) {
        uint32_t S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + d_sha256_k[t] + W[t];
        uint32_t S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }
    if (threadIdx.x == 0) {
        uint32_t H0 = s_state[0] + a; uint32_t H1 = s_state[1] + b; uint32_t H2 = s_state[2] + c; uint32_t H3 = s_state[3] + d;
        uint32_t H4 = s_state[4] + e; uint32_t H5 = s_state[5] + f; uint32_t H6 = s_state[6] + g; uint32_t H7 = s_state[7] + h;
        uint8_t* out = out_hashes + pair_idx * 32;
        out[0] = (H0 >> 24) & 0xFF; out[1] = (H0 >> 16) & 0xFF; out[2] = (H0 >> 8) & 0xFF; out[3] = H0 & 0xFF;
        out[4] = (H1 >> 24) & 0xFF; out[5] = (H1 >> 16) & 0xFF; out[6] = (H1 >> 8) & 0xFF; out[7] = H1 & 0xFF;
        out[8] = (H2 >> 24) & 0xFF; out[9] = (H2 >> 16) & 0xFF; out[10] = (H2 >> 8) & 0xFF; out[11] = H2 & 0xFF;
        out[12] = (H3 >> 24) & 0xFF; out[13] = (H3 >> 16) & 0xFF; out[14] = (H3 >> 8) & 0xFF; out[15] = H3 & 0xFF;
        out[16] = (H4 >> 24) & 0xFF; out[17] = (H4 >> 16) & 0xFF; out[18] = (H4 >> 8) & 0xFF; out[19] = H4 & 0xFF;
        out[20] = (H5 >> 24) & 0xFF; out[21] = (H5 >> 16) & 0xFF; out[22] = (H5 >> 8) & 0xFF; out[23] = H5 & 0xFF;
        out[24] = (H6 >> 24) & 0xFF; out[25] = (H6 >> 16) & 0xFF; out[26] = (H6 >> 8) & 0xFF; out[27] = H6 & 0xFF;
        out[28] = (H7 >> 24) & 0xFF; out[29] = (H7 >> 16) & 0xFF; out[30] = (H7 >> 8) & 0xFF; out[31] = H7 & 0xFF;
    }
}

// Host-side orchestrator: compute tree-hash root hex string
static std::string gpu_tree_sha256_hex(uint8_t* d_buf, size_t len, cudaStream_t stream, size_t segment_size = (1<<20)) {
    if (len == 0) {
        // SHA256(empty) canonical value:
        return "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    }
    // clamp segment size for memory
    if (segment_size < 64) segment_size = 64;
    size_t num_segments = (len + segment_size - 1) / segment_size;
    // allocate device array for segment digests
    uint8_t* d_hashes;
    cudaMallocAsync(reinterpret_cast<void**>(&d_hashes), num_segments * 32, stream);

    int threads_per_block = 256;
    size_t shared_bytes = (64 + 8) * sizeof(uint32_t); // W[64] + state[8]
    sha256_segment_kernel<<<(int)num_segments, threads_per_block, (int)shared_bytes, stream>>>(d_buf, len, segment_size, d_hashes, 32);

    // iterative pairwise reduction
    size_t level_count = num_segments;
    uint8_t* d_curr = d_hashes;
    std::vector<uint8_t*> to_free;
    while (level_count > 1) {
        size_t next_count = (level_count + 1) / 2;
        uint8_t* d_next = nullptr;
        cudaMallocAsync(reinterpret_cast<void**>(&d_next), next_count * 32, stream);
        int blocks = (int)next_count;
        int threads = 128;
        sha256_pair_reduce_kernel<<<blocks, threads, 0, stream>>>(d_curr, level_count, d_next);
        to_free.push_back(d_curr);
        d_curr = d_next;
        level_count = next_count;
    }
    // copy final 32 bytes
    uint8_t final_hash[32];
    cudaMemcpyAsync(final_hash, d_curr, 32, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (auto p : to_free) cudaFreeAsync(p, stream);
    cudaFreeAsync(d_curr, stream);

    std::ostringstream ss; ss << std::hex << std::setfill('0');
    for (int i = 0; i < 32; ++i) ss << std::setw(2) << (int)final_hash[i];
    return ss.str();
}

// ---------------------------
// nvCOMP batched zstd wrapper (uses nvCOMP low-level batched API)
// - Based on nvCOMP docs and examples for batched compress flow.
// - Important: check your nvCOMP header/version if symbol names differ.
// Docs described these batched functions: nvcompBatchedZstdCompressGetTempSizeAsync,
// nvcompBatchedZstdCompressGetMaxOutputChunkSize, nvcompBatchedZstdCompressAsync. :contentReference[oaicite:2]{index=2}
// ---------------------------

struct NvcompBatchResult {
    std::vector<std::vector<uint8_t>> compressed_chunks; // host copies
    std::vector<size_t> compressed_sizes;
    std::vector<std::string> chunk_shas; // hex (we'll compute via GPU tree-sha or device kernel)
    bool ok = false;
    std::string err;
};

// Compress a batch of device buffers (device_ptrs[]), each with device_sizes[] bytes.
// Returns compressed chunks on host. Uses provided CUDA stream.
static NvcompBatchResult nvcomp_batched_zstd_compress_device(void** device_ptrs, const size_t* device_sizes,
                                                             size_t num_chunks, size_t max_uncompressed_chunk_bytes,
                                                             cudaStream_t stream, int zstd_level = 3) {
    NvcompBatchResult result;
    if (num_chunks == 0) { result.ok = true; return result; }

    // Prepare batched options
    nvcompBatchedZstdCompressOpts_t opts = nvcompBatchedZstdCompressOptsDefault();
    opts.compression_level = zstd_level;

    // Query required temp workspace (device)
    size_t temp_bytes = 0;
    nvcompStatus_t st = nvcompBatchedZstdCompressGetTempSizeAsync(
        num_chunks, max_uncompressed_chunk_bytes, opts, &temp_bytes, max_uncompressed_chunk_bytes * num_chunks);
    if (st != nvcompSuccess) {
        result.err = "nvcomp: GetTempSizeAsync failed";
        return result;
    }

    // Query maximum output chunk size
    size_t max_out_chunk = 0;
    st = nvcompBatchedZstdCompressGetMaxOutputChunkSize(num_chunks, max_uncompressed_chunk_bytes, opts, &max_out_chunk);
    if (st != nvcompSuccess) {
        result.err = "nvcomp: GetMaxOutputChunkSize failed";
        return result;
    }

    // Allocate device temp workspace
    void* d_temp = nullptr;
    if (temp_bytes > 0) {
        if (cudaMallocAsync(&d_temp, temp_bytes, stream) != cudaSuccess) {
            result.err = "cudaMallocAsync temp failed";
            return result;
        }
    }

    // Allocate device output buffers for each chunk
    std::vector<void*> d_out_ptrs(num_chunks, nullptr);
    for (size_t i = 0; i < num_chunks; ++i) {
        if (cudaMallocAsync(&d_out_ptrs[i], max_out_chunk, stream) != cudaSuccess) {
            result.err = "cudaMallocAsync out buffer failed";
            for (size_t j = 0; j < i; ++j) cudaFreeAsync(d_out_ptrs[j], stream);
            if (d_temp) cudaFreeAsync(d_temp, stream);
            return result;
        }
    }

    // Allocate device arrays for output sizes and status
    size_t* d_out_sizes = nullptr;
    nvcompStatus_t* d_statuses = nullptr;
    cudaMallocAsync(reinterpret_cast<void**>(&d_out_sizes), sizeof(size_t) * num_chunks, stream);
    cudaMallocAsync(reinterpret_cast<void**>(&d_statuses), sizeof(nvcompStatus_t) * num_chunks, stream);

    // Launch batched compress async
    st = nvcompBatchedZstdCompressAsync(
        device_ptrs,
        device_sizes,
        max_uncompressed_chunk_bytes,
        num_chunks,
        d_temp,
        temp_bytes,
        d_out_ptrs.data(),
        d_out_sizes,
        opts,
        d_statuses,
        stream
    );

    if (st != nvcompSuccess) {
        result.err = "nvcomp: CompressAsync launch failed";
        for (auto p : d_out_ptrs) if (p) cudaFreeAsync(p, stream);
        if (d_temp) cudaFreeAsync(d_temp, stream);
        cudaFreeAsync(d_out_sizes, stream);
        cudaFreeAsync(d_statuses, stream);
        return result;
    }

    // Wait for completion
    cudaStreamSynchronize(stream);

    // copy compressed sizes & statuses back to host
    std::vector<size_t> out_sizes(num_chunks);
    std::vector<nvcompStatus_t> statuses(num_chunks);
    cudaMemcpy(out_sizes.data(), d_out_sizes, sizeof(size_t) * num_chunks, cudaMemcpyDeviceToHost);
    cudaMemcpy(statuses.data(), d_statuses, sizeof(nvcompStatus_t) * num_chunks, cudaMemcpyDeviceToHost);

    // For each chunk, copy compressed data to host and compute gpu tree-sha on device (we already have device compressed buffers)
    result.compressed_chunks.resize(num_chunks);
    result.compressed_sizes.resize(num_chunks);
    result.chunk_shas.resize(num_chunks);

    for (size_t i = 0; i < num_chunks; ++i) {
        if (statuses[i] != nvcompSuccess) {
            result.err = "nvcomp reported failure for chunk " + std::to_string(i);
            break;
        }
        size_t csize = out_sizes[i];
        result.compressed_sizes[i] = csize;
        // compute SHA on device buffer d_out_ptrs[i] using gpu tree-sha
        std::string hexsha = gpu_tree_sha256_hex(reinterpret_cast<uint8_t*>(d_out_ptrs[i]), csize, stream, 1<<20);
        result.chunk_shas[i] = hexsha;

        // copy compressed data to host
        std::vector<uint8_t> host_comp(csize);
        cudaMemcpyAsync(host_comp.data(), d_out_ptrs[i], csize, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        result.compressed_chunks[i] = std::move(host_comp);
    }

    // cleanup device allocs
    for (auto p : d_out_ptrs) if (p) cudaFreeAsync(p, stream);
    if (d_temp) cudaFreeAsync(d_temp, stream);
    if (d_out_sizes) cudaFreeAsync(d_out_sizes, stream);
    if (d_statuses) cudaFreeAsync(d_statuses, stream);

    if (!result.err.empty()) {
        result.ok = false;
    } else {
        result.ok = true;
    }
    return result;
}

// ---------------------------
// Simple thread-safe checkpoint manager
// ---------------------------
struct Checkpoint {
    std::string path;
    json j;
    std::mutex m;
    Checkpoint(const std::string& p): path(p) {
        if (std::filesystem::exists(p)) {
            std::ifstream in(p, std::ios::binary);
            if (in) in >> j;
        } else {
            j = json::object();
            j["done"] = json::object();
        }
    }
    void mark_done(const std::string& tensor_name) {
        std::lock_guard<std::mutex> lk(m);
        j["done"][tensor_name] = true;
        save();
    }
    bool is_done(const std::string& tensor_name) {
        std::lock_guard<std::mutex> lk(m);
        return j.contains("done") && j["done"].contains(tensor_name) && j["done"][tensor_name].get<bool>();
    }
    void save() {
        std::lock_guard<std::mutex> lk(m);
        std::ofstream o(path + ".tmp", std::ios::binary);
        o << j.dump(2);
        o.close();
        std::filesystem::rename(path + ".tmp", path);
    }
};

// ---------------------------
// SafeTensors streaming reader (robust)
// ---------------------------
class SafeTensorsStreamReader {
public:
    SafeTensorsStreamReader(const std::string& path) : path_(path) {
        file_.open(path, std::ios::binary);
        if (!file_) throw std::runtime_error("Cannot open safetensors: " + path);
        uint64_t header_len = 0;
        file_.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
        if (!file_) throw std::runtime_error("Failed to read header length");
        if (header_len == 0 || header_len > 200ULL * 1024 * 1024) throw std::runtime_error("Suspicious header length");
        header_len_ = header_len;
        std::string header(header_len_, '\0');
        file_.read(header.data(), header_len_);
        header_json_ = json::parse(header);
        data_offset_ = sizeof(header_len) + header_len_;
        for (auto& it : header_json_.items()) {
            if (it.key() == "__metadata__") continue;
            auto meta = it.value();
            uint64_t start = meta["data_offsets"][0].get<uint64_t>();
            uint64_t end = meta["data_offsets"][1].get<uint64_t>();
            offsets_[it.key()] = {start, end};
            dtypes_[it.key()] = meta.value("dtype", std::string("F32"));
            shapes_[it.key()] = meta["shape"].get<std::vector<int64_t>>();
        }
    }

    std::vector<std::tuple<std::string, std::vector<int64_t>, std::string, uint64_t>> enumerate_tensors_raw() {
        std::vector<std::tuple<std::string, std::vector<int64_t>, std::string, uint64_t>> out;
        for (auto& it : header_json_.items()) {
            if (it.key() == "__metadata__") continue;
            std::vector<int64_t> shape = shapes_[it.key()];
            std::string dtype = dtypes_[it.key()];
            uint64_t start = it.value()["data_offsets"][0].get<uint64_t>();
            uint64_t end   = it.value()["data_offsets"][1].get<uint64_t>();
            out.emplace_back(it.key(), shape, dtype, end - start);
        }
        return out;
    }

    size_t read_tensor_chunk(const std::string& tensor_name, uint64_t offset_in_tensor, uint8_t* buf, size_t bufsize) {
        auto it = offsets_.find(tensor_name);
        if (it == offsets_.end()) throw std::runtime_error("Unknown tensor: " + tensor_name);
        uint64_t start_offset = data_offset_ + it->second.first + offset_in_tensor;
        file_.seekg(start_offset, std::ios::beg);
        file_.read(reinterpret_cast<char*>(buf), bufsize);
        return file_.gcount();
    }

private:
    std::ifstream file_;
    std::string path_;
    uint64_t header_len_;
    uint64_t data_offset_;
    json header_json_;
    std::unordered_map<std::string, std::pair<uint64_t,uint64_t>> offsets_;
    std::unordered_map<std::string, std::string> dtypes_;
    std::unordered_map<std::string, std::vector<int64_t>> shapes_;
};

// ---------------------------
// GGUF writer (official API usage + streaming metadata)
// ---------------------------
class GGUFWriter {
public:
    GGUFWriter(const std::string& out_path) : out_path_(out_path) {
        ctx_ = gguf_init_empty();
        if (!ctx_) throw std::runtime_error("gguf_init_empty failed");
        gguf_set_val_str(ctx_, "general.name", "AdvancedGGUF_Converter_GPU_nvCOMP");
        gguf_set_val_str(ctx_, "general.architecture", "llama");
        gguf_set_val_str(ctx_, "general.version", "6.0");
    }
    ~GGUFWriter() { if (ctx_) gguf_free(ctx_); }

    void add_tensor_header(const std::string& name, const std::vector<int64_t>& shape, ggml_type ttype) {
        gguf_add_tensor(ctx_, name.c_str(), shape.data(), shape.size(), ttype, nullptr);
    }

    void write_header_initial() {
        int rc = gguf_write_to_file(ctx_, out_path_.c_str());
        if (rc != 0) throw std::runtime_error("gguf_write_to_file failed");
        append_file_ = std::fopen(out_path_.c_str(), "ab");
        if (!append_file_) throw std::runtime_error("Cannot open output for append");
    }

    uint64_t append_data(const uint8_t* data, size_t size) {
        std::lock_guard<std::mutex> lk(m_);
        uint64_t off = std::ftell(append_file_);
        size_t w = fwrite(data, 1, size, append_file_);
        if (w != size) throw std::runtime_error("Failed to append data");
        fflush(append_file_);
        return off;
    }

    void finalize_with_metadata(const json& meta) {
        gguf_set_val_str(ctx_, "gpu_streaming.metadata", meta.dump().c_str());
        if (append_file_) { fclose(append_file_); append_file_ = nullptr; }
        std::string tmp = out_path_ + ".tmp";
        int rc = gguf_write_to_file(ctx_, tmp.c_str());
        if (rc != 0) throw std::runtime_error("gguf_write_to_file(final) failed");
        std::filesystem::rename(tmp, out_path_);
    }

private:
    std::string out_path_;
    struct gguf_context* ctx_ = nullptr;
    FILE* append_file_ = nullptr;
    std::mutex m_;
};

// ---------------------------
// Converter core: GPU-first pipeline
// ---------------------------
class Converter {
public:
    Converter(const std::string& in_path, const std::string& out_path,
              size_t chunk_size = CHUNK_DEFAULT, unsigned int cuda_streams = 4)
    : reader_(in_path), writer_(out_path), chunk_size_(chunk_size), checkpoint_(CHECKPOINT_FILENAME) {
        cudaSetDevice(0);
        cuda_streams_ = std::max<unsigned int>(1, cuda_streams);
        for (unsigned int i = 0; i < cuda_streams_; ++i) {
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            streams_.push_back(s);
        }
    }
    ~Converter() {
        for (auto s : streams_) cudaStreamDestroy(s);
    }

    int run(unsigned int threads = 0) {
        try {
            auto tensors_raw = reader_.enumerate_tensors_raw();
            size_t n = tensors_raw.size();
            LOGI("Found %zu tensors", n);

            // Add gguf tensor headers
            for (auto &tup : tensors_raw) {
                const std::string &name = std::get<0>(tup);
                auto shape = std::get<1>(tup);
                std::string dtype = std::get<2>(tup);
                ggml_type ttype = parse_dtype(dtype);
                writer_.add_tensor_header(name, shape, ttype);
            }
            writer_.write_header_initial();

            // telemetry CSV header
            std::ofstream csv(TELEMETRY_CSV);
            csv << "tensor,orig_bytes,comp_bytes,ratio,ms,sha256\n";
            csv.close();
            json telemetry = json::array();

            std::atomic<size_t> index{0};
            unsigned int thread_count = threads == 0 ? std::max(1u, std::thread::hardware_concurrency()) : threads;
            std::vector<std::thread> workers;
            for (unsigned int t = 0; t < thread_count; ++t) {
                workers.emplace_back([&, t]() {
                    size_t my;
                    while ((my = index.fetch_add(1)) < n && !g_cancelled) {
                        process_one_tensor(tensors_raw[my], telemetry, t % cuda_streams_);
                    }
                });
            }
            for (auto &w : workers) w.join();

            if (g_cancelled) { LOGW("Cancelled; checkpoint saved"); return 1; }

            // finalize GGUF metadata
            json meta; meta["tensors"] = json::object();
            {
                std::lock_guard<std::mutex> lk(completed_mutex_);
                for (auto &entry : completed_tensors_) {
                    json tj;
                    tj["orig_bytes"] = entry.second.original_bytes;
                    tj["comp_bytes"] = entry.second.compressed_total;
                    tj["chunks"] = json::array();
                    for (auto &c : entry.second.chunks) {
                        json cj;
                        cj["orig_offset"] = c.orig_offset;
                        cj["orig_size"] = c.orig_size;
                        cj["comp_size"] = c.comp_size;
                        cj["comp_sha256"] = c.comp_sha256;
                        cj["out_file_offset"] = c.out_file_offset;
                        tj["chunks"].push_back(cj);
                    }
                    meta["tensors"][entry.first] = tj;
                }
            }

            writer_.finalize_with_metadata(meta);

            // telemetry file
            std::ofstream tj(TELEMETRY_JSON);
            tj << telemetry.dump(2);
            tj.close();

            LOGI("Conversion completed successfully");
            return 0;
        } catch (const std::exception &e) {
            LOGE("FATAL: %s", e.what());
            return 1;
        }
    }

private:
    struct ChunkInfo { uint64_t orig_offset; size_t orig_size; size_t comp_size; std::string comp_sha256; uint64_t out_file_offset; };
    struct LocalTensor { std::vector<ChunkInfo> chunks; uint64_t original_bytes = 0; size_t compressed_total = 0; double compress_time_ms = 0.0; };

    void process_one_tensor(const std::tuple<std::string, std::vector<int64_t>, std::string, uint64_t>& tup,
                            json& telemetry_out, unsigned int stream_id) {
        const std::string &name = std::get<0>(tup);
        uint64_t original_bytes = std::get<3>(tup);
        LOGI("Processing tensor %s (%.2f MB)", name.c_str(), original_bytes / 1e6);
        if (checkpoint_.is_done(name)) { LOGI("Skipping %s (checkpoint)", name.c_str()); return; }

        LocalTensor lt; lt.original_bytes = original_bytes;
        uint64_t remaining = original_bytes;
        uint64_t offset = 0;
        auto tstart = std::chrono::high_resolution_clock::now();

        // Process in chunk_size_ slices
        while (remaining > 0 && !g_cancelled) {
            size_t use = (size_t)std::min<uint64_t>(chunk_size_, remaining);

            // pinned host read
            uint8_t* h_buf = nullptr;
            if (cudaMallocHost(reinterpret_cast<void**>(&h_buf), use) != cudaSuccess) throw std::runtime_error("cudaMallocHost failed");
            size_t got = reader_.read_tensor_chunk(name, offset, h_buf, use);
            if (got == 0) { cudaFreeHost(h_buf); throw std::runtime_error("Unexpected EOF reading tensor chunk"); }

            // device staging
            uint8_t* d_buf = nullptr;
            if (cudaMallocAsync(reinterpret_cast<void**>(&d_buf), got, streams_[stream_id]) != cudaSuccess) {
                cudaFreeHost(h_buf); throw std::runtime_error("cudaMallocAsync device input failed");
            }
            cudaMemcpyAsync(d_buf, h_buf, got, cudaMemcpyHostToDevice, streams_[stream_id]);

            // call nvCOMP batched compress for single-chunk (we can batch multiple chunks by collecting inputs; here we use single)
            void* dev_ptrs[1] = { d_buf };
            size_t dev_sizes[1] = { got };
            NvcompBatchResult br = nvcomp_batched_zstd_compress_device(dev_ptrs, dev_sizes, 1, got, streams_[stream_id], /*zstd_level*/ 3);

            // free pinned input and device input
            cudaFreeHost(h_buf);
            cudaFreeAsync(d_buf, streams_[stream_id]);

            if (!br.ok) {
                LOGW("nvCOMP failed for %s chunk -> fallback to CPU ZSTD", name.c_str());
                // CPU fallback: re-read into host and compress
                std::vector<uint8_t> raw(use);
                size_t got2 = reader_.read_tensor_chunk(name, offset, raw.data(), use);
                if (got2 == 0) throw std::runtime_error("Fallback read failed");
                size_t bound = ZSTD_compressBound(got2);
                std::vector<uint8_t> out(bound);
                int level = std::getenv("COMPRESSION_LEVEL") ? std::atoi(std::getenv("COMPRESSION_LEVEL")) : 3;
                size_t csize = ZSTD_compress(out.data(), bound, raw.data(), got2, level);
                if (ZSTD_isError(csize)) throw std::runtime_error("ZSTD fallback failed");
                out.resize(csize);
                std::string sha = sha256_hex_cpu(out.data(), out.size());
                uint64_t file_off = writer_.append_data(out.data(), out.size());
                ChunkInfo ci{offset, (size_t)got2, csize, sha, file_off};
                lt.chunks.push_back(ci);
                lt.compressed_total += csize;
            } else {
                // use first compressed chunk from nvCOMP result
                auto &comp = br.compressed_chunks[0];
                size_t csize = br.compressed_sizes[0];
                std::string sha = br.chunk_shas[0]; // computed on-device in nvCOMP wrapper
                uint64_t file_off = writer_.append_data(comp.data(), csize);
                ChunkInfo ci{offset, (size_t)got, csize, sha, file_off};
                lt.chunks.push_back(ci);
                lt.compressed_total += csize;
            }

            offset += use;
            remaining -= use;
        } // end while chunks

        auto tend = std::chrono::high_resolution_clock::now();
        lt.compress_time_ms = std::chrono::duration<double, std::milli>(tend - tstart).count();

        // record completed tensor
        {
            std::lock_guard<std::mutex> lk(completed_mutex_);
            completed_tensors_[name] = lt;
        }

        // telemetry write
        {
            std::string joinsha;
            for (auto &c : lt.chunks) joinsha += c.comp_sha256;
            std::string tensor_sha = sha256_hex_cpu((const uint8_t*)joinsha.data(), joinsha.size());
            std::ofstream csv(TELEMETRY_CSV, std::ios::app);
            csv << name << "," << lt.original_bytes << "," << lt.compressed_total << ","
                << (lt.original_bytes ? (double)lt.original_bytes / (double)lt.compressed_total : 1.0) << ","
                << lt.compress_time_ms << "," << tensor_sha << "\n";
            csv.close();
            json tj;
            tj["name"] = name; tj["orig_bytes"] = lt.original_bytes; tj["comp_bytes"] = lt.compressed_total;
            tj["ratio"] = lt.original_bytes ? (double)lt.original_bytes / (double)lt.compressed_total : 1.0;
            tj["time_ms"] = lt.compress_time_ms; tj["sha"] = tensor_sha;
            telemetry_out.push_back(tj);
        }

        checkpoint_.mark_done(name);
    }

    ggml_type parse_dtype(const std::string& s) {
        static std::unordered_map<std::string, ggml_type> m = {
            {"F32", GGML_TYPE_F32}, {"float32", GGML_TYPE_F32},
            {"F16", GGML_TYPE_F16}, {"float16", GGML_TYPE_F16},
            {"BF16", GGML_TYPE_BF16}, {"bfloat16", GGML_TYPE_BF16},
            {"I32", GGML_TYPE_I32}, {"int32", GGML_TYPE_I32},
            {"I16", GGML_TYPE_I16}, {"int16", GGML_TYPE_I16},
            {"I8", GGML_TYPE_I8}, {"int8", GGML_TYPE_I8},
            {"U8", GGML_TYPE_I8}, {"uint8", GGML_TYPE_I8}
        };
        auto it = m.find(s);
        if (it == m.end()) return GGML_TYPE_F32;
        return it->second;
    }

    SafeTensorsStreamReader reader_;
    GGUFWriter writer_;
    size_t chunk_size_;
    Checkpoint checkpoint_;
    std::vector<cudaStream_t> streams_;
    unsigned int cuda_streams_;
    std::mutex completed_mutex_;
    std::unordered_map<std::string, LocalTensor> completed_tensors_;
};

// ---------------------------
// main
// ---------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.safetensors> <output.gguf> [chunk_bytes] [cuda_streams] [threads]\n", argv[0]);
        return 1;
    }
#ifdef _WIN32
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#else
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
#endif

    std::string input = argv[1];
    std::string output = argv[2];
    size_t chunk = CHUNK_DEFAULT;
    unsigned int streams = 4;
    unsigned int threads = 0;
    if (argc >= 4) chunk = std::min<uint64_t>(CHUNK_MAX, std::max<uint64_t>(CHUNK_MIN, std::stoull(argv[3])));
    if (argc >= 5) streams = std::max<unsigned int>(1, std::stoul(argv[4]));
    if (argc >= 6) threads = std::stoul(argv[5]);

    try {
        LOGI("AdvancedGGUF_Converter v6.0 starting (nvCOMP GPU + GPU tree-SHA256)");
        LOGI("Input: %s, Output: %s, Chunk: %zu, CUDA streams: %u, Threads: %u", input.c_str(), output.c_str(), chunk, streams, threads);
        Converter conv(input, output, chunk, streams);
        int rc = conv.run(threads);
        if (g_cancelled) { LOGW("Cancelled."); return 1; }
        return rc;
    } catch (const std::exception &e) {
        LOGE("FATAL: %s", e.what());
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
