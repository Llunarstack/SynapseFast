param(
  [Parameter(Mandatory=$true, Position=0)]
  [string]$Exe,

  [Parameter(ValueFromRemainingArguments=$true)]
  [string[]]$Args
)

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (!(Test-Path $vcvars)) {
  Write-Error "vcvars64.bat not found at: $vcvars"
  exit 1
}

function Quote-CmdArg([string]$s) {
  if ($null -eq $s) { return '""' }
  # Escape double-quotes for cmd.exe
  $s = $s -replace '"', '\"'
  # Always quote to preserve spaces/special chars
  return '"' + $s + '"'
}

$exeQ = Quote-CmdArg $Exe
$argsQ = @()
if ($Args) {
  foreach ($a in $Args) { $argsQ += (Quote-CmdArg $a) }
}
$cmd = "$exeQ $($argsQ -join ' ')"

# Run the command inside the MSVC developer environment.
cmd.exe /s /c "call `"$vcvars`" && $cmd"

