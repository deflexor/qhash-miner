$ErrorActionPreference = 'Continue'

function Probe($hostname, $port) {
  try {
    $c = New-Object System.Net.Sockets.TcpClient
    $iar = $c.BeginConnect($hostname, $port, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne(5000, $false)
    if (-not $ok) { $c.Close(); return 'TIMEOUT_CONNECT' }
    $c.EndConnect($iar)
    $stream = $c.GetStream()
    $stream.ReadTimeout = 5000
    $msg = '{"id":1,"method":"mining.subscribe","params":["winprobe/1.0"]}' + "`n"
    $bytes = [Text.Encoding]::ASCII.GetBytes($msg)
    $stream.Write($bytes, 0, $bytes.Length)
    $buf = New-Object byte[] 2048
    try {
      $n = $stream.Read($buf, 0, $buf.Length)
      if ($n -le 0) { return 'CONNECTED_EOF' }
      $txt = [Text.Encoding]::ASCII.GetString($buf, 0, $n)
      $len = [Math]::Min(220, $txt.Length)
      return ('OK ' + $txt.Substring(0, $len))
    } catch {
      return ('CONNECTED_NO_DATA ' + $_.Exception.Message)
    } finally {
      $c.Close()
    }
  } catch {
    return ('ERR ' + $_.Exception.Message)
  }
}

Write-Host ('hero IP: ' + (Probe '88.99.59.165' 1195))
Write-Host ('hero name: ' + (Probe 'qubitcoin.herominers.com' 1195))
Write-Host ('suprnova: ' + (Probe '138.68.178.28' 5555))
Write-Host ('lucky: ' + (Probe '51.68.35.181' 8610))

Write-Host '--- DNS ---'
try {
  Resolve-DnsName qubitcoin.herominers.com -Type A -ErrorAction Stop | Format-Table -AutoSize | Out-String | Write-Host
} catch {
  Write-Host $_.Exception.Message
}

Write-Host '--- proxy-ish processes ---'
Get-Process | Where-Object { $_.Name -match 'clash|verge|v2ray|mihomo|sing-box|wintun' } |
  Select-Object Name, Id | Format-Table -AutoSize | Out-String | Write-Host
