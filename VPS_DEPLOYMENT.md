# VPS Deployment Guide - AETHER Trading Bot

## 🚀 Quick Start (VPS)

### Prerequisites
- Windows VPS (Windows Server 2016+ or Windows 10+)
- Python 3.8 or higher
- MetaTrader 5 installed and logged in
- Minimum 2GB RAM, 10GB disk space

---

## 📋 Step-by-Step Deployment

### 1. Upload Files to VPS

**Option A: Git Clone (Recommended)**
```batch
cd C:\
git clone https://github.com/Gunasekar87/Scalping_Gold.git
cd Scalping_Gold
```

**Option B: Manual Upload**
- Upload entire project folder to VPS (e.g., `C:\Scalping_Gold`)
- Ensure all files are present

### 2. Install Python (if not installed)

1. Download Python 3.11: https://www.python.org/downloads/
2. Run installer
3. ✅ **CHECK**: "Add Python to PATH"
4. Click "Install Now"
5. Verify: Open CMD and type `python --version`

### 3. Configure MT5 Credentials

Edit `config\secrets.env`:
```env
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
```

**Example**:
```env
MT5_LOGIN=12345678
MT5_PASSWORD=MySecurePass123
MT5_SERVER=ICMarkets-Demo
```

### 4. Configure Trading Settings

Edit `config\settings.yaml`:

**For Testing (PAPER mode)**:
```yaml
mode: "PAPER"  # Safe testing mode
```

**For Live Trading**:
```yaml
mode: "LIVE"  # Real money trading
```

### 5. Run the Bot

**Double-click**: `start_bot.bat`

Or from command line:
```batch
cd C:\Scalping_Gold
start_bot.bat
```

---

## 🔧 VPS-Specific Configuration

### Auto-Start on VPS Boot

1. Press `Win + R`, type `shell:startup`, press Enter
2. Create shortcut to `start_bot.bat`
3. Right-click shortcut → Properties
4. Set "Start in" to: `C:\Scalping_Gold`
5. Click OK

### Keep Bot Running 24/7

**Option A: Task Scheduler (Recommended)**

1. Open Task Scheduler
2. Create Basic Task
3. Name: "AETHER Trading Bot"
4. Trigger: "When computer starts"
5. Action: "Start a program"
6. Program: `C:\Scalping_Gold\start_bot.bat`
7. Start in: `C:\Scalping_Gold`
8. ✅ Run whether user is logged on or not
9. ✅ Run with highest privileges

**Option B: Windows Service**
```batch
# Install NSSM (Non-Sucking Service Manager)
# Download from: https://nssm.cc/download

nssm install AETHER "C:\Scalping_Gold\.venv\Scripts\python.exe"
nssm set AETHER AppDirectory "C:\Scalping_Gold"
nssm set AETHER AppParameters "run_bot.py"
nssm start AETHER
```

---

## 🛡️ Security Best Practices

### 1. Secure Credentials
```batch
# Set file permissions (Admin CMD)
icacls config\secrets.env /inheritance:r /grant:r "%USERNAME%:F"
```

### 2. Firewall Rules
- Allow Python through Windows Firewall
- Allow MetaTrader 5 connections
- Block unnecessary inbound connections

### 3. VPS Hardening
- Enable Windows Defender
- Keep Windows updated
- Use strong RDP password
- Change default RDP port (3389 → custom)
- Enable Network Level Authentication

---

## 📊 Monitoring

### Check Bot Status

**View Logs**:
```batch
cd C:\Scalping_Gold
type logs\trading_YYYYMMDD.log
```

**Monitor in Real-Time**:
- Keep CMD window open
- Watch for dashboard updates
- Check MT5 terminal for trades

### Remote Monitoring

**Option A: TeamViewer/AnyDesk**
- Install on VPS
- Monitor from anywhere

**Option B: RDP (Remote Desktop)**
```
mstsc /v:your_vps_ip
```

---

## 🔄 Updates & Maintenance

### Update Bot (Git)
```batch
cd C:\Scalping_Gold
git pull origin main
start_bot.bat
```

### Update Bot (Manual)
1. Stop the bot (Ctrl+C)
2. Upload new files
3. Run `start_bot.bat`

### Backup Configuration
```batch
# Backup before updates
xcopy config config_backup\ /E /I /Y
```

---

## ⚠️ Troubleshooting

### Bot Won't Start

**Check 1: Python Installation**
```batch
python --version
# Should show: Python 3.8 or higher
```

**Check 2: Virtual Environment**
```batch
cd C:\Scalping_Gold
.venv\Scripts\activate
python -c "import MetaTrader5; print('MT5 OK')"
```

**Check 3: MT5 Connection**
- Open MT5
- Ensure logged in
- Check "Tools → Options → Expert Advisors"
- ✅ Allow automated trading
- ✅ Allow DLL imports

### "Module Not Found" Error
```batch
cd C:\Scalping_Gold
.venv\Scripts\activate
pip install -r requirements.txt --force-reinstall
```

### MT5 Connection Failed
1. Check MT5 is running
2. Verify credentials in `config\secrets.env`
3. Check broker server name (exact match)
4. Restart MT5

### High CPU Usage
- Normal: 5-15% CPU
- High: >30% CPU
- Check for multiple bot instances
- Restart VPS if needed

---

## 📈 Performance Optimization

### VPS Settings

**Power Plan**:
```batch
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
# Sets "High Performance" mode
```

**Disable Sleep**:
```batch
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```

**Increase Process Priority**:
- Already handled by bot (sets HIGH priority automatically)

---

## 🔐 VPS Provider Recommendations

### Recommended Specs
- **CPU**: 2+ cores
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 20GB SSD
- **Network**: 100Mbps+
- **Location**: Near broker server (low latency)

### Recommended Providers
1. **Vultr** - $10-20/month, global locations
2. **DigitalOcean** - $12-24/month, reliable
3. **Contabo** - €5-15/month, budget-friendly
4. **AWS EC2** - Pay-as-you-go, enterprise-grade
5. **ForexVPS** - Optimized for trading, $20-40/month

---

## 📞 Support

### Check Logs
```batch
cd C:\Scalping_Gold
dir logs
type logs\trading_YYYYMMDD.log
```

### Common Issues
- See `TROUBLESHOOTING.md`
- Check GitHub Issues
- Review bot logs

### Emergency Stop
- Press `Ctrl+C` in CMD window
- Or close MT5
- Or stop Task Scheduler task

---

## ✅ Deployment Checklist

Before going live:

- [ ] Python 3.8+ installed
- [ ] MT5 installed and logged in
- [ ] Bot files uploaded to VPS
- [ ] `config\secrets.env` configured
- [ ] `config\settings.yaml` set to PAPER mode
- [ ] Tested in PAPER mode for 24+ hours
- [ ] Verified trades execute correctly
- [ ] Checked logs for errors
- [ ] Set up auto-start (optional)
- [ ] Configured monitoring
- [ ] Backed up configuration
- [ ] Switched to LIVE mode (when ready)

---

**Version**: 5.6.5  
**Last Updated**: January 5, 2026  
**Status**: Production Ready
