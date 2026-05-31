#!/usr/bin/env python3
"""
Script to fix the alerts by combining individual alert files properly.
"""

import json
from pathlib import Path
from src.config import settings

def fix_alerts():
    """Combine individual alert files into the combined alerts file."""
    print("🔧 Fixing alerts...")
    
    # Get all individual alert files
    alert_files = sorted(settings.ALERTS_DIR.glob("frame_*_alert.json"))
    print(f"Found {len(alert_files)} individual alert files")
    
    all_alerts = []
    high_count = 0
    medium_count = 0
    low_count = 0
    
    for alert_file in alert_files:
        try:
            with open(alert_file, 'r', encoding='utf-8') as f:
                alert = json.load(f)
                
            # Only include alerts that were actually triggered
            if alert.get('alert_triggered', False):
                all_alerts.append(alert)
                
                severity = alert.get('severity', 'NONE')
                if severity == 'HIGH':
                    high_count += 1
                elif severity == 'MEDIUM':
                    medium_count += 1
                elif severity == 'LOW':
                    low_count += 1
                    
                print(f"✅ Added alert: {alert_file.name} - {severity}")
            else:
                print(f"⚠️ Skipped non-triggered alert: {alert_file.name}")
                
        except Exception as e:
            print(f"❌ Error processing {alert_file.name}: {e}")
    
    # Create combined alerts
    combined = {
        "session_date": "2026-05-29",
        "total_alerts": len(all_alerts),
        "high_severity": high_count,
        "medium_severity": medium_count,
        "low_severity": low_count,
        "alerts": all_alerts
    }
    
    # Save combined alerts
    output_path = settings.ALERTS_DIR / "all_alerts.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Combined alerts saved: {len(all_alerts)} total alerts")
    print(f"   - High: {high_count}")
    print(f"   - Medium: {medium_count}")
    print(f"   - Low: {low_count}")
    
    return combined

if __name__ == "__main__":
    fix_alerts()
