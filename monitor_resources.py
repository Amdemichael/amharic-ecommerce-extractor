#!/usr/bin/env python3
"""
Resource Monitor for NER Training
"""

import psutil
import time
import os
import gc

def monitor_resources():
    """Monitor system resources"""
    print("🖥️  System Resource Monitor")
    print("=" * 50)
    
    # Get system info
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory.percent}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
    print(f"Disk Usage: {disk.percent}%")
    
    # Check if we should proceed
    if cpu_percent > 90:
        print("⚠️  WARNING: High CPU usage detected!")
        return False
    
    if memory.percent > 85:
        print("⚠️  WARNING: High memory usage detected!")
        return False
    
    print("✅ System resources are acceptable for training")
    return True

def clear_memory():
    """Force memory cleanup"""
    print("🧹 Clearing memory...")
    gc.collect()
    
    # Clear torch cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    print("✅ Memory cleared")

def main():
    """Main monitoring function"""
    print("Resource Monitor for Amharic NER Training")
    print("=" * 50)
    
    # Check initial state
    if not monitor_resources():
        print("\n❌ System resources are too high for safe training")
        print("Please close other applications and try again")
        return False
    
    # Clear memory
    clear_memory()
    
    # Monitor again
    print("\n" + "=" * 50)
    if monitor_resources():
        print("\n✅ Ready for lightweight training!")
        return True
    else:
        print("\n❌ Still too many resources in use")
        return False

if __name__ == "__main__":
    main() 