#!/usr/bin/env python3
"""
Watch paper directory for changes and auto-create zip
Alternative to shell script for better cross-platform support
"""

import os
import sys
import time
import hashlib
import subprocess
from pathlib import Path

def calculate_hash(paper_dir: Path) -> str:
    """Calculate hash of all paper files"""
    files = []
    for ext in ['*.tex', '*.bib', '*.sty']:
        files.extend(paper_dir.glob(f'**/{ext}'))
    
    # Sort for consistent hashing
    files = sorted([f for f in files if f.name != 'paper.zip'])
    
    # Calculate combined hash
    hash_md5 = hashlib.md5()
    for f in files:
        try:
            with open(f, 'rb') as file_obj:
                hash_md5.update(file_obj.read())
        except Exception:
            continue
    
    return hash_md5.hexdigest()

def create_zip(paper_dir: Path) -> bool:
    """Create paper.zip for Overleaf upload"""
    try:
        result = subprocess.run(
            ['./create_zip.sh'],
            cwd=paper_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ Updated paper.zip")
            return True
        else:
            print(f"❌ Error creating zip: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def watch_directory(paper_dir: Path):
    """Watch directory for changes"""
    print(f"👀 Watching {paper_dir} for changes...")
    print("   Press Ctrl+C to stop\n")
    
    last_hash = calculate_hash(paper_dir)
    
    while True:
        try:
            time.sleep(2)
            
            current_hash = calculate_hash(paper_dir)
            
            if current_hash != last_hash:
                print(f"\n📝 Change detected!")
                create_zip(paper_dir)
                last_hash = current_hash
                
        except KeyboardInterrupt:
            print("\n\n👋 Stopping watcher...")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")
            time.sleep(5)

def main():
    # Get paper directory
    script_dir = Path(__file__).parent.absolute()
    
    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--once':
            create_zip(script_dir)
            return
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  ./watch_paper.py          # Create zip and watch for changes")
            print("  ./watch_paper.py --once   # Create zip once and exit")
            print("  ./watch_paper.py --help   # Show this help")
            return
    
    # Initial zip creation
    create_zip(script_dir)
    
    # Start watching
    watch_directory(script_dir)

if __name__ == '__main__':
    main()
