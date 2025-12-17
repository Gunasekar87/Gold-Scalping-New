import sys
import os
import compileall

def check_syntax(directory):
    print(f"Checking syntax in {directory}...")
    success = compileall.compile_dir(directory, force=True, quiet=1)
    if success:
        print("Syntax check passed!")
        sys.exit(0)
    else:
        print("Syntax check failed!")
        sys.exit(1)

if __name__ == "__main__":
    check_syntax("src")