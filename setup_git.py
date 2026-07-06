"""Setup script to prepare repository for GitHub push"""
import os
import shutil
import subprocess

# 1. Copy README_TEMPLATE.md to README.md (overwrite)
print("Copying README_TEMPLATE.md to README.md...")
shutil.copy2("README_TEMPLATE.md", "README.md")
print("✓ README.md updated")

# 2. Check if .gitignore exists, create if not
if not os.path.exists(".gitignore"):
    print("Creating .gitignore...")
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt
checkpoints/

# Testing
.pytest_cache/
.hypothesis/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
*.png.backup
readme.txt
"""
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("✓ .gitignore created")
else:
    print("✓ .gitignore already exists")

# 3. Check if LICENSE exists, create if not
if not os.path.exists("LICENSE"):
    print("Creating MIT LICENSE...")
    license_content = """MIT License

Copyright (c) 2024 Nati Birhauz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    with open("LICENSE", "w") as f:
        f.write(license_content)
    print("✓ LICENSE created")
else:
    print("✓ LICENSE already exists")

print("\n" + "="*60)
print("Repository setup complete!")
print("="*60)
