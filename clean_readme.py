"""Clean README by removing duplicates and unwanted sections"""

# Read the file
with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

# Find the first occurrence of "Made with ❤️ and 🧠"
first_end = content.find("<p align=\"center\">Made with ❤️ and 🧠</p>")

if first_end != -1:
    # Keep everything up to and including the first closing
    clean_content = content[:first_end + len("<p align=\"center\">Made with ❤️ and 🧠</p>")]
    
    # Write the clean version
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(clean_content)
    
    print("✓ README.md cleaned - removed duplicate sections")
    print(f"  Original length: {len(content)} characters")
    print(f"  New length: {len(clean_content)} characters")
else:
    print("✗ Could not find closing marker")
