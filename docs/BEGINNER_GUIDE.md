# Beginner guide: VS Code on a MacBook (M2)

## Words you will see
- **Terminal**: a text window where you type commands instead of clicking.
- **Folder / directory**: same thing. `cd name` means "go into folder name".
- **Virtual environment (.venv)**: a private box of Python packages for this project only, so projects do not fight each other.
- **pip**: the tool that installs Python packages into that box.
- **Test (pytest)**: a small program that checks your code gives the right answer.

## Part 1: One-time setup
1. Install VS Code from https://code.visualstudio.com (download for Mac, Apple Silicon). Drag it to Applications.
2. Install Python 3 from https://www.python.org/downloads/macos/ (the latest 3.12 or 3.13 installer).
3. Open VS Code. Press `Cmd+Shift+X`, search **Python** (by Microsoft), click Install. Also install **Pylance** if offered.
4. Check Python: open Terminal (`Cmd+Space`, type Terminal) and run `python3 --version`. You should see 3.x.

## Part 2: Open the project
1. Unzip `hospital-inventory-day1.zip`. Move the folder `hospital-inventory-intelligence` to Documents.
2. In VS Code: File > Open Folder... > choose that folder > Open. Click "Yes, I trust the authors".
3. Left side = Explorer (your files). Click files to open them. `Ctrl+backtick` opens the built-in terminal at the bottom.

## Part 3: Create the virtual environment
In the VS Code terminal:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
You should see `(.venv)` at the start of the terminal line. If you close the terminal later, run `source .venv/bin/activate` again.
Tell VS Code to use it: `Cmd+Shift+P` > "Python: Select Interpreter" > pick the one that says `.venv`.

## Part 4: Run things
```bash
python scripts/generate_data.py
python run_audit.py
pytest -q
```
Expected: data files written, a table saying "Products reconciled: 10 of 10", and `11 passed`.

## Part 5: Git and GitHub
```bash
git init
git add .
git commit -m "Day 1: reconciled synthetic data, validation, tests"
```
Then create an empty repository on github.com (name it hospital-inventory-intelligence) and follow the 2 commands it shows (`git remote add origin ...` and `git push -u origin main`).
