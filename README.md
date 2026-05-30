# 20226 World Cup Insights

## Setup Instructions

### macOS

1. **Install Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, add Poetry to your PATH by adding this line to your shell profile (`~/.zshrc` or `~/.bash_profile`):
   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```
   
   Then reload your shell:
   ```bash
   source ~/.zshrc
   ```

2. **Configure Poetry to use in-project virtual environments**
   ```bash
   poetry config virtualenvs.in-project true
   ```

3. **Install project dependencies**
   ```bash
   poetry install
   ```

### Windows

1. **Install Poetry**
   
   Using PowerShell:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   ```
   
   Or using Windows Package Manager:
   ```powershell
   winget install Python.Poetry
   ```

2. **Configure Poetry to use in-project virtual environments**
   ```cmd
   poetry config virtualenvs.in-project true
   ```

3. **Install project dependencies**
   ```cmd
   poetry install
   ```