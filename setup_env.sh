#!/usr/bin/env bash

# Assortment Benchmark Environment Setup Script
# This script automatically installs uv and creates a Python virtual environment
# Compatible with: macOS, Linux, Windows WSL

set -e  # Exit immediately on error
set -u  # Error on undefined variables
set -o pipefail  # Catch errors in pipes

# Color definitions
# Check if terminal supports colors
if [ -t 1 ]; then
    # Check if terminal supports colors
    ncolors=$(tput colors 2>/dev/null || echo 0)
    if [ "$ncolors" -ge 8 ]; then
        RED='\033[0;31m'
        GREEN='\033[0;32m'
        YELLOW='\033[1;33m'
        BLUE='\033[0;34m'
        NC='\033[0m' # No Color
    else
        RED=''
        GREEN=''
        YELLOW=''
        BLUE=''
        NC=''
    fi
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Print colored messages
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Get script directory (cross-platform compatible)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Detect operating system
detect_os() {
    case "$OSTYPE" in
        darwin*)
            echo "macOS"
            ;;
        linux*)
            if grep -qi microsoft /proc/version 2>/dev/null; then
                echo "WSL"
            else
                echo "Linux"
            fi
            ;;
        msys*|mingw*|cygwin*)
            echo "Windows"
            ;;
        *)
            echo "Unknown"
            ;;
    esac
}

OS_TYPE=$(detect_os)

echo "========================================"
echo "  Assortment Benchmark Setup"
echo "========================================"
echo ""
print_info "Detected OS: $OS_TYPE"
echo ""

# 1. Check and install uv
print_info "Checking if uv is installed..."
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version)
    print_success "uv is already installed: $UV_VERSION"
else
    print_warning "uv is not installed. Installing now..."
    
    case "$OS_TYPE" in
        macOS|Linux|WSL)
            # Install uv using official installer
            if command -v curl &> /dev/null; then
                curl -LsSf https://astral.sh/uv/install.sh | sh
            elif command -v wget &> /dev/null; then
                wget -qO- https://astral.sh/uv/install.sh | sh
            else
                print_error "Neither curl nor wget is available. Please install one of them first."
                exit 1
            fi
            
            # Add uv to current shell PATH
            if [ -f "$HOME/.cargo/env" ]; then
                source "$HOME/.cargo/env"
            fi
            export PATH="$HOME/.cargo/bin:$PATH"
            
            if command -v uv &> /dev/null; then
                print_success "uv installed successfully!"
            else
                print_error "Failed to install uv. Please try manually: https://github.com/astral-sh/uv"
                print_info "You may need to restart your terminal and run this script again."
                exit 1
            fi
            ;;
        Windows)
            print_error "Please install uv manually on Windows:"
            print_info "  Using PowerShell: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
            print_info "  Or using pip: pip install uv"
            exit 1
            ;;
        *)
            print_error "Unsupported operating system: $OSTYPE"
            print_info "Please install uv manually: https://github.com/astral-sh/uv"
            exit 1
            ;;
    esac
fi

echo ""

# 2. Select Python version
print_info "Available Python versions:"
echo "  1) Python 3.9"
echo "  2) Python 3.10"
echo "  3) Python 3.11 (recommended)"
echo "  4) Python 3.12"
echo "  5) Use system default Python"
echo ""

# Use default value if running non-interactively
if [ -t 0 ]; then
    read -p "Select Python version (1-5) [default: 1]: " python_choice
    python_choice=${python_choice:-1}
else
    print_info "Running in non-interactive mode, using default (Python 3.9)"
    python_choice=1
fi

case $python_choice in
    1)
        PYTHON_VERSION="3.9"
        VENV_NAME=".venv-py39"
        ;;
    2)
        PYTHON_VERSION="3.10"
        VENV_NAME=".venv-py310"
        ;;
    3)
        PYTHON_VERSION="3.11"
        VENV_NAME=".venv-py311"
        ;;
    4)
        PYTHON_VERSION="3.12"
        VENV_NAME=".venv-py312"
        ;;
    5)
        PYTHON_VERSION="python3"
        VENV_NAME=".venv"
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""

# 3. Create virtual environment
if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment $VENV_NAME already exists"
    
    # Handle non-interactive mode
    if [ -t 0 ]; then
        read -p "Do you want to delete and recreate it? (y/N): " recreate
    else
        print_info "Running in non-interactive mode, using existing environment"
        recreate="n"
    fi
    
    if [[ $recreate =~ ^[Yy]$ ]]; then
        print_info "Removing old virtual environment..."
        rm -rf "$VENV_NAME"
    else
        print_info "Skipping environment creation, using existing one"
        SKIP_CREATE=true
    fi
fi

if [ "${SKIP_CREATE:-false}" != true ]; then
    print_info "Creating Python $PYTHON_VERSION virtual environment: $VENV_NAME"
    
    # Try to create virtual environment
    if uv venv --python "$PYTHON_VERSION" "$VENV_NAME"; then
        print_success "Virtual environment created successfully!"
    else
        print_error "Failed to create virtual environment"
        print_info "Possible reasons:"
        print_info "  - Python $PYTHON_VERSION is not installed on your system"
        print_info "  - uv cannot find Python $PYTHON_VERSION"
        print_info ""
        print_info "Solutions:"
        print_info "  - Install Python $PYTHON_VERSION from https://www.python.org/"
        print_info "  - Use option 5 to use system default Python"
        print_info "  - Specify full path to Python executable"
        exit 1
    fi
fi

echo ""

# 4. Install dependencies
print_info "Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    if uv pip install --python "$VENV_NAME" -r requirements.txt; then
        print_success "Dependencies installed successfully!"
    else
        print_error "Failed to install dependencies"
        print_info "You can try installing manually:"
        print_info "  source $VENV_NAME/bin/activate"
        print_info "  pip install -r requirements.txt"
        exit 1
    fi
else
    print_error "requirements.txt not found"
    exit 1
fi

echo ""

# 5. Verify installation
print_info "Verifying installed packages..."
echo ""
uv pip list --python "$VENV_NAME" || {
    print_warning "Could not list packages, but installation may have succeeded"
}

echo ""
echo "========================================"
print_success "Environment setup complete!"
echo "========================================"
echo ""

# Determine activation command based on OS
case "$OS_TYPE" in
    Windows)
        ACTIVATE_CMD="$VENV_NAME\\Scripts\\activate"
        ;;
    *)
        ACTIVATE_CMD="source $VENV_NAME/bin/activate"
        ;;
esac

print_info "To activate the virtual environment:"
echo "  $ACTIVATE_CMD"
echo ""
print_info "To run Jupyter Notebook:"
echo "  jupyter notebook"
echo ""
print_info "Or in VS Code:"
echo "  1. Open a .ipynb file"
echo "  2. Click on kernel selector in top right"
echo "  3. Select the $VENV_NAME environment"
echo ""
print_info "To deactivate the virtual environment:"
echo "  deactivate"
echo ""

# Additional platform-specific notes
case "$OS_TYPE" in
    WSL)
        print_info "WSL Note: If using VS Code, install the 'Remote - WSL' extension"
        ;;
    Windows)
        print_info "Windows Note: You may need to run 'Set-ExecutionPolicy RemoteSigned' in PowerShell"
        ;;
esac
