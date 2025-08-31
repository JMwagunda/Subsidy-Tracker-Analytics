# Dev Container Setup for Fraud Detection App

This directory contains the Dev Container configuration for the Fraud Detection Streamlit application.

## What's Included

- **devcontainer.json**: Main configuration file
- **Dockerfile**: Custom container setup with all dependencies
- **README.md**: This documentation

## Prerequisites

1. **Docker Desktop**: Install and run Docker Desktop
2. **VS Code**: Install Visual Studio Code
3. **Dev Containers Extension**: Install the "Dev Containers" extension in VS Code

## How to Use

### Method 1: VS Code Command Palette
1. Open VS Code in your project folder
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
3. Type "Dev Containers: Reopen in Container"
4. Select the command and wait for the container to build

### Method 2: VS Code Notification
1. Open the project folder in VS Code
2. VS Code will detect the `.devcontainer` folder
3. Click "Reopen in Container" when prompted

## What Happens

1. **Container Build**: Docker builds a container with Python 3.11 and all dependencies
2. **Extensions Install**: VS Code installs Python development extensions
3. **Dependencies Install**: All packages from `requirements.txt` are installed
4. **Port Forward**: Streamlit port (8501) is automatically forwarded

## Running the App

Once inside the container:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Benefits

- ✅ **Consistent Environment**: Same Python version and dependencies for everyone
- ✅ **No Local Setup**: No need to install Python or packages locally
- ✅ **Isolated**: Won't conflict with other Python projects
- ✅ **Ready to Code**: All development tools pre-configured
- ✅ **Team Collaboration**: Same setup across different machines

## Troubleshooting

- **Container won't build**: Check Docker Desktop is running
- **Port conflicts**: Change port in `devcontainer.json` if 8501 is in use
- **Slow build**: First build takes longer; subsequent builds are faster due to caching