import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class PackageError(Exception):
    """Custom exception for package management errors"""
    pass

def build_package(package_dir: str = ".") -> None:
    """Build the package."""
    try:
        subprocess.run(["python", "-m", "build", package_dir], check=True)
        logger.info("Package built successfully")
    except subprocess.CalledProcessError as e:
        raise PackageError(f"Package build failed: {str(e)}")

def upload_to_pypi(username: Optional[str] = None, password: Optional[str] = None) -> None:
    """Upload package to PyPI."""
    try:
        cmd = ["twine", "upload", "dist/*"]
        if username and password:
            cmd.extend(["--username", username, "--password", password])
        subprocess.run(cmd, check=True)
        logger.info("Package uploaded to PyPI successfully")
    except subprocess.CalledProcessError as e:
        raise PackageError(f"PyPI upload failed: {str(e)}")

def setup_development(package_dir: str = ".") -> None:
    """Setup package for development."""
    try:
        subprocess.run(["pip", "install", "-e", package_dir], check=True)
        logger.info("Development setup completed")
    except subprocess.CalledProcessError as e:
        raise PackageError(f"Development setup failed: {str(e)}")
