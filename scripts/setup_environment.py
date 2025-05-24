import subprocess
import sys
import os

def install_package(package, use_wheel=True):
    print(f"Installing {package}...")
    cmd = [sys.executable, "-m", "pip", "install"]
    if use_wheel:
        cmd.append("--only-binary=:all:")
    cmd.append(package)
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {str(e)}")
        print("Trying without wheel...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {str(e)}")
            print("Continuing with other packages...")

def main():
    print("Setting up environment for CAD Fusion Lab...")
    
    # Core dependencies (use wheels)
    core_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "beautifulsoup4",
        "requests",
        "tqdm"
    ]
    
    # NLP dependencies (some may need compilation)
    nlp_packages = [
        "nltk",
        "textblob",
        "googletrans==3.1.0a0",
        "textaugment",
        "nlpaug",
        "sentence-transformers"
    ]
    
    # Image processing dependencies (use wheels)
    image_packages = [
        "Pillow",
        "opencv-python-headless",
        "albumentations",
        "imgaug",
        "scikit-image"
    ]
    
    # ML and visualization dependencies (use wheels)
    ml_packages = [
        "wandb",
        "tensorboard",
        "matplotlib",
        "seaborn",
        "plotly"
    ]
    
    # Install all packages
    all_packages = core_packages + nlp_packages + image_packages + ml_packages
    
    for package in all_packages:
        install_package(package)
    
    # Download NLTK data
    print("Downloading NLTK data...")
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    
    print("\nEnvironment setup complete!")
    print("You can now run your data collection and training scripts.")

if __name__ == "__main__":
    main() 