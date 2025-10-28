from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Core dependencies
install_requires = [
    # Deep Learning
    'torch>=1.10.0',
    'torchvision>=0.11.0',
    
    # Diffusion Models
    'diffusers>=0.11.1',
    
    # Data Processing
    'numpy>=1.20.0',
    'zarr>=2.10.0',
    'h5py>=3.6.0',
    'opencv-python>=4.5.0',
    
    # Robotics & Transformations
    'scipy>=1.7.0',
    
    # Configuration & Logging
    'hydra-core>=1.2.0',
    'omegaconf>=2.2.0',
    'wandb>=0.12.0',
    
    # Utilities
    'tqdm>=4.62.0',
    'filelock>=3.4.0',
    'threadpoolctl>=3.0.0',
    'av>=9.0.0',
    
    # Visualization (optional but recommended)
    'matplotlib>=3.5.0',
    'imageio>=2.9.0',
    'imageio-ffmpeg>=0.4.5',
]

# Development dependencies
dev_requires = [
    'pytest>=7.0.0',
    'pytest-cov>=3.0.0',
    'black>=22.0.0',
    'flake8>=4.0.0',
    'ipython>=8.0.0',
    'jupyter>=1.0.0',
]

# Real robot deployment dependencies
robot_requires = [
    'pyrealsense2>=2.50.0',  # Intel RealSense camera
    # Add your robot-specific dependencies here
]

setup(
    name='fadp',
    version='1.0.0',
    description='Force-Aware Diffusion Policy for Robot Manipulation',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    author='FADP Contributors',
    author_email='',
    url='https://github.com/Elycyx/Force-Aware-Diffusion-Policy',
    license='MIT',
    
    packages=find_packages(exclude=['tests', 'tests.*', 'data', 'data.*']),
    include_package_data=True,
    
    python_requires='>=3.8',
    install_requires=install_requires,
    
    extras_require={
        'dev': dev_requires,
        'robot': robot_requires,
        'all': dev_requires + robot_requires,
    },
    
    entry_points={
        'console_scripts': [
            'fadp-train=fadp.scripts.train:main',
            'fadp-eval=fadp.scripts.eval:main',
            'fadp-convert-dataset=scripts.convert_hdf5_to_fadp:main',
        ],
    },
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Robotics',
    ],
    
    keywords='robot-learning diffusion-policy force-control manipulation',
)
