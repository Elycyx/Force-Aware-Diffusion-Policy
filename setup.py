"""
Force-Aware Diffusion Policy (FADP) Setup Configuration
"""

from setuptools import setup, find_packages
import os

# Read the long description from README
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Core dependencies
INSTALL_REQUIRES = [
    'torch>=2.0.0',
    'torchvision>=0.15.0',
    'numpy>=1.20.0',
    'scipy>=1.7.0',
    'zarr>=2.10.0',
    'numcodecs>=0.10.0',
    'opencv-python>=4.5.0',
    'pillow>=9.0.0',
    'h5py>=3.0.0',
    'tqdm>=4.60.0',
    'click>=8.0.0',
    'pyyaml>=5.4.0',
    'matplotlib>=3.3.0',
    'av>=9.0.0',
    'wandb>=0.12.0',
    'hydra-core>=1.2.0',
    'omegaconf>=2.2.0',
    'dill>=0.3.4',
    'einops>=0.6.0',
    'diffusers>=0.11.1',
    'accelerate>=0.20.0',
    'transformers>=4.30.0',
    'timm>=0.9.0',
]

# Optional dependencies for specific features
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=3.0.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
        'isort>=5.10.0',
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
    ],
}

setup(
    name='fadp',
    version='1.0.0',
    author='FADP Team',
    author_email='',
    description='Force-Aware Diffusion Policy for Robot Manipulation',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/universal_manipulation_interface',
    packages=find_packages(exclude=['data', 'docs', 'examples', 'tests']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        'console_scripts': [
            'fadp-train=train:main',
            'fadp-convert=convert_session_to_zarr:main',
            'fadp-test=test_fadp_model:main',
            'fadp-inspect=inspect_zarr:main',
        ],
    },
    include_package_data=True,
    package_data={
        'diffusion_policy': [
            'config/**/*.yaml',
            'config/**/*.yml',
        ],
    },
    zip_safe=False,
)

