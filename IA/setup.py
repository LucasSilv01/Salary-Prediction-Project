
from setuptools import setup, find_packages

setup(
    name='salary_project',
    version='0.1.0',
    description='Projeto de previsão de salários (regressão e classificação)',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn'
    ],
    author='Auto-generated',
    python_requires='>=3.8',
)
