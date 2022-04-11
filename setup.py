from setuptools import setup, find_namespace_packages

setup(
    name="kitsunetorch",
    version="0.2.1",
    description="Kitsune anomaly detection model implemented in PyTorch.",
    author="Guillem Orellana Trullols",
    author_email="guillem.orellana@gmail.com",
    maintainer="Luís Seabra",
    maintainer_email="luismavseabra@innowave.tech",
    packages=find_namespace_packages(include=["kitsune"]),
    python_requires="~=3.8",
    install_requires=[
        "typer~=0.3.2",
        "torch~=1.11.0",
        "torchdata~=0.3.0",
        "pandas~=1.2",
        "scikit-learn~=0.24",
        "scipy~=1.8.0",
        "tqdm~=4.64",
    ],
    zip_safe=False,
)
