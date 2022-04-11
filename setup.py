from setuptools import find_namespace_packages, setup

setup(
    name="kitsunetorch",
    version="0.2.3",
    description="Kitsune anomaly detection model implemented in PyTorch.",
    author="Guillem Orellana Trullols",
    author_email="guillem.orellana@gmail.com",
    maintainer="Luís Seabra",
    maintainer_email="luismavseabra@innowave.tech",
    packages=find_namespace_packages(include=["kitsune"]),
    python_requires="~=3.8",
    install_requires=[
        "typer",
        "torch~=1.11",
        "torchdata~=0.3",
        "pandas~=1.2",
        "scikit-learn~=0.24",
        "scipy~=1.8",
        "tqdm",
    ],
    zip_safe=False,
)
