from setuptools import setup, find_packages


def get_requirements(file_path: str):
    """Read requirements.txt and return list of dependencies, excluding -e ."""
    requirements = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and line != "-e .":
                requirements.append(line)
    return requirements


setup(
    name="inventory-demand-forecasting",
    version="0.1.0",
    author="Harsh",
    email="nimsatkarharsh@gmail.com",
    description="End-to-end ML pipeline for inventory demand forecasting",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
