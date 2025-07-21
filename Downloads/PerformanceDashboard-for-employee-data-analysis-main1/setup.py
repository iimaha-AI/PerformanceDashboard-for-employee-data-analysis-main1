from setuptools import setup, find_packages

setup(
    name="employee_events",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "employee_events": ["employee_events.db"],
    },
    install_requires=[
        "email-validator>=2.2.0",
        "flask>=3.1.1",
        "flask-sqlalchemy>=3.1.1",
        "gunicorn>=23.0.0",
        "matplotlib>=3.9.2",
        "numpy>=1.26.4",
        "pandas>=2.2.3",
        "psycopg2-binary>=2.9.10",
        "pytest>=8.4.1",
        "python-fasthtml>=0.12.21",
        "scikit-learn>=1.5.2",
        "scipy>=1.14.1",
        "setuptools>=80.9.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for employee events analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
