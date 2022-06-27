import setuptools

def read(path):
    with open(path, encoding = 'utf-8') as f:
        return f.read()


setuptools.setup(
    name="replication",
    version="0.0.1",
    author="Akshil Patel, Jordan Wei Taylor",
    author_email="ap2078@bath.ac.uk, jt2006@bath.ac.uk",
    description="Implementations of various machine learning algorithms.",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/akshilpatel/ai_replications",
    project_urls={
        "Bug Tracker": "https://github.com/akshilpatel/ai_replications/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
    license=read('LICENSE'), 
    install_requires=read('requirements.txt').split()
)