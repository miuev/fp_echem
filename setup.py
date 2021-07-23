import setuptools

# read in version of canela package
with open('fpec/_version.py', 'r') as fid:
    # this will create the __version__ variable to be used below
    exec(fid.read())

# use README file to create long description of package
# ignore images (lines that start with '![')
with open('README.md', 'r') as readme:
    long_description = ''.join([i for i in readme.readlines()
                                if not i.startswith('![')])

setuptools.setup(
    name="fpec",
    version=__version__,
    author="Evan Miu",
    author_email="evm@pitt.edu",
    description="first-principles current-potential relationships",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/miuev/fp_echem",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=['numpy>=1.21.0']
)
