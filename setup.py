from setuptools import setup, find_packages

setup(
    name='antm',
    version='0.1.0',
    author='Hamed Rahimi',
    author_email='<hamed.rahimi@sorbonne-universite.fr',
    description='Aligned Neural Topic Model for Exploring Evolving Topics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/hamedR96/ANTM',
    packages=find_packages(),
    install_requires=["hdbscan","swifter","sentence-transformers","umap-learn","transformers","torch",
                      "gensim","plotly","matplotlib","pyarrow","kaleido"
                      ],
    
    project_urls={
        'Bug Tracker': 'https://github.com/hamedR96/ANTM/issues'
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X ",
        "Operating System :: Unix ",
        "Operating System :: Microsoft :: Windows ",
    ],

)
