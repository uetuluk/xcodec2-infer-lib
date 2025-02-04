from setuptools import setup, find_packages

setup(
    name='xcodec2-infer-lib',
    version='0.1.0',
    packages=find_packages(),
    python_requires='>=3.12',
    install_requires=[],
    license='CC-BY-4.0',
    author='Utku Ege Tuluk',
    description='Trying to achieve M chip Macbook support for https://huggingface.co/HKUSTAudio/xcodec2.',
    long_description=open('README.md').read(),
    
    long_description_content_type='text/markdown',
    url='https://github.com/uetuluk/xcodec2-infer-lib',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
    ],
)
