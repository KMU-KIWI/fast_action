import setuptools


def main():
    setuptools.setup(
        name="fast_action",
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
        version="0.1.0",
        description="",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.7",
        install_requires=[
            "torch",
        ],
    )


if __name__ == "__main__":
    main()
