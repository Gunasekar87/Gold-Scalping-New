from setuptools import setup, find_packages

setup(
    name="trading_mad",
    version="4.0.4",
    description="AETHER Trading System - The Global Brain",
    author="Gunasekar87",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch==2.0.1",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "onnxruntime==1.16.3",
        "stable-baselines3==2.1.0",
        "shimmy==1.3.0",
        "gymnasium==0.29.1",
        "pyzmq==25.1.1",
        "pandas==2.1.3",
        "numpy<2.0.0",
        "pydantic==2.5.0",
        "python-dotenv==1.0.0",
        "pyyaml==6.0.1",
        "requests==2.31.0",
        "supabase==2.8.1",
        "psycopg2-binary==2.9.9",
        "ccxt==4.1.63",
        "MetaTrader5==5.0.45",
        "yfinance==0.2.18",
        "aiosqlite==0.19.0",
        "asyncpg==0.29.0",
    ],
    entry_points={
        "console_scripts": [
            "trading-mad=main_bot:main",
        ],
    },
    python_requires=">=3.10",
)
