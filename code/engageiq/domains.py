"""The 15 technical domains the dataset must span (from the spec), plus
starter per-source query config.

🟢 PERMANENT — DOMAINS / DOMAIN_LABELS (the 15-domain taxonomy itself).
🟡 PROVISIONAL — the per-source query maps below. We will tune these for
   coverage and balance once we see real data volume/quality per domain.
"""
from __future__ import annotations

DOMAINS = [
    "machine_learning", "devops_k8s", "open_source_trending", "developer_tools",
    "cybersecurity", "frontend_web", "b2b_saas", "blockchain", "python_data_eng",
    "gamedev_cpp", "ai_research", "embedded_systems", "cloud_apis", "mobile_dev",
    "beginner_coding",
]

DOMAIN_LABELS = {
    "machine_learning": "Machine Learning",
    "devops_k8s": "DevOps / Kubernetes",
    "open_source_trending": "Trending Open-Source",
    "developer_tools": "Developer Tools",
    "cybersecurity": "Cybersecurity",
    "frontend_web": "Frontend (React/Web)",
    "b2b_saas": "B2B SaaS",
    "blockchain": "Blockchain",
    "python_data_eng": "Python Data Eng",
    "gamedev_cpp": "GameDev (C++)",
    "ai_research": "AI Research",
    "embedded_systems": "Embedded (C/RTOS)",
    "cloud_apis": "Cloud APIs",
    "mobile_dev": "Mobile Dev (iOS/Flutter)",
    "beginner_coding": "Beginner Coding",
}

# Reddit subreddits per domain (provisional)
SUBREDDITS = {
    "machine_learning": ["MachineLearning", "learnmachinelearning"],
    "devops_k8s": ["devops", "kubernetes"],
    "open_source_trending": ["opensource", "coolgithubprojects"],
    "developer_tools": ["programming", "commandline"],
    "cybersecurity": ["netsec", "cybersecurity"],
    "frontend_web": ["reactjs", "webdev"],
    "b2b_saas": ["SaaS", "startups"],
    "blockchain": ["ethdev", "solidity"],
    "python_data_eng": ["dataengineering", "Python"],
    "gamedev_cpp": ["gamedev", "cpp"],
    "ai_research": ["MachineLearning", "artificial"],
    "embedded_systems": ["embedded", "microcontrollers"],
    "cloud_apis": ["aws", "googlecloud"],
    "mobile_dev": ["FlutterDev", "iOSProgramming"],
    "beginner_coding": ["learnprogramming", "AskProgramming"],
}

# Dev.to tags per domain (provisional)
DEVTO_TAGS = {
    "machine_learning": ["machinelearning", "ai"],
    "devops_k8s": ["devops", "kubernetes"],
    "open_source_trending": ["opensource", "github"],
    "developer_tools": ["tooling", "productivity"],
    "cybersecurity": ["security", "cybersecurity"],
    "frontend_web": ["react", "webdev"],
    "b2b_saas": ["saas", "startup"],
    "blockchain": ["blockchain", "web3"],
    "python_data_eng": ["python", "dataengineering"],
    "gamedev_cpp": ["gamedev", "cpp"],
    "ai_research": ["ai", "machinelearning"],
    "embedded_systems": ["embedded", "iot"],
    "cloud_apis": ["aws", "cloud"],
    "mobile_dev": ["flutter", "android"],
    "beginner_coding": ["beginners", "codenewbie"],
}

# GitHub free-text search hints per domain (provisional)
GITHUB_QUERIES = {
    "machine_learning": "machine-learning OR deep-learning",
    "devops_k8s": "kubernetes OR terraform OR devops",
    "open_source_trending": "stars:>500 pushed:>2026-05-01",
    "developer_tools": "developer-tools OR cli",
    "cybersecurity": "security OR pentesting",
    "frontend_web": "react OR frontend",
    "b2b_saas": "saas OR b2b",
    "blockchain": "blockchain OR solidity OR web3",
    "python_data_eng": "data-engineering OR etl language:python",
    "gamedev_cpp": "game-engine language:cpp",
    "ai_research": "llm OR transformer OR diffusion",
    "embedded_systems": "embedded OR rtos language:c",
    "cloud_apis": "aws OR cloud-api",
    "mobile_dev": "flutter OR ios OR android",
    "beginner_coding": "good-first-issues OR beginner-friendly",
}

# Free-text search keyword per domain (Hacker News Algolia search, Bluesky) (provisional)
KEYWORDS = {
    "machine_learning": "machine learning",
    "devops_k8s": "kubernetes",
    "open_source_trending": "open source",
    "developer_tools": "developer tools",
    "cybersecurity": "cybersecurity",
    "frontend_web": "react",
    "b2b_saas": "saas",
    "blockchain": "blockchain",
    "python_data_eng": "data engineering",
    "gamedev_cpp": "game development",
    "ai_research": "large language model",
    "embedded_systems": "embedded systems",
    "cloud_apis": "aws",
    "mobile_dev": "flutter",
    "beginner_coding": "beginner programming",
}
