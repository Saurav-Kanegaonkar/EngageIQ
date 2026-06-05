"""Rich, disambiguated definitions of the 15 domains — used by the LLM/agent
classifier (and good source material for the technical brief). Each says what the
domain IS, what it is NOT, and calls out the common false-positive traps.
"""
from __future__ import annotations

OTHER = "other"  # for records that genuinely fit none of the 15 (noise/off-topic)

DOMAIN_DEFINITIONS = {
    "machine_learning": "Training and using ML / deep-learning MODELS: neural networks, model architectures, training/fine-tuning, ML frameworks (PyTorch, TensorFlow, scikit-learn), computer vision, applied ML. NOT general LLM/AI-agent topics (-> ai_research); NOT generic data wrangling (-> python_data_eng).",
    "ai_research": "AI/ML RESEARCH & frontier work: how LLMs/transformers work, novel techniques, model architectures, alignment, agentic-systems research, papers/labs. NOT 'how to use ChatGPT' tutorials or AI-powered developer tools (-> developer_tools/beginner_coding), and NOT merely 'mentions AI'. NOT classical ML training (-> machine_learning).",
    "python_data_eng": "Data pipelines, ETL, pandas/dataframes, data warehousing, analytics engineering, processing/cleaning datasets in Python. NOT model training (-> machine_learning). Beware: 'data' + 'engineering' appearing in unrelated text (jobs, civil/financial engineering).",
    "devops_k8s": "Kubernetes, Docker, containers, CI/CD, Terraform, Helm, observability, infrastructure/platform engineering. NOT cloud-provider product news unless it's about operating infra (-> cloud_apis).",
    "cloud_apis": "AWS / Google Cloud / Azure services, serverless / Lambda, cloud storage, managed cloud services and their APIs. NOT self-managed Kubernetes/infra (-> devops_k8s).",
    "cybersecurity": "Vulnerabilities, CVEs, exploits, penetration testing, malware, application/network security, infosec. NOT vague AI-safety opinion unless it's a concrete security issue.",
    "frontend_web": "React, Vue, Svelte, JavaScript/TypeScript, CSS, HTML, web UI, browser apps, web frameworks. NOT mobile apps (-> mobile_dev). TRAP: 'react' the verb / 'reactor' (nuclear) are NOT this domain.",
    "developer_tools": "CLIs, IDEs/editors, build tools, debuggers, SDKs, linters, code generators, developer-productivity tooling — the TOOL itself for developers. NOT a product in a specific domain.",
    "b2b_saas": "Software-as-a-service / business software: pricing, go-to-market, startups, founders, subscriptions, the business/product angle. NOT the underlying tech domain.",
    "blockchain": "Cryptocurrency, smart contracts, Solidity, Ethereum, DeFi, NFTs, decentralized apps, crypto protocols. NOT traditional fintech/payments (Stripe/PayPal) unless crypto.",
    "gamedev_cpp": "BUILDING video games: game engines (Unreal, Unity, Godot, or web/JS engines), graphics/rendering, shaders, game physics, gameplay programming. (Spec emphasizes C++, but ANY game-making counts.) TRAP: gaming news / reviews / culture (e.g. GTA economics) is NOT this — it must be about MAKING games.",
    "mobile_dev": "iOS, Android, Swift/SwiftUI, Kotlin, Flutter, React Native — building mobile apps. TRAP: 'flutter' the butterfly/song/emotion is NOT this domain.",
    "embedded_systems": "Microcontrollers, firmware, RTOS, Arduino, STM32, C for devices, IoT hardware, bare-metal/low-level. TRAP: metaphorical 'embedded' (sociology, 'embedded in daily life') and generic hardware business news are NOT this.",
    "beginner_coding": "Learning to code: beginner tutorials, first projects, programming fundamentals, career-starter content for newcomers. NOT intermediate/advanced engineering.",
    "open_source_trending": "Notable/popular open-source projects, repos, releases, and community/contribution topics that DON'T fit a more specific domain above — a catch-all for genuine cross-cutting tech/OSS. NOT non-tech noise (politics, prayers, TV shows).",
}


def as_prompt_block() -> str:
    lines = [f"- {k}: {v}" for k, v in DOMAIN_DEFINITIONS.items()]
    lines.append(f"- {OTHER}: none of the above — genuine noise / off-topic / non-technical.")
    return "\n".join(lines)
