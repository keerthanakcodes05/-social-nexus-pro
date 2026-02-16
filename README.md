# 🎯 Social Nexus Pro
### AI-Powered Social Media Content Generator

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-AI-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📌 Overview

**Social Nexus Pro** is a production-ready AI-powered social media content generation platform built with Streamlit. It helps content creators, marketers, and businesses generate high-quality captions, trending hashtags, engagement predictions, and content calendars — all in one place.

> ✅ Works with or without an API key — AI mode with Groq/OpenAI, or smart Template fallback.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🤖 AI Caption Generation | Powered by Groq (LLaMA 3.1) or OpenAI (GPT-4o-mini) |
| 📊 Engagement Prediction | Predicts likes, comments, shares, and reach |
| #️⃣ Smart Hashtags | Trending hashtag recommendations by category |
| 📅 Content Calendar | 7-day auto-generated posting schedule |
| 📥 Multi-format Export | Download as PDF, JSON, or CSV |
| 🗄️ SQLite History | Saves all generated content locally |
| 🎨 Premium UI | Animated glassmorphism design |
| 🔄 Fallback Mode | Template mode when no API key is set |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOURUSERNAME/social-nexus-pro.git
cd social-nexus-pro
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Open `.env` and add your API keys.

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🔑 API Keys

| Provider | Where to Get | Cost |
|----------|-------------|------|
| Groq | console.groq.com | ✅ Free |
| OpenAI | platform.openai.com | 💳 Paid |

No API key? The app runs in Template Mode automatically.

---

## 📁 Project Structure
```
social-nexus-pro/
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
├── README.md                # Project documentation
└── LICENSE                  # MIT License
```

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Plotly, Custom CSS
- **AI:** Groq API (LLaMA 3.1 8B), OpenAI (GPT-4o-mini)
- **Database:** SQLite3
- **Data:** Pandas
- **Export:** ReportLab (PDF), CSV, JSON
- **Config:** python-dotenv

---

## ☁️ Deploy Free on Streamlit Cloud

1. Push this repo to GitHub (Public)
2. Go to share.streamlit.io
3. Select repo → set main file as `app.py`
4. Add secrets: `GROQ_API_KEY` and `OPENAI_API_KEY`
5. Click Deploy 🚀

---

## 📄 License

MIT License — see LICENSE file for details.

## 👤 Author

**Kandakatla Keerthana** — GitHub: https://github.com/keerthanakcodes05
```



