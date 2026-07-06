# 🧠 Havan Vision

**An emotion-aware AI chat companion** — combines live webcam facial-emotion sensing with text-based sentiment/emotion analysis to adapt an LLM's tone in real time, backed by a crisis-safety interceptor.

![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-F55036?style=for-the-badge)
![face--api.js](https://img.shields.io/badge/face--api.js-vladmandic-orange?style=for-the-badge)
![JWT](https://img.shields.io/badge/Auth-JWT-000000?style=for-the-badge&logo=jsonwebtokens&logoColor=white)
![Tests](https://img.shields.io/badge/backend_tests-62_passing-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

---

## 📖 Table of Contents

- [🧠 Havan Vision](#-havan-vision)
  - [📖 Table of Contents](#-table-of-contents)
  - [🧭 What This Is](#-what-this-is)
  - [🏗 Architecture](#-architecture)
  - [🎭 Emotion Pipeline](#-emotion-pipeline)
  - [🚨 Crisis Safety Layer](#-crisis-safety-layer)
  - [🗄 Data Model](#-data-model)
  - [🔌 API Surface](#-api-surface)
  - [🖥 Frontend Structure](#-frontend-structure)
  - [🖼 Screenshots](#-screenshots)
  - [⚙️ Local Setup](#️-local-setup)
    - [Backend](#backend)
    - [Frontend](#frontend)
  - [🔑 Environment Variables](#-environment-variables)
  - [🧪 Testing](#-testing)
  - [⚠️ Known Limitations](#️-known-limitations)
  - [📄 License](#-license)

---

## 🧭 What This Is

Havan Vision reads emotion from **two independent channels at once** — the words
a user types, and their live facial expression via webcam — and feeds both into
an LLM system prompt so the AI's response tone actually matches how the person
is feeling, not just what they wrote.

It is not a toy sentiment demo: it has a dedicated **crisis-detection safety
layer** that runs on every message regardless of which emotion backend is
active, a **fail-fast configuration system** that refuses to start with
missing secrets, and **rate-limited authentication** to resist credential
stuffing.

```
┌───────────────────────────────────────────────────────────────────┐
│                          HAVAN VISION                              │
├────────────────────────────┬────────────────────────────────────────┤
│         FRONTEND            │              BACKEND                   │
│   React 19 + Vite            │        Flask 3 + SQLAlchemy             │
│                              │                                         │
│  ┌────────────┐ ┌──────────┐│  ┌───────────────┐  ┌─────────────────┐│
│  │ WebcamScanner│ │ ChatInput││  │ EmotionEngine  │  │ LLM Service      ││
│  │ (face-api.js)│ │          ││  │ (text emotion  │  │ (Groq Llama 3.1  ││
│  └──────┬───────┘ └────┬─────┘│  │  + crisis flag)│  │  + rule fallback)││
│         │              │      │  └───────┬───────┘  └────────┬────────┘│
│  ┌──────▼──────────────▼─────┐│          │                   │         │
│  │      AuthContext (JWT)     ││          └─────────┬─────────┘         │
│  └────────────────────────────┘│                    ▼                   │
│                              │           ┌─────────────────────┐        │
│                              │           │ Crisis Interceptor   │        │
│                              │           │ (always runs first)  │        │
│                              │           └─────────────────────┘        │
└────────────────────────────┴────────────────────────────────────────┘
```

---

## 🏗 Architecture

```
  Browser (webcam + keyboard)
        │
        ├─► face-api.js  ──── visual_emotion (client-side inference)
        │
        └─► POST /api/chat/sessions/<id>/messages ─────────┐
                                                             ▼
                                              ┌───────────────────────┐
                                              │   Flask (JWT-gated)    │
                                              │                         │
                                              │  1. emotion_engine      │
                                              │     .analyze(text)      │
                                              │     → primary_emotion,  │
                                              │       sentiment,        │
                                              │       is_crisis         │
                                              │                         │
                                              │  2. llm_service         │
                                              │     .generate_response()│
                                              │     ┌─────────────────┐│
                                              │     │ is_crisis?      ││
                                              │     │  → safety msg   ││
                                              │     │ groq_api_key?   ││
                                              │     │  → Groq LLM     ││
                                              │     │  → rule fallback││
                                              │     └─────────────────┘│
                                              └───────────┬─────────────┘
                                                           ▼
                                                  PostgreSQL / SQLite
                                                  (Message + emotion data)
```

**Key design choice:** the crisis check runs **before** any LLM call — a
message flagged as crisis never reaches Groq at all. It returns a fixed,
region-aware safety response directly from `llm_service.py`.

---

## 🎭 Emotion Pipeline

`app/services/emotion_engine.py` implements a **two-tier strategy**:

| Tier | Trigger | Behavior |
|---|---|---|
| **HuggingFace transformers** | `USE_ML_MODELS=true` env var set | Loads `j-hartmann/emotion-english-distilroberta-base` (7-way emotion) and `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **Rule-based fallback** | Default, or if ML models fail to load | Lightweight keyword scoring across 6 emotion categories, with neutral computed proportionally to keyword-match density |

This is a deliberate **kill switch**, not an accident — the code comments
explicitly call out preventing OOM crashes on 512MB free-tier hosting (Render).
The fallback isn't a degraded afterthought; it has its own dedicated test file
(`test_emotion_engine.py`) with 17 passing cases, including neutral-scaling and
intensity-range assertions.

**Emotion → tone mapping** happens via `EMOTION_COLORS` and `EMOTION_EMOJI`
lookup tables, feeding both the LLM system prompt and the frontend's
`EmotionBadge` component.

---

## 🚨 Crisis Safety Layer

A dedicated keyword-based check runs on **every single message**, independent
of which emotion backend is active:

```python
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die", "self-harm",
    "harm myself", "cutting myself", "no reason to live", "hopeless",
    "can't go on", "give up on life",
]
```

If triggered, `llm_service.py` **short-circuits before calling Groq** and
returns region-aware crisis resources:

- 🇮🇳 **India:** iCall, Vandrevala Foundation (24/7)
- 🇺🇸 **US:** 988 Suicide & Crisis Lifeline, Crisis Text Line
- 🇬🇧 **UK:** Samaritans
- 🌍 **International:** findahelpline.com

This is covered by `test_llm_service.py::TestCrisisInterceptor`, verifying the
interceptor actually fires and actually bypasses the normal LLM path.

---

## 🗄 Data Model

```
User ──< ConversationSession ──< Message
 │              │                   │
 │              │                   ├── primary_emotion / visual_emotion
 │              │                   ├── emotion_scores (JSON)
 │              │                   └── sentiment / sentiment_score / intensity
 │              └── dominant_emotion, message_count, is_archived
 └── password_hash (bcrypt), preferences (JSON), avatar_emoji
```

| Model | Purpose |
|---|---|
| `User` | Account with bcrypt-hashed password, display preferences |
| `ConversationSession` | One chat thread; soft-deleted via `is_archived` |
| `Message` | Individual message with **both** text-derived and facial-derived emotion fields stored per message |

---

## 🔌 API Surface

All routes are JWT-gated except register/login.

| Method | Endpoint | Notes |
|---|---|---|
| `POST` | `/api/auth/register` | Rate-limited `5/min`; bcrypt hash; 8-char password minimum |
| `POST` | `/api/auth/login` | Rate-limited `5/min`; accepts username or email |
| `POST` | `/api/auth/refresh` | Refresh-token flow |
| `GET/PATCH` | `/api/auth/me` | Profile read/update |
| `POST` | `/api/auth/logout` | Stateless JWT — informs client to discard tokens; does not server-side revoke |
| `GET/POST` | `/api/chat/sessions` | Paginated session list / create |
| `GET/DELETE` | `/api/chat/sessions/<id>` | Fetch with messages / soft-delete |
| `POST` | `/api/chat/sessions/<id>/messages` | Runs emotion analysis + crisis check + LLM response |

---

## 🖥 Frontend Structure

```
frontend/src/
├── components/
│   ├── WebcamScanner.jsx     face-api.js live facial emotion capture
│   ├── ChatInput.jsx         message composer
│   ├── MessageBubble.jsx     per-message display with emotion badge
│   ├── EmotionBadge.jsx      color/emoji rendering from backend emotion data
│   ├── TypingIndicator.jsx
│   ├── Sidebar.jsx           session list
│   └── ProtectedRoute.jsx    JWT-gated route guard
├── pages/
│   ├── Login.jsx / Register.jsx
│   └── Chat.jsx
├── context/
│   └── AuthContext.jsx       token storage + refresh logic
├── utils/
│   └── emotions.js           (+ emotions.test.js)
└── api.js                    (+ api.test.js)
```

Built on **React 19 + Vite**, using **`@vladmandic/face-api`** for in-browser
facial emotion inference (no server-side image upload) and **`jwt-decode`**
for client-side token expiry checks.

---

## 🖼 Screenshots

> Real screenshots committed at `docs/screenshots/` — reference them directly
> rather than embedding placeholders:

- `docs/screenshots/login_page.png`
- `docs/screenshots/chat_interface.png`
- `docs/screenshots/emotion_sensor.png`

```markdown
![Login](docs/screenshots/login_page.png)
![Chat Interface](docs/screenshots/chat_interface.png)
![Emotion Sensor](docs/screenshots/emotion_sensor.png)
```

---

## ⚙️ Local Setup

### Backend

```bash
git clone https://github.com/Madhavan-dev18/havan-vision.git
cd havan-vision/backend

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # fill in real values — app will refuse to
                                  # start without SECRET_KEY / JWT_SECRET_KEY
python run.py
```

### Frontend

```bash
cd havan-vision/frontend
npm install
npm run dev
```

---

## 🔑 Environment Variables

| Variable | Required | Notes |
|---|---|---|
| `SECRET_KEY` | ✅ **hard-required** | App raises `RuntimeError` at startup if missing — no insecure fallback |
| `JWT_SECRET_KEY` | ✅ **hard-required** | Same fail-fast behavior |
| `DATABASE_URL` | ✅ | `sqlite:///havanvision.db` for local dev |
| `GROQ_API_KEY` | Optional | Without it, the app runs entirely on the rule-based fallback — never crashes |
| `ALLOWED_ORIGINS` | ✅ | Comma-separated CORS origins; regex-matches any `havan-vision*.vercel.app` preview deploy automatically |
| `USE_ML_MODELS` | Optional | `true` loads HuggingFace transformer models (~1GB+ RAM); default is the lightweight rule-based path |

---

## 🧪 Testing

**Backend: 62 tests, verified passing** across:

| File | Covers |
|---|---|
| `test_auth.py` | Register/login validation, duplicate checks, JWT refresh, profile updates |
| `test_chat.py` | Session CRUD, message send, emotion-tag propagation, auth gating |
| `test_emotion_engine.py` | Rule-based scoring, neutral/intensity ranges, crisis-keyword detection |
| `test_llm_service.py` | Crisis interceptor firing/bypass, rule-based fallback response mapping |

**Frontend:** `api.test.js`, `emotions.test.js` (Vitest).

Run locally:
```bash
# Backend
cd backend && pytest tests/ -v

# Frontend
cd frontend && npm run test
```

---

## ⚠️ Known Limitations

- **No CI pipeline** — the 62 backend tests and frontend suite exist and pass locally, but nothing runs them automatically on push/PR yet.
- **`/logout` cannot server-side revoke tokens** — this is a stateless-JWT design tradeoff, not a bug, but it means a stolen token remains valid until natural expiry regardless of logout.
- **Crisis detection is keyword-based**, not ML-based — it will miss crisis language that doesn't match the fixed keyword list.

---

## 📄 License

MIT — see [`LICENSE`](./LICENSE).