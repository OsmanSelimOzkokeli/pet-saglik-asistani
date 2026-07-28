"""
Pet Sağlık Asistanı - Semptom & Aciliyet Triyaj Servisi
---------------------------------------------------------
Kullanıcının evcil hayvanı için yazdığı semptomları (ve isteğe bağlı fotoğrafı)
OpenAI ile analiz edip, olası nedenler + aciliyet seviyesi + genel ilk müdahale
bilgisi + "gerçek veterinere gitmesi gerekip gerekmediği" önerisi üretir.

ÖNEMLİ TASARIM KARARI:
Kritik/acil semptomlar sabit kodlanmış bir anahtar kelime katmanıyla tespit
edilir ve AI'nın yorumundan BAĞIMSIZ olarak her zaman "ACİL" seviyesine
zorlanır. AI, bu tür durumlarda aciliyeti düşürecek şekilde konuşamaz.

Çalıştırma (mock mod, para harcamadan):
    pip install fastapi uvicorn openai --break-system-packages
    set MOCK_MODE=1
    uvicorn app:app --port 8000

Çalıştırma (gerçek OpenAI ile):
    set OPENAI_API_KEY=sk-...
    uvicorn app:app --port 8000
"""

import base64
import os
import random
import json
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Pet Sağlık Asistanı")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Aciliyet seviyeleri
# ---------------------------------------------------------------------------

class Urgency(str, Enum):
    ACIL = "ACİL"
    YAKINDA_VET = "YAKINDA_VET"
    GOZLEMLE = "GÖZLEMLE"
    DUSUK_ENDISE = "DÜŞÜK_ENDİŞE"


# ---------------------------------------------------------------------------
# Sabit kodlanmış acil durum katmanı (AI'dan bağımsız, override edilemez)
# ---------------------------------------------------------------------------

EMERGENCY_KEYWORDS = [
    "nefes al", "nefes darlığı", "nefes alamıyor", "soluk alamıyor",
    "zehir", "zehirlenme", "yuttu", "boğuluyor", "boğazına",
    "nöbet", "kasıl", "titreme kontrolsüz", "bayıldı", "bilinç kaybı", "tepki vermiyor",
    "kanama", "kan geliyor", "çok kanıyor", "durmayan kanama",
    "şişmiş karın", "karnı şişti", "kusmadan şişme",
    "araba çarptı", "yüksekten düştü", "travma",
    "morarma", "morardı", "diş eti beyaz", "diş eti mor",
    "sıcak çarpması", "aşırı sıcak", "aşırı ısınma",
    "felç", "yürüyemiyor", "ayağa kalkamıyor",
]


def check_hardcoded_emergency(symptom_text: str) -> Optional[str]:
    text_lower = symptom_text.lower()
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text_lower:
            return keyword
    return None


# ---------------------------------------------------------------------------
# Modeller (sohbet modu)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    rol: str = Field(..., description="'kullanici' veya 'asistan'")
    icerik: str


class ChatRequest(BaseModel):
    tur: str = Field(..., description="Hayvan türü")
    yas: Optional[str] = Field(None, description="Yaklaşık yaş")
    gecmis: list[ChatMessage] = Field(default_factory=list)
    yeni_mesaj: str = Field(..., min_length=1)
    fotograf_base64: Optional[str] = None


class ChatResponse(BaseModel):
    yanit: str
    aciliyet: Urgency
    aciliyet_sabit_kodlu_mu: bool
    uyari: str = (
        "Bu değerlendirme bir veteriner hekim muayenesinin yerini tutmaz. "
        "Kesin teşhis ve tedavi için mutlaka bir veteriner hekime başvurun."
    )


# ---------------------------------------------------------------------------
# OpenAI entegrasyonu (sohbet modu)
# ---------------------------------------------------------------------------

CHAT_SYSTEM_PROMPT = """Sen bir evcil hayvan sağlığı ön-değerlendirme asistanısın. Sahibiyle
doğal, sıcak ve sakin bir sohbet dilinde konuşuyorsun — form doldurtmuyorsun,
gerçek bir danışman gibi merak ettiğin şeyleri soruyorsun (ne zamandır böyle,
başka belirti var mı, daha önce böyle bir şey oldu mu gibi).

KURALLAR:
- Asla kesin teşhis koyma, her zaman "olabilir" gibi ihtimalli ifadeler kullan.
- Asla ilaç ismi, doz, veya insan ilacı önerisi verme.
- Yeterli bilgi yoksa netleştirici bir soru sor, hemen sonuca atlama — ama
  semptomlar açıkça ciddiyse gereksiz yere soru sorup vakit kaybettirme.
- Aciliyet seviyesini her mesajında şu 4 kategoriden değerlendir: ACİL,
  YAKINDA_VET, GÖZLEMLE, DÜŞÜK_ENDİŞE. Şüphedeysen bir üst seviyeyi seç.
- Konuşmanın bir noktasında mutlaka "bir veteriner hekime danışın" ifadesi geçsin.
- Yanıtını SADECE şu JSON formatında ver, başka hiçbir metin ekleme:
  {"yanit": "kullanıcıya gösterilecek doğal, sohbet tarzı yanıt/soru metni",
   "aciliyet": "ACİL|YAKINDA_VET|GÖZLEMLE|DÜŞÜK_ENDİŞE"}
"""


def generate_mock_chat_response(req: ChatRequest) -> dict:
    text_lower = req.yeni_mesaj.lower()
    turn_count = len(req.gecmis)

    if any(k in text_lower for k in ["kusuyor", "kusma", "ishal"]):
        return {
            "yanit": "[MOCK YANIT] Anlıyorum, bu endişe verici olabilir. Kusma ne zaman başladı, "
                     "ve son 24 saatte yediği bir şey değişti mi?",
            "aciliyet": "YAKINDA_VET",
        }

    if turn_count == 0:
        return {
            "yanit": "[MOCK YANIT] Merhaba, yardımcı olmak isterim. Az biraz daha anlatır mısın — "
                     "bu ne zaman başladı ve başka fark ettiğin bir şey var mı?",
            "aciliyet": "GÖZLEMLE",
        }

    return {
        "yanit": "[MOCK YANIT] Bu bir test yanıtıdır — gerçek OpenAI çağrısı henüz yapılmadı. "
                 "Verdiğin bilgilere göre durumu gözlemlemeye devam edip değişiklik olursa "
                 "bir veterinere danışmanı öneririm.",
        "aciliyet": random.choice(["GÖZLEMLE", "DÜŞÜK_ENDİŞE"]),
    }


def call_openai_chat(req: ChatRequest) -> dict:
    if os.environ.get("MOCK_MODE") == "1":
        return generate_mock_chat_response(req)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY tanımlı değil")

    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(status_code=500, detail="openai paketi kurulu değil")

    client = OpenAI(api_key=api_key)

    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    messages.append({
        "role": "system",
        "content": f"Hayvan türü: {req.tur}, Yaş: {req.yas or 'belirtilmedi'}",
    })

    for m in req.gecmis:
        messages.append({
            "role": "assistant" if m.rol == "asistan" else "user",
            "content": m.icerik,
        })

    new_user_content = [{"type": "text", "text": req.yeni_mesaj}]
    if req.fotograf_base64:
        new_user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{req.fotograf_base64}"},
        })
    messages.append({"role": "user", "content": new_user_content})

    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=400,
    )

    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    tum_kullanici_mesajlari = [m.icerik for m in req.gecmis if m.rol == "kullanici"]
    tum_kullanici_mesajlari.append(req.yeni_mesaj)

    matched_keyword = None
    for msg in tum_kullanici_mesajlari:
        matched_keyword = check_hardcoded_emergency(msg)
        if matched_keyword:
            break

    ai_result = call_openai_chat(req)
    ai_urgency = Urgency(ai_result.get("aciliyet", Urgency.GOZLEMLE))

    if matched_keyword:
        final_urgency = Urgency.ACIL
        sabit_kodlu = True
        yanit = ai_result.get("yanit", "") + (
            "\n\n⚠️ Belirttiğiniz semptomlar acil olabilecek işaretler içeriyor. "
            "LÜTFEN HEMEN bir acil veteriner kliniğine gidin."
        )
    else:
        final_urgency = ai_urgency
        sabit_kodlu = False
        yanit = ai_result.get("yanit", "")

    return ChatResponse(
        yanit=yanit,
        aciliyet=final_urgency,
        aciliyet_sabit_kodlu_mu=sabit_kodlu,
    )


@app.get("/")
def root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "ok", "mesaj": "Pet sağlık sohbet servisi çalışıyor."}