"""
Pet Sağlık Asistanı - Semptom & Aciliyet Triyaj + Sohbet Servisi
---------------------------------------------------------
Sahip, hayvanının semptomlarını (metin ve isteğe bağlı fotoğrafla) anlatır,
AI ile sohbet ederek aciliyet seviyesi ve genel yönlendirme alır.
Ayrıca Google Places API ile yakındaki veteriner/petshop önerisi sunar.

ÖNEMLİ TASARIM KARARI:
Kritik/acil semptomlar sabit kodlanmış bir anahtar kelime katmanıyla tespit
edilir ve AI'nın yorumundan BAĞIMSIZ olarak her zaman "ACİL" seviyesine
zorlanır.

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
# Sabit kodlanmış acil durum katmanı
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

    if req.fotograf_base64:
        return {
            "yanit": "[MOCK YANIT] Fotoğrafı aldım (gerçek OpenAI modunda görsel analiz edilecek). "
                     "Şimdilik bunu bir test verisi olarak değerlendiriyorum — anlattığın "
                     "semptomlarla birlikte durumu gözlemlemeni öneririm.",
            "aciliyet": "GÖZLEMLE",
        }

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
# Aşama 2: Veteriner + Petshop rehberi (Google Places API)
# ---------------------------------------------------------------------------

class NearbyRequest(BaseModel):
    konum: str = Field(..., description="Şehir/ilçe, örn: 'Afyonkarahisar' ya da 'Zafer, Afyonkarahisar'")
    tur: str = Field(..., description="'veteriner' veya 'petshop'")


class NearbyPlace(BaseModel):
    isim: str
    adres: str
    puan: Optional[float] = None
    puan_sayisi: Optional[int] = None
    harita_linki: str
    acik_mi: Optional[bool] = None


class NearbyResponse(BaseModel):
    sonuclar: list[NearbyPlace]
    kaynak: str = Field(..., description="'mock' veya 'google_places'")


MOCK_NEARBY_DATA = {
    "veteriner": [
        {"isim": "[MOCK] Afyon Veteriner Kliniği", "adres": "[MOCK] Zafer Mah. Örnek Cad. No:12", "puan": 4.6, "puan_sayisi": 87, "acik_mi": True},
        {"isim": "[MOCK] Merkez Hayvan Hastanesi", "adres": "[MOCK] İstasyon Mah. Sağlık Sk. No:5", "puan": 4.3, "puan_sayisi": 54, "acik_mi": True},
        {"isim": "[MOCK] 7/24 Acil Veteriner", "adres": "[MOCK] Cumhuriyet Cad. No:34", "puan": 4.8, "puan_sayisi": 121, "acik_mi": True},
    ],
    "petshop": [
        {"isim": "[MOCK] Patili Dostlar Petshop", "adres": "[MOCK] Kurtuluş Mah. Çiçek Sk. No:8", "puan": 4.5, "puan_sayisi": 63, "acik_mi": True},
        {"isim": "[MOCK] Afyon Pet Market", "adres": "[MOCK] Dumlupınar Cad. No:21", "puan": 4.1, "puan_sayisi": 39, "acik_mi": False},
    ],
}


def generate_mock_nearby(req: NearbyRequest) -> dict:
    tur_key = "veteriner" if "vet" in req.tur.lower() else "petshop"
    raw = MOCK_NEARBY_DATA.get(tur_key, [])
    sonuclar = [
        {
            **item,
            "harita_linki": f"https://www.google.com/maps/search/{item['isim'].replace(' ', '+')}+{req.konum.replace(' ', '+')}",
        }
        for item in raw
    ]
    return {"sonuclar": sonuclar, "kaynak": "mock"}


def call_google_places(req: NearbyRequest) -> dict:
    if os.environ.get("MOCK_MODE") == "1":
        return generate_mock_nearby(req)

    api_key = os.environ.get("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_PLACES_API_KEY tanımlı değil")

    import urllib.request
    import urllib.parse

    query = f"{req.tur} {req.konum}"
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json?" + urllib.parse.urlencode({
        "query": query,
        "key": api_key,
        "language": "tr",
    })

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Google Places API hatası: {e}")

    if data.get("status") not in ("OK", "ZERO_RESULTS"):
        raise HTTPException(status_code=502, detail=f"Google Places API durumu: {data.get('status')}")

    sonuclar = []
    for place in data.get("results", [])[:10]:
        place_id = place.get("place_id", "")
        sonuclar.append({
            "isim": place.get("name", ""),
            "adres": place.get("formatted_address", ""),
            "puan": place.get("rating"),
            "puan_sayisi": place.get("user_ratings_total"),
            "acik_mi": place.get("opening_hours", {}).get("open_now"),
            "harita_linki": f"https://www.google.com/maps/place/?q=place_id:{place_id}" if place_id else "",
        })

    return {"sonuclar": sonuclar, "kaynak": "google_places"}


# ---------------------------------------------------------------------------
# Endpoint'ler
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


@app.post("/nearby", response_model=NearbyResponse)
def nearby(req: NearbyRequest):
    result = call_google_places(req)
    return NearbyResponse(**result)


@app.get("/")
def root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "ok", "mesaj": "Pet sağlık triyaj servisi çalışıyor. /docs adresine bakın."}


@app.get("/rehber")
def rehber():
    html_path = os.path.join(os.path.dirname(__file__), "nearby.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "hata", "mesaj": "nearby.html bulunamadı"}