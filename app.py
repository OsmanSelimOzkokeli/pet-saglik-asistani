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

Çalıştırma:
    pip install fastapi uvicorn openai httpx --break-system-packages
    export OPENAI_API_KEY=...
    uvicorn app:app --reload --port 8000
"""

import base64
import json
import os
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import asyncio
from datetime import datetime, timedelta
import httpx

app = FastAPI(title="Pet Sağlık Asistanı")

# Yerel geliştirme için CORS'u açıyoruz (index.html dosyasını doğrudan
# tarayıcıda açsan bile /triage'a istek atabilesin diye).
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
    ACIL = "ACİL"                     # hemen acil veteriner
    YAKINDA_VET = "YAKINDA_VET"       # 24 saat içinde veteriner
    GOZLEMLE = "GÖZLEMLE"             # birkaç gün gözlem, kötüleşirse veteriner
    DUSUK_ENDISE = "DÜŞÜK_ENDİŞE"     # muhtemelen önemsiz


# ---------------------------------------------------------------------------
# Sabit kodlanmış acil durum katmanı (AI'dan bağımsız, override edilemez)
# ---------------------------------------------------------------------------
# Not: Bu liste kapsamlı değildir; sadece gösterge niteliğindedir. Gerçek
# kullanımda veteriner hekim danışmanlığıyla genişletilmeli/doğrulanmalıdır.

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
    """Metinde acil anahtar kelime var mı kontrol eder. Varsa eşleşen kelimeyi döner."""
    text_lower = symptom_text.lower()
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text_lower:
            return keyword
    return None


# ---------------------------------------------------------------------------
# Modeller
# ---------------------------------------------------------------------------

class TriageRequest(BaseModel):
    tur: str = Field(..., description="Hayvan türü, örn: 'köpek', 'kedi', 'tavşan', 'muhabbet kuşu'")
    yas: Optional[str] = Field(None, description="Yaklaşık yaş, örn: '2 yaşında', '6 aylık'")
    semptom_metni: str = Field(..., min_length=3, description="Kullanıcının yazdığı semptom açıklaması")
    fotograf_base64: Optional[str] = Field(None, description="İsteğe bağlı, base64 kodlu fotoğraf")


class TriageResponse(BaseModel):
    aciliyet: Urgency
    aciliyet_sabit_kodlu_mu: bool = Field(..., description="True ise bu seviye AI değil, sabit kural tarafından belirlendi")
    olasi_nedenler: list[str]
    genel_tavsiye: str
    veteriner_onerisi: str
    uyari: str = (
        "Bu değerlendirme bir veteriner hekim muayenesinin yerini tutmaz. "
        "Kesin teşhis ve tedavi için mutlaka bir veteriner hekime başvurun."
    )



class ChatMessage(BaseModel):
    rol: str = Field(..., description="'kullanici' veya 'asistan'")
    icerik: str


class ChatRequest(BaseModel):
    tur: str = Field(..., description="Hayvan türü")
    yas: Optional[str] = Field(None, description="Yaklaşık yaş")
    gecmis: list[ChatMessage] = Field(default_factory=list, description="Önceki mesajlar (bu yeni mesaj hariç)")
    yeni_mesaj: str = Field(..., min_length=1, description="Kullanıcının yeni mesajı")
    fotograf_base64: Optional[str] = None


class ChatResponse(BaseModel):
    yanit: str = Field(..., description="Asistanın konuşma dilinde yanıtı")
    aciliyet: Urgency
    aciliyet_sabit_kodlu_mu: bool
    uyari: str = (
        "Bu değerlendirme bir veteriner hekim muayenesinin yerini tutmaz. "
        "Kesin teşhis ve tedavi için mutlaka bir veteriner hekime başvurun."
    )


# ---------------------------------------------------------------------------
# OpenAI entegrasyonu
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Sen bir evcil hayvan sağlığı ön-değerlendirme asistanısın. Amacın kesin
teşhis koymak DEĞİL, sahibine olası nedenler, genel bakım önerileri ve ne kadar
aciliyetle bir veterinere gitmesi gerektiği konusunda rehberlik etmek.

KURALLAR:
- Asla kesin teşhis koyma ("kesinlikle X hastalığı" deme), her zaman "olabilir",
  "şu ihtimaller arasında" gibi ifadeler kullan.
- Asla ilaç ismi, doz, veya insan ilacı önerisi verme (ör. parasetamol gibi
  hayvanlar için TOKSİK olabilecek insan ilaçları asla önerme).
- Aciliyet seviyesini şu 4 kategoriden seç: ACİL, YAKINDA_VET, GÖZLEMLE, DÜŞÜK_ENDİŞE.
  Şüphedeysen her zaman bir üst aciliyet seviyesini seç (daha temkinli ol).
  Aciliyeti gerçekte olduğundan DÜŞÜK gösterme.
- Her yanıtta mutlaka "bir veteriner hekime danışın" ifadesi bulunsun.
- Yanıtını SADECE şu JSON formatında ver, başka hiçbir metin ekleme:
  {"aciliyet": "ACİL|YAKINDA_VET|GÖZLEMLE|DÜŞÜK_ENDİŞE", "olasi_nedenler": ["...", "..."],
   "genel_tavsiye": "...", "veteriner_onerisi": "..."}
"""

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


import random


def generate_mock_response(req: TriageRequest) -> dict:
    """
    Gerçek OpenAI çağrısı yapmadan, geliştirme/test amaçlı sahte bir yanıt üretir.
    MOCK_MODE=1 olduğunda kullanılır. Böylece ödeme yapmadan formu, akışı ve
    arayüzü uçtan uca test edebilirsin.
    """
    text_lower = req.semptom_metni.lower()

    if any(k in text_lower for k in ["kusuyor", "kusma", "ishal"]):
        return {
            "aciliyet": "YAKINDA_VET",
            "olasi_nedenler": [
                "Hafif mide-bağırsak rahatsızlığı (sahte/mock yanıt)",
                "Beslenme değişikliği veya yediği bir şeyin hassasiyet yaratması (mock)",
            ],
            "genel_tavsiye": "[MOCK YANIT] 12 saat kadar yemek verilmeyip su takip edilebilir, kusma devam ederse veterinere gidin.",
            "veteriner_onerisi": "[MOCK YANIT] 24 saat içinde bir veterinere danışmanız önerilir.",
        }

    return {
        "aciliyet": random.choice(["GÖZLEMLE", "DÜŞÜK_ENDİŞE"]),
        "olasi_nedenler": ["Bu bir MOCK (sahte) yanıttır — gerçek API henüz çağrılmadı."],
        "genel_tavsiye": "[MOCK YANIT] Bu, gerçek OpenAI çağrısı olmadan üretilen test verisidir.",
        "veteriner_onerisi": "[MOCK YANIT] Gerçek bir değerlendirme için lütfen OPENAI_API_KEY ekleyip MOCK_MODE'u kapatın.",
    }


def call_openai_triage(req: TriageRequest) -> dict:
    # --- MOCK MOD: gerçek API'ye para harcamadan test etmek için ---
    if os.environ.get("MOCK_MODE") == "1":
        return generate_mock_response(req)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY tanımlı değil")

    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(status_code=500, detail="openai paketi kurulu değil")

    client = OpenAI(api_key=api_key)

    user_content = [
        {
            "type": "text",
            "text": (
                f"Hayvan türü: {req.tur}\n"
                f"Yaş: {req.yas or 'belirtilmedi'}\n"
                f"Semptom: {req.semptom_metni}"
            ),
        }
    ]

    if req.fotograf_base64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{req.fotograf_base64}"},
        })

    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
    )

    return json.loads(response.choices[0].message.content)


def generate_mock_chat_response(req: ChatRequest) -> dict:
    """Sohbet modu için sahte yanıt - para harcamadan arayüzü test etmek için."""
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
# Endpoint
# ---------------------------------------------------------------------------

URGENCY_ORDER = {
    Urgency.DUSUK_ENDISE: 0,
    Urgency.GOZLEMLE: 1,
    Urgency.YAKINDA_VET: 2,
    Urgency.ACIL: 3,
}


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest):
    # 1. Sabit kodlanmış acil durum kontrolü (AI'dan ÖNCE ve bağımsız)
    matched_keyword = check_hardcoded_emergency(req.semptom_metni)

    # 2. AI değerlendirmesi
    ai_result = call_openai_triage(req)

    ai_urgency = Urgency(ai_result.get("aciliyet", Urgency.GOZLEMLE))

    if matched_keyword:
        final_urgency = Urgency.ACIL
        sabit_kodlu = True
    else:
        final_urgency = ai_urgency
        sabit_kodlu = False

    return TriageResponse(
        aciliyet=final_urgency,
        aciliyet_sabit_kodlu_mu=sabit_kodlu,
        olasi_nedenler=ai_result.get("olasi_nedenler", []),
        genel_tavsiye=ai_result.get("genel_tavsiye", ""),
        veteriner_onerisi=(
            "Belirttiğiniz semptomlar acil olabilecek işaretler içeriyor. "
            "LÜTFEN HEMEN bir acil veteriner kliniğine gidin."
            if sabit_kodlu
            else ai_result.get("veteriner_onerisi", "")
        ),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Acil durum kontrolünü SADECE yeni mesajla değil, konuşmanın tamamıyla
    # (kullanıcının önceki mesajları dahil) yapıyoruz - biri iki mesaj önce
    # "kan geliyor" dediyse bu bilgiyi unutmamalıyız.
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


# ---------------------------------------------------------------------------
# Aşama 3: Sahiplendirme destek sayfası (Supabase)
# ---------------------------------------------------------------------------

class SahiplendirmeIlan(BaseModel):
    hayvan_adi: str
    tur: str
    yas: Optional[str] = None
    cinsiyet: Optional[str] = None
    aciklama: str
    fotograf_url: Optional[str] = None
    konum: str
    iletisim: str


def supabase_headers() -> dict:
    key = os.environ.get("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFrdnhsbmxzZHptY296d2t2YmdmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODYzNjEyNzEsImV4cCI6MjEwMTkzNzI3MX0.CUF-vLEVwGmlZWH4iUzGSpUEfLlXc-z-J0lvTwp_56E")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def supabase_url() -> str:
    url = os.environ.get("SUPABASE_URL", "https://qkvxlnlsdzmcozwkvbgf.supabase.co")
    return url.rstrip("/")


@app.post("/sahiplendirme")
def sahiplendirme_ekle(ilan: SahiplendirmeIlan):
    import urllib.request
    import urllib.error

    endpoint = f"{supabase_url()}/rest/v1/sahiplendirme_ilanlari"
    body = json.dumps(ilan.model_dump()).encode()

    req = urllib.request.Request(endpoint, data=body, headers=supabase_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        raise HTTPException(status_code=502, detail=f"Supabase hatası: {detail}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Supabase bağlantı hatası: {e}")

    return {"basarili": True, "ilan": data[0] if data else None,
            "mesaj": "İlanınız alındı, onaylandıktan sonra listede görünecek."}


@app.get("/sahiplendirme")
def sahiplendirme_listele():
    import urllib.request
    import urllib.error
    import urllib.parse

    params = urllib.parse.urlencode({
        "durum": "eq.onaylı",
        "order": "created_at.desc",
        "select": "*",
    })
    endpoint = f"{supabase_url()}/rest/v1/sahiplendirme_ilanlari?{params}"

    req = urllib.request.Request(endpoint, headers=supabase_headers(), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        raise HTTPException(status_code=502, detail=f"Supabase hatası: {detail}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Supabase bağlantı hatası: {e}")

    return {"ilanlar": data}


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


@app.get("/sahiplendir")
def sahiplendir_sayfa():
    html_path = os.path.join(os.path.dirname(__file__), "sahiplendirme.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "hata", "mesaj": "sahiplendirme.html bulunamadı"}


@app.get("/veteriner-basvuru")
def veteriner_basvuru_sayfa():
    html_path = os.path.join(os.path.dirname(__file__), "veteriner-basvuru.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "hata", "mesaj": "veteriner-basvuru.html bulunamadı"}


@app.get("/veteriner-takvim")
def veteriner_takvim_sayfa():
    html_path = os.path.join(os.path.dirname(__file__), "veteriner-takvim.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "hata", "mesaj": "veteriner-takvim.html bulunamadı"}


@app.get("/randevu-al")
def randevu_al_sayfa():
    html_path = os.path.join(os.path.dirname(__file__), "randevu-al.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "hata", "mesaj": "randevu-al.html bulunamadı"}


@app.get("/randevularim")
def randevularim_sayfa():
    html_path = os.path.join(os.path.dirname(__file__), "randevularim.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "hata", "mesaj": "randevularim.html bulunamadı"}


@app.get("/admin")
def admin_sayfa():
    html_path = os.path.join(os.path.dirname(__file__), "admin.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "hata", "mesaj": "admin.html bulunamadı"}


@app.get("/giris")
def giris_sayfa():
    html_path = os.path.join(os.path.dirname(__file__), "auth.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "hata", "mesaj": "auth.html bulunamadı"}


@app.get("/hesabim")
def hesabim_sayfa():
    html_path = os.path.join(os.path.dirname(__file__), "hesabim.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "hata", "mesaj": "hesabim.html bulunamadı"}


@app.get("/sifre-sifirla")
def sifre_sifirla_sayfa():
    html_path = os.path.join(os.path.dirname(__file__), "sifre-sifirla.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "hata", "mesaj": "sifre-sifirla.html bulunamadı"}


@app.get("/forum")
def forum_sayfa():
    html_path = os.path.join(os.path.dirname(__file__), "forum.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "hata", "mesaj": "forum.html bulunamadı"}


# ---------------------------------------------------------------------------
# Aşama 5: Randevu hatırlatmaları + hesap dondurma/silme (arka plan görevleri)
# ---------------------------------------------------------------------------

SUPABASE_URL = "https://qkvxlnlsdzmcozwkvbgf.supabase.co"
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")


async def randevu_hatirlatmalarini_kontrol_et():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Hatırlatma kontrolü çalıştı.")
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/randevular",
            headers=headers,
            params={
                "durum": "eq.onaylandı",
                "hatirlatma_gonderildi": "eq.false",
                "select": "*",
            },
        )
        randevular = res.json()
        if not isinstance(randevular, list):
            print("Randevu sorgusu beklenmeyen sonuç döndürdü:", randevular)
            return

        simdi = datetime.now()

        for r in randevular:
            try:
                randevu_zamani = datetime.strptime(f"{r['tarih']} {r['saat']}", "%Y-%m-%d %H:%M")
            except Exception:
                continue

            fark_dakika = (randevu_zamani - simdi).total_seconds() / 60

            # Randevuya 50-70 dakika kala tek seferlik hatırlatma gönder
            if 50 <= fark_dakika <= 70:
                kullanici_res = await client.get(
                    f"{SUPABASE_URL}/rest/v1/kullanici_profilleri",
                    headers=headers,
                    params={"id": f"eq.{r['kullanici_id']}", "select": "kullanici_adi"},
                )
                vet_res = await client.get(
                    f"{SUPABASE_URL}/rest/v1/kullanici_profilleri",
                    headers=headers,
                    params={"id": f"eq.{r['veteriner_id']}", "select": "kullanici_adi"},
                )
                basvuru_res = await client.get(
                    f"{SUPABASE_URL}/rest/v1/veteriner_basvurulari",
                    headers=headers,
                    params={
                        "kullanici_id": f"eq.{r['veteriner_id']}",
                        "durum": "eq.onaylı",
                        "select": "klinik_adi",
                        "limit": "1",
                    },
                )

                kullanici_data = kullanici_res.json()
                vet_data = vet_res.json()
                basvuru_data = basvuru_res.json()

                kullanici_adi = kullanici_data[0]["kullanici_adi"] if kullanici_data else "Kullanıcı"
                vet_adi = vet_data[0]["kullanici_adi"] if vet_data else "Veteriner"
                klinik_adi = basvuru_data[0]["klinik_adi"] if basvuru_data else vet_adi

                await client.post(
                    f"{SUPABASE_URL}/rest/v1/bildirimler",
                    headers=headers,
                    json={
                        "kullanici_id": r["kullanici_id"],
                        "mesaj": f"1 saat sonra, saat {r['saat']}'de {klinik_adi} ile randevunuz var.",
                        "link": "/randevularim",
                    },
                )
                await client.post(
                    f"{SUPABASE_URL}/rest/v1/bildirimler",
                    headers=headers,
                    json={
                        "kullanici_id": r["veteriner_id"],
                        "mesaj": f"1 saat sonra, saat {r['saat']}'de {kullanici_adi} ile randevunuz var.",
                        "link": "/veteriner-takvim",
                    },
                )

                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/randevular",
                    headers=headers,
                    params={"id": f"eq.{r['id']}"},
                    json={"hatirlatma_gonderildi": True},
                )

                print(f"Hatırlatma gönderildi: randevu {r['id']}")


async def dondurulmus_hesaplari_kontrol_et():
    """1 yıldır dondurulmuş (hesap_kapatildi=true) hesapları kalıcı olarak siler."""
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }
    esik_tarih = (datetime.now() - timedelta(days=365)).isoformat()

    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/kullanici_profilleri",
            headers=headers,
            params={
                "hesap_kapatildi": "eq.true",
                "hesap_dondurulma_tarihi": f"lt.{esik_tarih}",
                "select": "id",
            },
        )
        profiller = res.json()
        if not isinstance(profiller, list):
            print("Dondurulmuş hesap sorgusu beklenmeyen sonuç döndürdü:", profiller)
            return

        for p in profiller:
            user_id = p["id"]
            del_res = await client.delete(
                f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                headers=headers,
            )
            if del_res.status_code in (200, 204):
                print(f"1 yıldır dondurulmuş hesap kalıcı olarak silindi: {user_id}")
            else:
                print(f"Dondurulmuş hesap silinemedi ({user_id}): {del_res.text}")


async def arka_plan_dongusu():
    if not SUPABASE_SERVICE_KEY:
        print("UYARI: SUPABASE_SERVICE_KEY tanımlı değil, arka plan görevleri (hatırlatma + hesap silme) devre dışı.")
        return
    while True:
        try:
            await randevu_hatirlatmalarini_kontrol_et()
        except Exception as e:
            print("Hatırlatma kontrolünde hata:", e)
        try:
            await dondurulmus_hesaplari_kontrol_et()
        except Exception as e:
            print("Dondurulmuş hesap kontrolünde hata:", e)
        await asyncio.sleep(300)  # 5 dakikada bir kontrol et


@app.on_event("startup")
async def baslangicta_arka_plan_gorevlerini_baslat():
    asyncio.create_task(arka_plan_dongusu())


@app.post("/api/hesap-sil")
async def hesap_sil(request: Request):
    """Kullanıcının kendi hesabını kalıcı olarak silmesi için.
    Frontend, oturumun access_token'ını gönderir; biz onunla kullanıcının
    kimliğini doğrulayıp SADECE o kullanıcıyı silebiliriz."""
    if not SUPABASE_SERVICE_KEY:
        return JSONResponse({"status": "hata", "mesaj": "Sunucu yapılandırması eksik (SUPABASE_SERVICE_KEY)."}, status_code=500)

    body = await request.json()
    access_token = body.get("access_token")
    if not access_token:
        return JSONResponse({"status": "hata", "mesaj": "access_token gerekli."}, status_code=400)

    async with httpx.AsyncClient(timeout=15) as client:
        # 1) Bu token gerçekten geçerli mi, kime ait?
        user_res = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {access_token}",
            },
        )
        if user_res.status_code != 200:
            return JSONResponse({"status": "hata", "mesaj": "Geçersiz veya süresi dolmuş oturum."}, status_code=401)

        user_id = user_res.json().get("id")
        if not user_id:
            return JSONResponse({"status": "hata", "mesaj": "Kullanıcı bulunamadı."}, status_code=400)

        # 2) Sadece doğrulanan bu kullanıcıyı sil (bağlı tüm veriler CASCADE ile otomatik silinir)
        delete_res = await client.delete(
            f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            },
        )
        if delete_res.status_code not in (200, 204):
            return JSONResponse({"status": "hata", "mesaj": f"Silme başarısız: {delete_res.text}"}, status_code=500)

    return {"status": "basarili"}