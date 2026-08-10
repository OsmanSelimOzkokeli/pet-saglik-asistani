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
    pip install fastapi uvicorn openai --break-system-packages
    export OPENAI_API_KEY=...
    uvicorn app:app --reload --port 8000
"""

import base64
import os
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Pet Sağlık Asistanı")


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


def call_openai_triage(req: TriageRequest) -> dict:
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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
    )

    import json
    return json.loads(response.choices[0].message.content)


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


@app.get("/")
def root():
    return {"status": "ok", "mesaj": "Pet sağlık triyaj servisi çalışıyor. /docs adresine bakın."}