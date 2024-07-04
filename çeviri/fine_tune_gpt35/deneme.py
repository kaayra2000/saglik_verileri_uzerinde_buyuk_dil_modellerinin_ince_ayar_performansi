from openai import OpenAI

with open("../../api_key.txt", "r") as file:
    api_key = file.read().strip()

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)

completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0125:personal::9h9ruhq9",
    messages=[
        {
            "role": "system",
            "content": "Kullanıcının yazdığı metni temizle ve gereksiz bilgileri kaldır. Sadece gerekli olan ve anlamlı bilgileri bırak.",
        },
        {
            "role": "user",
            "content": "SELAMLAR: ANNEM69 YAŞINDA BOĞAZ YADA GUATIRLA İLGİLİ ŞİKAYETİ YOK DAHİLİYE DR.U TAHLİLİNDE GULUKOZ129 LDLKOLLETEROL 132ÇIKTI FT4 : 1.5 TSH : 1.35ÇIKTI TİROİD ULTRASONOGRAFİSİ:TRİOİD HERİKİ LOBU NORMAL BOYUTTA EN BÜYÜKLERİ SAĞDA VE SOLDA 6.4 MM.ÇAPLARINDA OLMAK ÜZRE HOMOJEN VE KİSTİK DEJENERASYONLU,MULTIPL-HİPOEKOİK NODÜLER İZLENMİŞTİR SONUCU ÇIKTI BU NE ANLAMA GELİYOR.DAHİLİYE DR.U BİOPSİ İSTEDİ ANNEM BİOPSİDEN ÇEKİNİYOR SİZCE GEREKLİMİ.HER 2 İNSANSAN BİRİNDE NODÜLER OLABİLİRMİŞ DOĞRUMU TŞKKRLER",
        },
    ],
)
print(completion.choices[0].message.content)
