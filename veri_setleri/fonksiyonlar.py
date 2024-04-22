from degiskenler import *
def yas_getir(age):
    if age < 18:
        durum = "çocuk"
    elif age < 65:
        durum = "yetişkin"
    else:
        durum = "yaşlı"

    return durum


def cinsiyet_getir(cinsiyet):
    if cinsiyet == "Male":
        return "Ben bir erkeğim."
    else:
        return "Ben bir kadınım."


def bmi_hesapla(bmi):
    if bmi < 18.5:
        durum = "Düşük"
    elif 18.5 <= bmi < 25:
        durum = "Normal"
    elif 25 <= bmi < 30:
        durum = "Fazla Kilolu"
    else:
        durum = "Obez"

    return durum


def hba1c_hesapla(hba1c_seviyesi):
    if hba1c_seviyesi < 5.7:
        durum = "Normal"
    elif 5.7 <= hba1c_seviyesi < 6.5:
        durum = "Prediyabet"
    else:
        durum = "Diyabet"

    return durum


def kan_sekeri_hesapla(kan_sekeri_seviyesi):
    if kan_sekeri_seviyesi < 70:
        durum = "Düşük"
    elif 70 <= kan_sekeri_seviyesi <= 99:
        durum = "Normal"
    elif 100 <= kan_sekeri_seviyesi <= 125:
        durum = "Yüksek Normal"
    else:
        durum = "Çok Yüksek"

    return durum


def sigara_kullanimi_cevir(sigara_kullanimi):
    if sigara_kullanimi in secenekler_sigara_kullanimi:
        return sigara_kullanimi
    else:
        return "No Info"
