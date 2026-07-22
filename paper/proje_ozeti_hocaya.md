# Proje Özeti — Hocaya Anlatım

## Bir Cümlede
Kaliforniya'nın elektrik şebekesinde güneş enerjisinin yarattığı "duck curve" problemini, makine öğrenmesi ile tahmin edip, nükleer santral + batarya depolama sisteminin en uygun çalışma planını matematiksel optimizasyonla çıkaran bir çalışma.

---

## Problem Ne?

Kaliforniya'da çok fazla güneş paneli var (~18 GW). Öğlen saatlerinde güneş o kadar çok elektrik üretiyor ki, şebekenin ihtiyacından fazlası oluyor. Ama akşam güneş batınca, bir anda devasa bir enerji açığı oluşuyor — insanlar eve geliyor, klima/ışık açıyor ama güneş yok. Bu günlük pattern'e "duck curve" (ördek eğrisi) deniyor çünkü grafiği ördek gibi görünüyor.

**Sorun:** Bu ani iniş-çıkışları kim karşılayacak? Nükleer santral sabit çalışır (açıp kapayamazsın), doğalgaz hızlı ama pahalı ve kirletici, batarya depolama (BESS) esnek ama kapasitesi sınırlı. Bunları en uygun (ucuz + güvenilir + temiz) şekilde koordine etmek lazım — ama yarın ne olacağını bilmiyorsun.

---

## Biz Ne Yaptık?

### 1. Gerçek Veri Topladık
- ABD Enerji Bakanlığı'ndan (EIA API) Kaliforniya'nın 2023 yılı saatlik verilerini çektik:
  - Elektrik talebi (demand)
  - Güneş, rüzgar, nükleer, doğalgaz, hidroelektrik üretimi (yakıt bazlı)
  - Toptancı elektrik fiyatları (SP15 hub)
  - Henry Hub doğalgaz fiyatları
- NOAA'dan saatlik sıcaklık verisi (Los Angeles, San Francisco, Fresno)
- Toplam: **~115,000 kayıt**, tamamen gerçek ve açık kaynak

### 2. "Net Yük" Hesapladık
- **Net yük = Toplam talep − Güneş − Rüzgar**
- Bu, geleneksel santrallerin (nükleer, gaz) karşılaması gereken yük
- Sonuç: −10,369 MW (güneş talebi aşıyor!) ile +44,900 MW arası → 35 GW'lık devasa salınım

### 3. Makine Öğrenmesi ile Tahmin Ettik
- **Hedef:** Yarın her saatin net yükünü bugünden tahmin et (day-ahead, 24 saat öncesi)
- **35 feature:** Takvim, sıcaklık, dünkü talep, geçen haftanın aynı saati, ISO tahmin raporu, vb.
- **Kural:** Sadece tahmin anında bilinebilecek bilgileri kullandık (data leakage yok)
- **6 model karşılaştırdık:**

| Model | Test R² | Sonuç |
|-------|---------|-------|
| XGBoost | 0.854 | **En iyi** |
| LightGBM | 0.843 | İyi |
| Random Forest | 0.834 | İyi |
| LSTM (derin öğrenme) | 0.676 | Kötü |
| MLP (sinir ağı) | 0.481 | Çok kötü |
| Persistence (dünkü değer) | 0.736 | Baseline |

- XGBoost persistence'dan **%18.5 daha iyi** (istatistiksel olarak anlamlı, p < 0.001)
- 2022 verisinde de test ettik (model 2023'te eğitildi, 2022'ye uygulandı): **R² = 0.908** → genellenebilir

### 4. Tahmin Aralıkları Oluşturduk (CQR)
- Sadece "yarın 20,000 MW olacak" demek yetmez — "18,000–22,000 MW arası olacak, %90 eminim" demek lazım
- **Conformalized Quantile Regression (CQR)** kullandık — dağılım varsayımı gerektirmeyen, matematiksel garantili bir yöntem
- Sonuç: %90 hedefle %88.8 kapsama → çalışıyor

### 5. Matematiksel Optimizasyon Yaptık (MILP)
- **MILP = Mixed-Integer Linear Programming** (karışık tamsayılı doğrusal programlama)
- Soru: "168 saat (1 hafta) boyunca nükleer, gaz, batarya, ithalat/ihracatı nasıl çalıştırırsam toplam maliyet minimum olur?"
- **Pyomo + HiGHS solver** ile çözdük (açık kaynak)
- 6 senaryo + ilkbahar/sonbahar karşılaştırması + 22 sensitivity konfigürasyonu

### 5b. MILP Modeli Detaylı Açıklama

#### Ne yapıyor bu model?
Bir hafta (168 saat) boyunca, her saat için "hangi santral ne kadar üretsin, batarya şarj mı olsun boşalma mı yapsın, ne kadar ithalat yapılsın" sorularını **toplam maliyeti minimize ederek** cevaplıyor. Bu bir **ekonomik dağıtım (economic dispatch)** problemi.

**Neden 168 saat?** 24 saatlik horizon, bataryanın Pazar düşük-talebi şarj edip Pazartesi akşam tepesini karşılaması gibi günler-arası arbitraj fırsatlarını ıskalıyor. 168 saat = 1 tam haftalık döngü, CAISO'nun hafta içi/sonu talebi ve BESS davranışının periyodikliğini tam olarak kapsıyor. Yıllık modeller ise duck curve saatlik dinamiğini kaybeder.

#### Karar Değişkenleri (Decision Variables) — "Ne kontrol ediyoruz?"

Her saat t = 1, 2, ..., 168 için:

| Değişken | Birim | Açıklama | Sınırlar |
|----------|-------|----------|----------|
| P_nuc(t) | MW | Nükleer santral çıkışı | 1,800 – 2,256 MW |
| P_ccgt(t) | MW | CCGT (çevrim-çevrimi gaz) çıkışı | 0 – 10,000 MW |
| P_peak(t) | MW | Peaker (basit-çevrim) gaz çıkışı | 0 – 10,000 MW |
| P_ch(t) | MW | Batarya şarj gücü | 0 – 5,000 MW |
| P_dis(t) | MW | Batarya deşarj gücü | 0 – 5,000 MW |
| SoC(t) | MWh | Batarya doluluk seviyesi | 2,000 – 18,000 MWh |
| P_imp(t) | MW | Komşu bölgelerden ithalat | 0 – 10,000 MW |
| P_exp(t) | MW | Komşu bölgelere ihracat | 0 – 6,000 MW |
| P_shed(t) | MW | Yük kesintisi (istenmeyen) | 0 – 5,000 MW |
| P_curt(t) | MW | Yenilenebilir enerji kesintisi | 0 – 20,000 MW |
| u(t) | {0,1} | Batarya modu (1=boşalma) | Binary (0 veya 1) |

**Toplam:** 168 saat × 11 değişken = **1,848 değişken** (168 tanesi binary → MILP yapan kısım bu)

**İki-katmanlı gaz modeli + CCGT Unit Commitment neden?**
Hakem beklentisi: tek bir gaz bloğu çok basit, startup maliyeti ve min up/down time yok. Biz bunu çözdük:

- **CCGT**: 2 × 5,000 MW aggregate unit. Her birinin commitment binary'si (u_ccgt), startup binary'si (y_ccgt), startup maliyeti ($25,000/event), minimum 4 saat online, minimum 2 saat offline kısıtı var. Bu basitleştirilmiş ama gerçekçi bir **Unit Commitment (UC)** modeli.
- **Peaker**: 10,000 MW tek blok, kısıtsız (basit-çevrim GTs).

Bu yaklaşım, nükleer kaldırıldığında hem CCGT'nin maximum kapasiteye çalışmasını hem de pahalı peaker'ların devreye girmesini doğru şekilde gösteriyor. UC olmadan bu ikinci etki kaybolurdu.

#### Amaç Fonksiyonu (Objective Function) — "Neyi minimize ediyoruz?"

```
min Σ_{t=1}^{168} [ 12·P_nuc(t) + 45·P_gas(t) + 5·(P_ch(t) + P_dis(t))
                    + 55·P_imp(t) − 20·P_exp(t)
                    + 10000·P_shed(t) + 10·P_curt(t) ]
```

Her terimin anlamı:

| Terim | Maliyet | Açıklama |
|-------|---------|----------|
| 12 · P_nuc | $12/MWh | Nükleer yakıt + bakım (çok ucuz) |
| 45 · P_gas | $45/MWh | Doğalgaz yakıt + bakım + karbon |
| 5 · (P_ch + P_dis) | $5/MWh | Batarya yıpranma (degradation) maliyeti |
| 55 · P_imp | $55/MWh | Komşu bölgelerden elektrik alma |
| −20 · P_exp | $20/MWh | İhracat geliri (negatif maliyet = kazanç) |
| 10000 · P_shed | $10,000/MWh | Yük kesintisi cezası (VOLL — çok yüksek!) |
| 10 · P_curt | $10/MWh | Yenilenebilir enerji israfının fırsat maliyeti |

**Mantık:** Nükleer en ucuz ($12), CCGT orta ($40), Peaker pahalı ($65), ithalat daha pahalı ($55). Yük kesintisi astronomik pahalı ($10,000) — model bunu son çare olarak kullanır.

#### Kısıtlar (Constraints) — "Kurallar neler?"

**C1 — Güç dengesi (her saatte arz = talep):**
```
P_nuc(t) + P_gas(t) + P_dis(t) + P_imp(t) + P_shed(t)
  = Net_Load(t) + P_ch(t) + P_exp(t) + P_curt(t)
```
Sol taraf: arz (üreten + ithal + kesinti). Sağ taraf: talep (net yük + şarj + ihracat + curtailment).
**Her saat bu denklem sağlanmak zorunda** — fizik kuralı, elektrik depolanamaz.

**C2 — Batarya dinamiği (SoC güncelleme):**
```
SoC(t) = SoC(t-1) + √0.90 · P_ch(t) − (1/√0.90) · P_dis(t)
```
- √0.90 = 0.949 → şarj ederken %5.1 kayıp
- 1/√0.90 = 1.054 → boşaltırken %5.4 kayıp
- Round-trip efficiency = %90 (100 MWh koyarsan 90 MWh alırsın)
- Başlangıç: SoC(0) = 10,000 MWh (yarı dolu)

**C3 — Batarya aynı anda şarj ve deşarj olamaz:**
```
P_ch(t) ≤ 5000 × (1 − u(t))     → u=1 ise şarj = 0
P_dis(t) ≤ 5000 × u(t)          → u=0 ise deşarj = 0
```
Bu kısıt binary değişken u(t) ile sağlanıyor → **MILP yapan kısım bu** (LP olsaydı binary olmazdı).

**C4 — Nükleer rampa limiti:**
```
|P_nuc(t) − P_nuc(t-1)| ≤ 100 MW/saat
```
Nükleer santral aniden açılıp kapanamaz. Saatte max 100 MW değişebilir (PWR reaktör fiziği: nominal kapasitesinin %4.4/saat, NEA 2011). Toplam 2,256 MW üzerinden bu ~%4.4/h'e karşılık gelir.

**C5 — CCGT rampa limiti:**
```
|P_ccgt(t) − P_ccgt(t-1)| ≤ 800 MW/saat
```
CCGT'ler buhar türbini termal ataletinden dolayı yavaş rampa sahip.

**C6 — Peaker rampa limiti:**
```
|P_peak(t) − P_peak(t-1)| ≤ 5,000 MW/saat
```
Basit-çevrim gazlar çok esnek — saatte 5,000 MW değişebilir, pratikte kısıtsız.

**C7 — Dönem-sonu SoC kısıtı (yeni ekleme):**
```
SoC(168) ≥ 0.35 × 20,000 = 7,000 MWh
```
Bu olmadan model, son saatlerde bataryayı boşaltarak "işin bitti zaten" mantığıyla sahte düşük maliyet gösterir. Bu kısıt o sonu-horizon aldatmacasını önler (rolling-horizon MILP'in bilinen zafiyeti).

**C8 — Batarya doluluk sınırları:**
```
2,000 ≤ SoC(t) ≤ 18,000 MWh    (yani %10 – %90)
```
Batarya hiçbir zaman tamamen boşaltılmaz veya doldurulmaz (ömrü korumak için).

**C9 — Kapasite sınırları:**
Tüm değişkenler yukarıdaki tablodaki min-max arasında.

#### Sistem Parametreleri — "Nereden geliyor bu rakamlar?"

| Parametre | Değer | Kaynak |
|-----------|-------|--------|
| Nükleer kapasite | 2,256 MW | Diablo Canyon (2 × 1,128 MW PWR) |
| Nükleer minimum | 1,800 MW | ~%80 stabil çıkış alt limiti |
| CCGT kapasitesi | 10,000 MW | CAISO çevrim-çevrimi gaz filosu |
| Peaker kapasitesi | 10,000 MW | CAISO basit-çevrim + reciprocating |
| BESS güç | 5,000 MW | CAISO 2023 kurulu BESS kapasitesi |
| BESS enerji | 20,000 MWh | 4-saatlik batarya (5 GW × 4h) |
| BESS verimlilik | %90 | Li-ion tipik round-trip |
| İthalat limiti | 10,000 MW | CAISO hat kapasitesi |

#### Senaryolar — "Ne test ettik?"

| Senaryo | $/MWh | Açıklama | CCGT | Peaker | Shed |
|---------|-------|----------|------|--------|------|
| S1 Deterministik | 41.85 | ML point tahmin, nükleer var | 9,758 MW | 144 MW | 0 |
| S2 Worst-case | 43.59 | CQR Q90 üst sınır | 10,000 MW | 1,276 MW | 0 |
| S3 Best-case | 40.31 | CQR Q10 alt sınır | 9,278 MW | 0 MW | 0 |
| S4 No Nuclear | 47.36 | Nükleer YOK (+13.2%) | 9,991 MW | 1,153 MW | 0 |
| S5 Small BESS | 42.25 | BESS 1 GW / 4 GWh | 9,206 MW | 556 MW | 0 |
| S6 Robust | 42.64 | %50 point + %50 Q90 | 9,992 MW | 641 MW | 0 |

S4'te peaker kullanımının artması (144→1,153 MW) nükleer olmadan esneklik maliyetinin görünür hale gelmesini sağlıyor — bu iki-katmanlı gazın katkısı.

Ayrıca:
- **İlkbahar haftası** (Nisan 10-16): duck curve en derin, net yük negatife düşüyor
- **Kasım haftası**: standart sonbahar talebi
- **Robust γ sweep**: γ = 0 (deterministik) → 1 (worst-case) arası 5 nokta
- **Sensitivity:** Nükleer 0-4 GW, BESS 0.5-10 GW, Gaz fiyatı $25-100/MWh

#### Çözücü (Solver)

- **Pyomo** — Python'da optimizasyon modeli yazma framework'ü
- **HiGHS** — Açık kaynak MILP çözücü (Edinburgh Üniversitesi)
- Çözüm süresi: **< 0.3 saniye** (çok hızlı)
- Optimality gap: **< %0.01** (global optimuma ulaşıldı)
- Model boyutu: ~2,184 değişken, **840 binary** (168 BESS + 2×168 CCGT commitment + 2×168 CCGT startup), ~2,100 kısıt

#### "Linear" mı "Integer" mı?

- **Linear kısım:** Tüm maliyet ve kısıtlar doğrusal (P_nuc × 12 gibi). Karesel veya üssel terim yok.
- **Integer kısım:** Batarya modu u(t) ∈ {0,1} binary değişken. Bu olmasaydı saf LP olurdu.
- **Mixed-Integer:** Hem sürekli (P_nuc, P_gas...) hem binary (u) değişkenler var → MILP
- LP relaxation + branch-and-bound ile çözülüyor (HiGHS bunu otomatik yapar)

### 6. Temel Bulgular

**Nükleerin değeri:**
- Nükleer çıkarılırsa maliyet **%13.2 artar** ($41.85 → $47.36/MWh, iki-katmanlı gaz modeli)
- İlkbaharda nükleer olmadan: %14.7 maliyet artışı, load shedding yok
- Kasım'da nükleer olmadan: **11,349 MWh yük kesintisi** (haftalık talebin %3.7'si!)
- Yılda **6.8 milyon ton CO₂** tasarruf sağlıyor

**Batarya (BESS):**
- 5 GW'dan sonra ek batarya çok az fayda sağlıyor (azalan getiri)
- Fiyata duyarlı çalıştırılırsa **%3.74 maliyet düşüşü** + haftalık $4.2M gelir

**Doğalgaz fiyatı:**
- $55/MWh üstünde gaz kullanımı yarıya düşüyor → karbon fiyatlandırması için kritik eşik

**Robustness (sağlamlık) analizi:**
- Daha güvenli dispatch = daha pahalı + daha çok CO₂ → "trilemma" (üçlü ikilem)

---

## Hangi Araçları Kullandık?

| Araç | Ne İçin |
|------|---------|
| **Python** | Tüm pipeline |
| **EIA API** | Enerji verisi (demand, generation, prices) |
| **NOAA ISD** | Sıcaklık verisi |
| **LightGBM, XGBoost** | Makine öğrenmesi (gradient boosting) |
| **PyTorch** | LSTM derin öğrenme |
| **scikit-learn** | Random Forest, MLP, metrikler |
| **Pyomo + HiGHS** | Matematiksel optimizasyon (MILP) |
| **Matplotlib** | Figürler |
| **python-docx** | Makale Word dosyası |

---

## Makale Ne Durumda?

- **~12,000 kelime**, 11 tablo, 17 figür, 33 referans
- Applied Energy / EJOR formatında
- Hedef dergi: **Electric Power Systems Research** (Elsevier, IF ~3.8)
- Cover letter hazır
- Tüm veriler açık kaynak (EIA, NOAA) → tekrarlanabilir

---

## Makalenin Güçlü Yanları (Hocaya Söylenecek)

1. **Tamamen gerçek veri** — sentetik yok, EIA/NOAA'dan 2022+2023
2. **Metodolojik titizlik** — data leakage önlendi, 6 model benchmark, DM testi, 5-fold CV, 2022 OOS validation
3. **CQR** — nispeten yeni bir yöntem, enerji alanında az kullanılmış
4. **Policy-relevant** — nükleer değeri, karbon etkisi, batarya boyutlandırma
5. **Orijinal bulgu** — robustness-cost-emissions trilemma

---

## Hocaya Sorulacak

1. "Bu çalışmayı beraber yayınlamak ister misiniz?"
2. "Corresponding author olarak siz olursanız review sürecinde desteğiniz çok değerli olur"
3. "Tüm kod ve veri pipeline'ı hazır, istediğiniz değişikliği yapabiliriz"
4. "Electric Power Systems Research hedefliyorum, siz ne düşünürsünüz?"
