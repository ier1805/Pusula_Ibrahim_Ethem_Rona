# Pusula Data Science Intern Case Study — Tedavi Süresi (Regression)

**Ad Soyad:** İbrahim Ethem RONA  
**E‑posta:** ronaibrahimethem@gmail.com
**Depo Adı:** `Pusula_Ibrahim_Ethem_Rona` (öneri)  
**Proje Dosyası:** `case_study.ipynb`

---

## Kısa Özet
Bu çalışma; **2235 gözlem, 13 özellik** içeren fizik tedavi & rehabilitasyon veri kümesi üzerinde **Keşifsel Veri Analizi (EDA)**, **veri ön işleme** ve **sızıntısız modelleme iskeleti** kurmayı amaçlar. Hedef değişken **`TedaviSuresi`** (seans sayısı) olup regresyon problemi olarak ele alınmıştır.  
Sonuçta, **RidgeCV** tabanlı bir pipeline ile **RMSE ≈ 3.04**, **MAE ≈ 1.69**, **R² ≈ 0.312** elde edilmiştir. Aynı çapraz doğrulama içinde baz çizgi (ortalama tahmin) **Null RMSE ≈ 3.72**’dir.

> Özetle: Kurulan pipeline, baz tahmine göre anlamlı iyileşme sağlamaktadır.

---

## Veri Kümesi
- **HastaNo** (ID), **Yas**, **Cinsiyet**, **KanGrubu**, **Uyruk**, **KronikHastalik**, **Bolum**, **Alerji**, **Tanilar**, **TedaviAdi**, **TedaviSuresi**, **UygulamaYerleri**, **UygulamaSuresi**  
- Hedef: **`TedaviSuresi`** (seans sayısı, ≥ 0).

---

## Uygulanan İşlemler (Özet)
### 1) Veri Temizleme
- Metin alanlarında **strip + boşluk sadeleştirme**, boş stringleri **NaN** kabul etme.
- Çok-değerli alanlar (örn. `KronikHastalik`, `Alerji`, `Tanilar`) için normalize edilebilir yapı oluşturma (liste/dummy üretimi gereksinime göre).

### 2) Etiket Normalizasyonu
- **Cinsiyet**’te serbest girişleri eşleştirme (örn. `k, kadın, bayan → Kadın`, `e, erkek, bay → Erkek`, diğerleri → `Unknown`).

### 3) Kodlama
- **`Uyruk`** ve **`KanGrubu`**: **One-Hot Encoding** (0/1 dummy sütunlar).  
- **`Cinsiyet`**: **Label Encoding**

### 4) Ölçekleme (Tek Kez ve Kontrollü)
- **Binary (0/1) dummy sütunlar ölçeklenmemiştir** (Ridge’de ceza dengesini bozabilir).
- Ortalama ≈ 0 ve std ≈ 1 olan (hali hazırda **z‑skor** görünümlü) sütunlar tespit edilip **yeniden ölçeklenmemiştir**.
- Kalan sayısallar **StandardScaler** ile **sadece train verisinde fit** edilmiştir.

### 5) Sızıntısız Doğrulama
- **GroupKFold (n=5)** ile katlamalar **`HastaNo`** düzeyinde ayrılmış; aynı hastaya ait kayıtların hem train hem testte görünmesi engellenmiştir.
- Tüm ön‑işleme adımları **`ColumnTransformer` + `Pipeline`** içinde tanımlanmıştır (fit/transform sırası güvenli).

### 6) Modeller
- **RidgeCV** (otomatik `alpha` seçimi; log‑uzayda tarama).  
- Kıyas için **RandomForestRegressor** kısa denemesi yapılmıştır.  
- (Opsiyonel çalışma iskeleti hazırlandı): **PoissonRegressor**, **HistGradientBoosting (loss="poisson")**, **ElasticNetCV** ve **log1p(y) dönüşümlü Ridge** (Notebook içindeki alternatif hücreler).

---

## Değerlendirme Metrikleri
- Çapraz doğrulama (5‑fold, grup bazlı):
  - **RidgeCV** → **MAE ≈ 1.693**, **RMSE ≈ 3.039**, **R² ≈ 0.312**
  - **Null RMSE (mean baseline)** ≈ **3.724**
- Yorum: **R² > 0** ve **RMSE’nin baz çizgiye göre ~%18 düşmesi** modelin anlamlı öğrenme yaptığını gösterir.

> Not: CV’de **R²** katman bazında hesaplanıp ortalandığı için, RMSE’den türetilen basit R² ile küçük farklar olabilir (normaldir).

---

## Proje Yapısı (Öneri)
```
.
├─ case_study.ipynb       # Tüm akış (EDA + feature engineering + CV)
├─ README.md              # Bu dosya
└─ requirements.txt       # Ortam (opsiyonel)
```

---

## Nasıl Çalıştırılır?
1. Python 3.9+ ortamı kurun ve bağımlılıkları yükleyin:
   ```bash
   pip install -U pandas numpy scikit-learn matplotlib
   ```
2. Notebook’u açın ve sırasıyla çalıştırın:
   - **Veri okuma** hücresinde kendi veri dosyanızın yolunu belirtin.
   - Tüm hücreleri çalıştırın (ön‑işleme + CV değerlendirme).
3. Sonuç metrikleri ve karşılaştırmalar son hücrelerde raporlanır.

> Reprodüksiyon için `random_state=42` ve **GroupKFold** kullanılmıştır.

---

## Tasarım Kararları ve Gerekçeler
- **Pipeline içinde ön‑işleme:** Fit/transform sırasını garanti eder; sızıntıyı önler.
- **Binary’leri ölçeklememe:** Ceza dengesini korur; Ridge/ElasticNet’te daha stabil çözüm.
- **Cinsiyet’i One‑Hot:** Sahte sıralılık varsayımını önler; lineer model için daha doğru.
- **GroupKFold:** Aynı hastanın tekrarlı satırları arasında sızıntıyı engeller.

---

## Geliştirme Fikirleri
- **Hedef dönüşümü**: `log1p(y)` ile uzun kuyruk etkisini yumuşatma.
- **Poisson tabanlı kayıplar** (sayım doğasına uygun).
- **Özellik seçimi / önem sırası**: Permütasyon önem ile gürültülü kolonların elenmesi.
- **Hiperparametre araması**: HGB/ElasticNet için grid veya bayesçi arama.
- **Hata analizi**: Hataları `Bolum`, `Yas` kırılımında inceleyip yerel modeller.

---
