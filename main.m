%% --- 1. BAŞLANGIÇ VE MODEL KONTROLÜ ---
% Çalışma ortamını temizle:
clc;        % Komut penceresindeki (Command Window) yazıları temizler.
clear;      % Workspace'teki (Hafıza) tüm değişkenleri siler.
close all;  % Açık olan tüm grafik pencerelerini kapatır.

% Rastgelelik kontrolü:
rng('default'); % Kod her çalıştığında aynı rastgele sayıların üretilmesini sağlar (Tekrarlanabilirlik için).

% Dosya ve Kontrol Değişkenleri:
dosya_adi = 'kayitli_model.mat'; % Eğitilen modelin kaydedileceği veya okunacağı dosya adı.
egitim_yapilsin_mi = true;       % Başlangıçta eğitimin yapılacağını varsayıyoruz (True).

% Daha önce kaydedilmiş bir model dosyası var mı diye kontrol et:
if exist(dosya_adi, 'file')
    % Eğer dosya varsa kullanıcıya bilgi ver:
    fprintf('==========================================\n');
    fprintf('BULUNAN KAYIT: Daha önce eğitilmiş bir model bulundu.\n');
    
    % Kullanıcıya sor: Kayıtlı modeli mi kullanalım, yeniden mi eğitelim?
    choice = input('Kayıtlı model yüklensin mi? (E: Evet / H: Hayır, yeniden eğit): ', 's');
    
    % Kullanıcının cevabını kontrol et (Büyük/küçük harf duyarsız - strcmpi):
    if strcmpi(choice, 'e')
        fprintf('Model yükleniyor...\n');
        load(dosya_adi); % Dosyadaki 'model', 'sozluk', 'XTest' vb. değişkenleri hafızaya yükle.
        egitim_yapilsin_mi = false; % Kayıtlı model yüklendiği için tekrar eğitime gerek yok.
    else
        % Kullanıcı 'Hayır' dediyse veya başka bir tuşa bastıysa:
        fprintf('Model sıfırdan eğitilecek...\n');
    end
end

%% --- 2. EĞİTİM SÜRECİ (Sadece Gerekirse Çalışır) ---
% Eğer model yüklenmediyse (egitim_yapilsin_mi = true ise) bu blok çalışır:
if egitim_yapilsin_mi
    
    fprintf('\n--- VERİ HAZIRLAMA VE EĞİTİM BAŞLIYOR ---\n');
    
    % --- 2.1. Veri Yükleme ---
    data = readtable('Turkish-Review-Sentiment-Data.csv'); % CSV dosyasını tablo formatında oku.
    yorumlar = string(data.review);    % 'review' sütununu metin (string) dizisine çevir.
    etiketler = string(data.sentiment); % 'sentiment' sütununu etiket olarak al.

    % Sadece "olumlu" ve "olumsuz" etiketli verileri filtrele (Nötr vb. varsa atar):
    idx = ismember(etiketler, ["olumlu", "olumsuz"]);
    yorumlar = yorumlar(idx);              % İlgili satırların yorumlarını al.
    etiketler = categorical(etiketler(idx)); % Etiketleri kategorik veri tipine dönüştür (Makine öğrenmesi için gerekli).

    % --- 2.2. Temizleme ve Tokenization (Kelimelere Bölme) ---
    temiz_yorumlar = lower(yorumlar); % Tüm harfleri küçük harfe çevir.
    
    % Regex ile temizlik: Harfler, sayılar ve Türkçe karakterler hariç her şeyi (noktalama vb.) sil.
    temiz_yorumlar = regexprep(temiz_yorumlar, '[^\w\sçğıöşüÇĞİÖŞÜ]', ''); 

    % Değişkenleri hazırla:
    token_listesi = cell(length(temiz_yorumlar), 1); % Her cümlenin kelimelerini tutacak hücre dizisi.
    tum_kelimeler = []; % Sözlük oluşturmak için tüm kelimeleri biriktireceğimiz dizi.

    % Her bir yorumu tek tek işle:
    for i = 1:length(temiz_yorumlar)
        kelimeler = split(temiz_yorumlar(i));    % Cümleyi boşluklara göre böl.
        kelimeler = kelimeler(kelimeler ~= "");  % Boş stringleri (fazla boşluktan oluşan) temizle.
        token_listesi{i} = kelimeler;            % Temiz kelimeleri listeye ekle.
        tum_kelimeler = [tum_kelimeler; kelimeler]; % Genel havuza ekle.
    end

    % --- 2.3. Bag-of-Words (Kelime Çantası) Oluşturma ---
    sozluk = unique(tum_kelimeler); % Tüm kelimelerden benzersiz bir sözlük oluştur.
    
    num_docs = length(token_listesi); % Toplam doküman (yorum) sayısı.
    num_words = length(sozluk);       % Sözlükteki toplam kelime sayısı.
    X = zeros(num_docs, num_words);   % Özellik matrisini (Matrix X) sıfırlarla başlat.

    % Her dokümanı sayısal vektöre çevir:
    for i = 1:num_docs
        bu_doc_kelimeleri = token_listesi{i};           % i. yorumun kelimelerini al.
        [tf, loc] = ismember(bu_doc_kelimeleri, sozluk); % Bu kelimeler sözlükte kaçıncı sırada?
        loc = loc(tf); % Sadece sözlükte bulunanların indekslerini al.
        
        % Frekans sayımı (Hangi kelime kaç kere geçti):
        for j = 1:length(loc)
            X(i, loc(j)) = X(i, loc(j)) + 1; % İlgili kelimenin sayacını 1 artır.
        end
    end
    
    % Normalizasyon: Uzun cümlelerin avantajını dengelemek için her satırı kelime toplamına böl.
    % (+ 0.0001 sıfıra bölünme hatasını engellemek içindir)
    X = X ./ (sum(X, 2) + 0.0001); 
    Y = etiketler; % Hedef çıktı (etiketler).

    % --- 2.4. Bölümleme (%80 Train - %10 Val - %10 Test) ---
    num_samples = size(X, 1);       % Toplam örnek sayısı.
    p = randperm(num_samples);      % İndeksleri rastgele karıştır.
    
    nTrain = round(0.80 * num_samples); % %80 Eğitim verisi sayısı.
    nVal   = round(0.10 * num_samples); % %10 Doğrulama verisi sayısı.
    
    % İndeksleri parçala:
    idxTrain = p(1:nTrain);
    idxVal   = p(nTrain+1 : nTrain+nVal);
    idxTest  = p(nTrain+nVal+1 : end); % Geriye kalanlar Test verisi (%10).

    % Veri setlerini oluştur:
    XTrain = X(idxTrain, :);  YTrain = Y(idxTrain); % Eğitim seti
    XVal   = X(idxVal, :);    YVal   = Y(idxVal);   % Doğrulama seti
    XTest  = X(idxTest, :);   YTest  = Y(idxTest);  % Test seti

    % --- 2.5. Model Eğitimi ---
    fprintf('SVM Modeli eğitiliyor...\n');
    % fitcecoc: Çok sınıflı (Multi-class) Support Vector Machine (SVM) modeli eğitir.
    model = fitcecoc(XTrain, YTrain); 
    
    % Validation Skoru (Eğitim sırasında modelin durumunu görmek için):
    predVal = predict(model, XVal);               % Doğrulama verisiyle tahmin yap.
    accVal = sum(predVal == YVal) / numel(YVal);  % Doğru tahmin oranını hesapla.
    fprintf('Validation Başarısı: %% %.2f\n', accVal * 100);

    % --- 2.6. KAYDETME İŞLEMİ ---
    fprintf('Model ve veriler kaydediliyor (%s)...\n', dosya_adi);
    % Modeli, oluşturulan sözlüğü ve test verilerini daha sonra kullanmak üzere .mat dosyasına kaydet.
    save(dosya_adi, 'model', 'sozluk', 'XTest', 'YTest', 'num_words');
    fprintf('Kayıt Başarılı!\n');
    
else
    % Eğer model dosyalardan yüklendiyse burası çalışır:
    fprintf('\n--- EĞİTİM ATLANDI, MODEL HAZIR ---\n');
end

%% --- 3. PERFORMANS RAPORU (Her Durumda Çalışır) ---
% Model ya yüklendi ya da yeni eğitildi, şimdi final testi yapalım.

predTest = predict(model, XTest);              % Test verisi üzerinde tahmin yap.
accTest = sum(predTest == YTest) / numel(YTest); % Doğruluk oranını hesapla.

% Sonuçları ekrana yazdır:
fprintf('\n==================================\n');
fprintf('       FİNAL TEST SONUÇLARI       \n');
fprintf('==================================\n');
fprintf('Test Doğruluk       : %% %.2f\n', accTest * 100);
fprintf('Test Veri Sayısı    : %d\n', numel(YTest));
fprintf('==================================\n');

% Confusion Matrix (Karmaşıklık Matrisi) Çizimi:
% Hangi sınıfın hangi sınıfla karıştırıldığını görselleştirir.
figure('Name', 'Test Confusion Matrix', 'NumberTitle', 'off');
cmTest = confusionchart(YTest, predTest);
cmTest.Title = ['Test Başarısı: %' num2str(accTest*100, '%.1f')]; % Grafik başlığı.
cmTest.RowSummary = 'row-normalized';     % Satır sonlarına başarı oranlarını ekle.
cmTest.ColumnSummary = 'column-normalized'; % Sütun altlarına başarı oranlarını ekle.

%% --- 4. CANLI CÜMLE TESTİ ---
% Kullanıcıdan anlık veri alıp tahmin yapma bölümü.
% 'sozluk' değişkeni burada kritik öneme sahiptir çünkü yeni cümle, eğitilen sözlüğe göre kodlanmalıdır.

test_cumlesi = input('\nTest etmek için bir cümle girin (Örn: Ürün harika): ', 's');

% Eğer kullanıcı boş geçip Enter'a basarsa varsayılan bir cümle ata:
if isempty(test_cumlesi)
    test_cumlesi = "Bu ürün tam bir fiyat performans ürünü, çok beğendim.";
end

% --- Yeni Gelen Cümleyi Temizle ---
test_temiz = lower(test_cumlesi); % Küçük harfe çevir.
test_temiz = regexprep(test_temiz, '[^\w\sçğıöşüÇĞİÖŞÜ]', ''); % Noktalama temizle.
test_tokens = split(test_temiz); % Kelimelere böl.
test_tokens = test_tokens(test_tokens ~= ""); % Boşlukları at.

% --- Vektöre Çevir (Mevcut 'sozluk' yapısına göre) ---
X_yeni = zeros(1, length(sozluk)); % Tek satırlık, sözlük uzunluğunda boş bir vektör oluştur.

% Yeni cümledeki kelimelerin sözlükte olup olmadığına bak:
[tf_yeni, loc_yeni] = ismember(test_tokens, sozluk);
loc_yeni = loc_yeni(tf_yeni); % Sadece sözlükte var olan kelimelerin indekslerini al.

% Vektörü doldur:
for j = 1:length(loc_yeni)
    X_yeni(1, loc_yeni(j)) = X_yeni(1, loc_yeni(j)) + 1;
end
% Not: Tek cümle olduğu için normalizasyon yapmasak da SVM genelde çalışır, 
% ancak tutarlılık için istenirse eğitimdeki gibi normalize edilebilir.

% --- Tahmin ---
sonuc = predict(model, X_yeni); % Modeli kullanarak sınıfı tahmin et.

% Sonucu ekrana yazdır:
fprintf('\n----------------------------------\n');
fprintf('Cümle : %s\n', test_cumlesi);
fprintf('SONUÇ : %s\n', string(sonuc));
fprintf('----------------------------------\n');