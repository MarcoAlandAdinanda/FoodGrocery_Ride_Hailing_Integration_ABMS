# Rumusan Penelitian

## Latar Belakang dan Tujuan

Penelitian ini bertujuan mengevaluasi apakah penambahan layanan grocery ke dalam operasi *food delivery* yang sudah berjalan dapat meningkatkan produktivitas operasional, serta mengidentifikasi trade-off yang ditimbulkannya terhadap emisi karbon dan tingkat layanan (*service level*).

Integrasi yang dimaksud adalah pemenuhan kebutuhan food dan grocery melalui satu rangkaian layanan terintegrasi. Kondisi pembanding atau baseline penelitian adalah sistem *food-only*, yaitu kondisi awal sebelum layanan grocery ditambahkan. Penelitian ini tidak membandingkan sistem terintegrasi dengan dua layanan food dan grocery yang dioperasikan secara terpisah.

## Klasifikasi KPI Berdasarkan Subjek

KPI diklasifikasikan berdasarkan subjek yang kinerja, pengalaman, sumber daya, atau dampaknya diukur, bukan berdasarkan nomor *research question*. Klasifikasi kanonis penelitian adalah:

| Subjek | Ruang lingkup KPI |
| --- | --- |
| **Customer** | Penyelesaian, pembatalan, order belum selesai, dan waktu layanan untuk seluruh permintaan food maupun khusus order integrated. |
| **Driver** | Produktivitas, utilisasi, waktu tunggu pickup, perjalanan, pendapatan, biaya operasi, dan profit. |
| **Merchant** | Komponen yang diterima/diproses/diserahkan, pembatalan saat persiapan, disposal/return, nilai produk terdampak, dan revenue Restaurant/Store. |
| **Platform** | Revenue residual Platform dan turunannya per order. |
| **Environment** | Emisi total, intensitas emisi, emisi per order/output, dan efek emisi marginal. |
| **System** | Output lintas-subjek seperti total completed service units dan perubahan output layanan. |

Satu KPI memiliki tepat satu subjek kanonis. *Research question* menggunakan satu atau beberapa KPI dari klasifikasi tersebut sebagai lensa analisis. Dengan demikian, RQ tetap menentukan kontrast, interpretasi, dan pengujian statistik, tetapi tidak menjadi kategori pemilik KPI. Sebagai contoh, Sub-RQ kinerja operasional menggunakan KPI Driver dan System, sedangkan Sub-RQ service level menggunakan KPI Customer serta KPI tunggu pickup milik Driver.

Waktu tunggu pickup diklasifikasikan sebagai KPI Driver karena observasinya menghitung aktivasi Driver yang menunggu handover di Merchant. Dampaknya terhadap Customer tetap dapat digunakan dalam analisis service level tanpa mengubah subjek KPI. KPI Merchant dan status komponen terutama menjadi indikator diagnostik untuk menjelaskan perubahan KPI Customer dan Driver.

## Research Question Utama

> **RQ Utama:** Apakah penambahan layanan grocery ke dalam operasi *food delivery* melalui mekanisme integrasi dapat meningkatkan produktivitas operasional dibandingkan baseline *food-only*, dan bagaimana trade-off peningkatan tersebut terhadap emisi karbon dan *service level*?

## Sub-Research Questions

### Sub-RQ 1 — Kinerja Operasional

> **Sub-RQ 1:** Seberapa besar penambahan layanan grocery melalui mekanisme integrasi meningkatkan produktivitas operasional sistem dibandingkan baseline *food-only*?

Aspek yang dapat digunakan untuk menilai kinerja operasional meliputi:

- utilisasi Driver;
- produktivitas Driver;
- jumlah order atau komponen layanan yang diselesaikan;
- waktu penyelesaian order;
- waktu tunggu pada proses pickup;
- jarak tempuh per unit layanan yang diselesaikan;
- biaya operasional per unit layanan; dan
- profit Driver per hari atau per unit layanan.

KPI produktivitas utama yang disepakati adalah jumlah unit layanan selesai per Driver-jam:

\[
P = \frac{N_F + N_G}{M \times T}
\]

dengan \(N_F\) sebagai jumlah layanan food selesai, \(N_G\) sebagai jumlah layanan grocery selesai, \(M\) sebagai jumlah Driver, dan \(T\) sebagai durasi operasi dalam jam. Satu food dan satu grocery masing-masing dihitung sebagai satu unit layanan yang setara tanpa pembobotan. Dalam model, satu order food-only yang berstatus `COMPLETED` menghasilkan satu unit, sedangkan satu order integrated yang berstatus `COMPLETED` menghasilkan dua unit. Status `HANDED_OVER` tidak dihitung sebagai layanan selesai.

Perubahan produktivitas dilaporkan sebagai selisih absolut dan relatif terhadap baseline food-only:

\[
\Delta P = P_{integrated} - P_{food-only}
\]

\[
\Delta P_{\%} = \frac{P_{integrated} - P_{food-only}}{P_{food-only}} \times 100\%
\]

Utilisasi Driver tidak digunakan sebagai satu-satunya indikator karena utilisasi yang lebih tinggi juga dapat disebabkan oleh bertambahnya beban perjalanan atau waktu tunggu. KPI pendukung mencakup unit layanan selesai per busy Driver-hour dan per kilometer.

### Sub-RQ 2 — Emisi Karbon

> **Sub-RQ 2:** Bagaimana perubahan total emisi dan intensitas emisi ketika layanan grocery ditambahkan ke operasi *food delivery* melalui mekanisme integrasi dibandingkan baseline *food-only*?

Pengukuran emisi perlu menggunakan denominator yang memungkinkan perbandingan secara adil. Kandidat ukuran meliputi:

- total emisi sistem;
- emisi per order yang diselesaikan;
- emisi per komponen food atau grocery yang diselesaikan;
- emisi per kebutuhan food–grocery yang dipenuhi; dan
- tambahan emisi untuk setiap tambahan komponen grocery yang berhasil dilayani.

KPI emisi utama yang disepakati adalah emission intensity:

\[
EI = \frac{E}{N_F + N_G}
\]

dengan \(E\) sebagai total unit emisi, sedangkan \(N_F+N_G\) adalah completed service units. Perubahan absolut dan relatif terhadap baseline adalah:

\[
\Delta EI = EI_{integrated} - EI_{food-only}
\]

\[
\Delta EI_{\%} = \frac{EI_{integrated}-EI_{food-only}}{EI_{food-only}}\times100\%
\]

KPI tambahan pertama adalah total emission \(E\), termasuk perubahan absolut \(\Delta E=E_{integrated}-E_{food-only}\) dan persentasenya. KPI tambahan kedua adalah marginal emission:

\[
ME_G = \frac{E_{integrated}-E_{food-only}}{N_{G,integrated}}
\]

untuk tambahan emisi per completed grocery, serta:

\[
ME_S = \frac{E_{integrated}-E_{food-only}}{(N_{F,integrated}+N_{G,integrated})-N_{F,food-only}}
\]

untuk tambahan emisi per tambahan output layanan bersih. \(ME_S\) hanya diinterpretasikan ketika tambahan output layanan bersih bernilai positif. Persentase perubahan output juga dibandingkan dengan persentase perubahan emisi; emission intensity membaik ketika output meningkat lebih cepat daripada emisi.

Karena baseline hanya melayani food, total emisi pada kondisi integrated dapat meningkat akibat bertambahnya output layanan. Oleh sebab itu, klaim lingkungan membedakan perubahan total emisi dari perubahan intensitas emisi. Penurunan intensitas emisi menunjukkan pemanfaatan operasi yang lebih efisien, tetapi tidak selalu berarti total emisi absolut menurun. Interpretasi trade-off tersebut dibahas setelah hasil eksperimen tersedia dan tidak dijadikan threshold keberhasilan yang ditetapkan sebelumnya.

Emisi perjalanan dihitung menggunakan asumsi skenario motor rendah karbon dengan faktor emisi 60 g CO2/km. Nilai ini adalah asumsi penelitian, bukan hasil pengukuran kendaraan dalam penelitian. Dengan kecepatan representatif 30 km/jam, satu menit perjalanan setara dengan 0,5 km. Karena satu tick adalah satu menit dan paling banyak menghasilkan satu perpindahan grid, satu perpindahan grid merepresentasikan 0,5 km. Oleh karena itu:

\[
E_{grid}=60\ \text{g CO}_2/\text{km}\times0{,}5\ \text{km/grid}=30\ \text{g CO}_2/\text{grid}.
\]

Tick tanpa perpindahan tidak menambah emisi.

Probabilitas integrasi referensi \(p=0{,}44\) berasal dari nilai rata-rata data survei primer. Nilai 0,22 dan 0,66 digunakan sebagai level sensitivitas di sekitar kondisi referensi dan bukan sebagai estimasi tambahan dari populasi.

### Sub-RQ 3 — Service Level

> **Sub-RQ 3:** Apakah integrasi menghasilkan perbedaan yang signifikan secara statistik pada *service level* layanan food dibandingkan baseline *food-only*, sekaligus tetap memberikan gambaran pengalaman layanan pelanggan integrated?

Aspek *service level* yang dapat dievaluasi meliputi:

- completion rate;
- cancellation rate;
- unfinished-order rate pada akhir horizon simulasi;
- rata-rata waktu penyelesaian dan waktu tunggu pickup pada order yang memiliki observasi lengkap.

Evaluasi Sub-RQ 3 menggunakan pengujian statistik berpasangan. Setiap hasil skenario integrated dipasangkan dengan baseline *food-only* dari *seed*, permintaan, posisi awal, dan urutan aktivasi Driver yang sama. Untuk setiap KPI dihitung selisih (d_r=Y_{integrated,r}-Y_{food-only,r}), lalu diuji hipotesis dua arah (H_0): pusat selisih sama dengan nol, pada tingkat signifikansi α = 0,05. Uji Shapiro–Wilk diterapkan pada selisih antarpasangan. Jika normalitas tidak ditolak digunakan *paired t-test*; jika ditolak digunakan *Wilcoxon signed-rank test*. Untuk keluarga KPI utama dalam satu kondisi digunakan koreksi Holm. Selain *p-value* mentah dan terkoreksi, hasil melaporkan selisih rata-rata, interval bootstrap 95%, dan *Cohen's* (d_z) jika terdefinisi. Dengan demikian, pengujian ini menilai signifikansi statistik, bukan margin *non-inferiority* atau threshold kelayakan praktis.

Desain eksperimen integrated menggunakan faktorial penuh 3 × 3 × 3: probabilitas integrasi 0,22, 0,44, dan 0,66 disilangkan dengan jumlah Grocery Store 10, 15, dan 20 serta kebijakan pemilihan Store `uniform`, `nearest_restaurant`, dan `nearest_customer`. Kedua puluh tujuh sel integrated dijalankan bersama satu kontrol *food-only* pada setiap replikasi. Dengan 30 replikasi, desain menghasilkan 840 model runs. Kontrol tidak disilangkan dengan jumlah atau kebijakan Store karena pada probabilitas integrasi nol tidak terdapat komponen grocery. Struktur ini memungkinkan analisis efek probabilitas, ketersediaan Store, kebijakan Store, dan interaksinya, sementara perbandingan utama tetap kondisi 0,44–15 Store–uniform terhadap baseline.

Service level dinilai melalui dua dimensi yang disepakati. Dimensi pertama adalah **food service protection** untuk seluruh permintaan food, dengan KPI utama food completion rate dan mean food completion time. KPI pendukungnya adalah food cancellation rate, food unfinished rate, mean food pickup waiting time, serta mean food post-pickup delivery time. Food post-pickup delivery time dihitung dari handover food sampai order tiba di Customer.

Pada tabel eksperimen, KPI dimensi ini selalu menggunakan nama eksplisit berawalan `food_`. Alias generik seperti `completion_rate`, `cancellation_rate`, `active_rate`, dan `mean_completion_ticks` tidak digunakan agar populasi pengukuran tetap jelas dan definisinya tidak bergantung pada asumsi bahwa seluruh order selalu mengandung food.

Dimensi kedua adalah **integrated customer experience**, dengan KPI utama integrated-order completion rate dan mean integrated completion time. KPI pendukungnya adalah integrated cancellation rate, integrated unfinished rate, serta mean pickup waiting time untuk komponen food dan grocery pada order integrated.

On-time completion rate tidak digunakan karena penelitian tidak memiliki data empiris atau SLA yang dapat digunakan untuk menetapkan target waktu yang defensibel. Seluruh metrik waktu dilaporkan sebagai mean aritmetika per run agar setiap replikasi menghasilkan satu nilai yang dapat digunakan dalam analisis berpasangan. Metrik waktu harus dibaca bersama completion, cancellation, dan unfinished rate karena mean hanya menggunakan order dengan observasi waktu yang lengkap.

## Kriteria Umum Keberhasilan Integrasi

Secara konseptual, integrasi dinilai memberikan manfaat apabila menghasilkan trade-off yang dapat diterima berdasarkan ketiga kondisi berikut:

1. produktivitas operasional meningkat karena sistem dapat memenuhi tambahan kebutuhan grocery;
2. pertambahan total emisi terkendali dan/atau intensitas emisi per unit layanan menurun; dan
3. arah, besar, dan signifikansi statistik perubahan *service level* layanan food dilaporkan bersama pengalaman pelanggan integrated.

Ketiga dimensi perlu dinilai secara bersama-sama. Peningkatan satu KPI saja belum cukup untuk menyimpulkan bahwa integrasi memberikan perbaikan sistem secara keseluruhan.

## Batasan dan Pembahasan Lanjutan

Desain eksperimen dan hipotesis statistik tidak mensyaratkan threshold besar efek yang ditentukan peneliti. Keputusan apakah besar efek cukup bermakna untuk implementasi merupakan ranah decision maker setelah membaca hasil penelitian. Pembahasan hasil akan tetap menjelaskan besar dan arah efek, ketidakpastian statistik, serta trade-off antar-KPI tanpa menetapkan keputusan implementasi.

Parameter baseline final mencakup persiapan grocery Normal(31, 19) menit, nilai grocery Normal(Rp42.491, Rp29.634), 15 grocery store, 300 order per hari, 12 Driver, 50 Restaurant, timeout assignment 15 menit, timeout handover 30 menit sejak assignment, dan pemilihan store uniform sebagai kondisi referensi. Timeout handover berlaku selama masih ada komponen order yang belum `HANDED_OVER`. Durasi persiapan food dan grocery dipertahankan pada nilai referensi dan tidak menjadi faktor sensitivitas. Parameter tersebut perlu dibedakan dalam pelaporan antara yang berasal dari data dan yang merupakan asumsi skenario.

Interpretasi trade-off emisi akan disusun secara terpisah pada bagian hasil dan pembahasan setelah keluaran eksperimen tersedia.
