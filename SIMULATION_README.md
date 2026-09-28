# Simulasi baseline Food-Grocery

Engine: `src/simulation_engine.py`. Visualizer: `src/simulation_visualizer.py`.
Kedua script lama tidak diubah dan tidak diimpor oleh engine baru.

Sumber mekanisme satu-satunya adalah
`source/Integrated_Food_Grocery_Delivery_Model_Documentation.pdf`.
Pilihan yang masih terbuka dalam PDF diisi dengan keputusan pengguna pada rancangan
yang disetujui; seluruh pilihan tersebut juga tersimpan dalam metadata setiap run.

## Menjalankan

Gunakan environment conda `abms` yang sudah tersedia, dari direktori proyek:

```powershell
conda activate abms
python -B src/simulation_engine.py --p-integration 0.5
python -B src/simulation_visualizer.py --p-integration 0.5
```

`0.5` di atas hanya contoh input eksperimen. Secara bawaan `P_INTEGRATION = 0.44`,
yang berasal dari nilai rata-rata data survei primer,
di `src/simulation_parameters.py`; argumen CLI dapat menggantinya. Jika nilainya
diubah menjadi `None`, `--p-integration` menjadi wajib. Gunakan `0` untuk food-only atau `1` untuk seluruh order
integrated. Setiap order tetap mengandung food, dan tidak ada demand grocery
terpisah atau penggabungan dua order independen.

## Mengubah parameter terpusat

Ubah nilai di `src/simulation_parameters.py` sebelum menjalankan simulasi.
File tersebut memuat ukuran populasi dan grid, waktu, distribusi durasi dan
nilai produk, tarif, pajak, biaya, emisi, serta default desain eksperimen.
`P_INTEGRATION = None` mewajibkan input probabilitas melalui CLI;
nilai `0.44` saat ini menjadi default CLI.
Controller eksperimen memerlukan nilai utama lebih besar dari nol karena
membandingkannya dengan baseline `p=0`. Pola permintaan per jam tetap dibaca
dari `DEFAULT_DEMAND_CSV`, sedangkan total order harian diatur oleh
`DAILY_CUSTOMERS` atau `--daily-customers`. Restart proses Python setelah
mengubah file. Metadata run menyimpan nilai parameter dan hash file parameter.

### Paired experiments and integration sensitivity

The controller `src/simulation_experiments.py` compares each integrated condition
with `p=0` in the same seeded day. It uses a full-factorial design while keeping
daily food demand (300) and driver count (12) fixed:

```powershell
python -B src/simulation_experiments.py --p-integrated 0.44 --replications 30 --sensitivity --summary-only
```

`--sensitivity` uses these exploratory levels:

| Factor | Grid option | Default levels |
| --- | --- | --- |
| Integration probability | `--p-values` | `0.22,0.44,0.66` |
| Grocery Store count | `--store-count-values` | `10,15,20` |
| Grocery store selection | `--store-selection-values` | `uniform,nearest_restaurant,nearest_customer` |

Every positive probability level is crossed with every Store count and selection rule.
The reference `p=0.44` and uniform store selection are always included. `p=0`
is a control, not an integration level. Nearest store uses Manhattan distance;
ties choose the first store. Food and grocery preparation multipliers remain at
their reference value `1.0`; they are not sensitivity factors. The design has
27 integrated factorial cells and 1 control per replication: 840 model runs for 30
replications. Without `--sensitivity`, only reference integration and its
control run.

Store positions are nested: the 10-Store condition uses the first 10 positions,
15 uses the first 15, and 20 uses all 20 positions from the same seeded scenario.
Uniform selection uses one shared uniform draw per order, mapped to the active
Store count, so comparisons retain common random numbers.

`--demand-csv` supplies normalized hourly shares; multiplying every CSV weight
does not change the daily total. Use the count arguments for total demand.
All conditions within a replication share food arrivals, order attributes,
initial positions and driver activation permutations. Conditions with the same
seed are dependent day-block observations, not additional independent
replications. Potential grocery is instantiated only for integrated orders,
following Store → Restaurant → Customer.

Outputs under `output/experiments` include one scenario per replication;
`run_summaries.csv`, `paired_differences.csv`, `paired_statistics.csv`,
`emission_statistics.csv`, `service_level_statistics.csv`,
`sensitivity_effects.csv`, `manifest.json`, and `experiment.log`. The log records
the experiment configuration, seed and scenario progress, the start and final
KPI summary of every condition, failures with a traceback, completion of every
replication, and final output location. Detailed per-tick and per-agent audit
data remain in the CSV exports rather than being duplicated into the text log.
Each contrast subtracts the common `p=0` control
at the same replication. Pointwise 95% bootstrap
intervals resample whole day pairs separately for each condition. `paired_statistics.csv`
also performs a two-sided paired significance test at `alpha=0.05` on each vector
of replication-level differences. Shapiro–Wilk selects a paired t-test when
normality is not rejected and Wilcoxon signed-rank otherwise. Raw p-values are
reported for all eligible metrics; Holm-adjusted p-values and decisions cover
the primary-metric family within each condition. At least three observed pairs
are required for a significance test. The table also reports paired Cohen's
`d_z` when its standard deviation is nonzero. Columns include
`condition_id`, `factor`, all four factor levels, and `baseline_condition_id`.
`sensitivity_effects.csv` subtracts the reference condition's paired delta from
each alternative condition's paired delta within the same replication. This is
the direct sensitivity effect and is written when sensitivity alternatives exist.
Additional metrics include cancellation rate, unfinished-order rate, driver
utilization and daily profit per driver. The primary productivity metric is
`service_units_per_driver_hour`: one completed food component plus one completed
grocery component, divided by Drivers and elapsed hours. Food and grocery are
equal, unweighted service units. Read conditional time means with
completion rate and observation counts.

`--summary-only` skips the seven detailed engine export files per run.
`--visualize-replication 1` replays the selected day at the reference cell;
`--visualize-condition baseline` selects food-only (default: integrated).
Reset retains that scenario and its configured population. Playback adds no
statistical observation.

Keduanya menerima `--seed` (default 42), `--demand-csv`, dan `--output-dir`.
Lokasi CSV default ditentukan relatif terhadap proyek, bukan working directory.
Kolom yang digunakan adalah `hour` dan `avg_demand`; bobot dirata-ratakan per jam
melintasi seluruh baris hari, kemudian dinormalisasi. CSV default berisi tujuh hari.

The engine and visualizer accept `--daily-customers` (default 300),
`--num-drivers` (default 12), and `--num-stores` (default 15), all positive integers. Grid size 50x50, 50
Restaurants, 15 Stores, a 15-minute assignment timeout, a 30-minute handover
timeout and a 1440-tick horizon remain fixed.
Use the experiment controller for batch runs.

## Jarak dan gerak empat arah

Baseline 13 September 2026 menggunakan Manhattan, yaitu `abs(dx) + abs(dy)`,
untuk matching ke merchant pertama (Restaurant untuk food-only, Store untuk
integrated). Jarak sama diputuskan dengan waktu pembuatan lalu ID order.
Driver bergerak satu sel horizontal atau vertikal; diagonal tidak diperbolehkan.
Di antara empat tetangga dan sel saat ini yang berada di dalam grid, driver
memilih skor Euclidean kuadrat terkecil terhadap target, lalu tuple `(x, y)`.
Skor ini menentukan arah saja, bukan kilometer yang diakumulasikan.

Jarak tarif adalah 0,5 kali jumlah Manhattan Restaurant–Customer dan, untuk
integrated, Manhattan Store–Restaurant. Pickup pertama dari posisi driver tidak
ditagihkan. Quote dibuat saat order lahir, dengan ongkir
`max(9000, 2250 * billable_distance_km)` tanpa pembulatan kilometer. Nilai
Rp9.000 dan Rp2.250/km adalah titik tengah rentang Zona I KP 667/2022 yang
dipakai sebagai proxy generik jasa sepeda motor berbasis aplikasi, bukan tarif
resmi GrabFood atau platform tertentu. Seluruh ongkir menjadi revenue bruto
Driver karena data publik payout delivery spesifik platform tidak tersedia;
biaya seluruh gerak aktual tetap mengurangi profit Driver.
Sebaliknya, jarak aktual mencakup seluruh perpindahan yang sudah terjadi,
termasuk pickup pertama dan perjalanan order yang belum selesai.

Setiap perpindahan bernilai 0,5 km, biaya Rp200, dan emisi 30 g CO2.
Perjalanan `(0,0)` ke `(3,4)` membutuhkan 7 langkah: 3,5 km, Rp1.400, dan
210 g CO2. Menunggu tidak menambah jarak/biaya perjalanan/emisi.
Jumlah langkah bukan total waktu pelayanan karena ada aktivasi claim,
pickup, transisi state, dan acknowledgement.

Ini merupakan abstraksi jalan ortogonal tanpa hambatan, bukan jaringan jalan
nyata. Mode Chebyshev lama tidak tersedia; helper Python `manhattan()` menggantikan
`chebyshev()` tanpa alias. Garis visualizer tetap penghubung antartarget, bukan
jejak langkah. Hasil baseline baru dapat berbeda dari hasil diagonal lama.

Visualizer juga menerima `--interval-ms` (default 50), `--hide-driver-ids`, dan
`--hide-routes`. Matplotlib memerlukan backend GUI untuk jendela interaktif.
Pause/Resume mengontrol playback; Step menjalankan satu tick lalu berhenti;
Speed hanya mengubah kecepatan animasi. Reset mengekspor run sebelumnya terlebih
dahulu lalu membuat model baru dengan konfigurasi dan seed identik. Reset ditolak
jika isi CSV telah berubah, agar hasil pengulangan tetap dapat direproduksi.

## Hasil dan arti metrik

KPI dikategorikan berdasarkan subjek yang diukur, bukan berdasarkan research
question: `customer`, `driver`, `merchant`, `platform`, `environment`, dan
`system`. Research question memilih KPI lintas-subjek untuk suatu analisis,
tetapi tidak mengubah kepemilikan KPI. Contohnya, completion dan cancellation
adalah KPI Customer; produktivitas, utilisasi, pickup wait, revenue dan profit
adalah KPI Driver; revenue serta pembatalan persiapan adalah KPI Merchant;
revenue residual adalah KPI Platform; emisi adalah KPI Environment; dan
perubahan total output layanan adalah KPI System. Pickup wait dimiliki Driver
karena observasinya menghitung aktivasi Driver yang menunggu handover.

Controller menyimpan taksonomi kanonis dalam `manifest.json` pada
`kpi_subjects`. Tabel statistik berformat panjang (`paired_statistics.csv`,
`emission_statistics.csv`, `service_level_statistics.csv`, dan
`sensitivity_effects.csv`) membawa kolom `subject`. Label primary/secondary dan
metode paired/condition-only merupakan atribut analisis yang terpisah dari
subjek KPI.

Setiap run menghasilkan direktori baru di `output/simulations`. `--output-dir`
dapat menunjuk direktori baru atau kosong; keluaran yang sudah ada tidak ditimpa.
Untuk Reset dengan direktori eksplisit, run berikutnya memakai direktori sejajar
dengan akhiran `_reset_1`, `_reset_2`, dan seterusnya.

| File | Isi |
|---|---|
| `kpi_ticks.csv` | KPI akhir setiap tick dan urutan aktivasi Driver |
| `orders.csv` | Semua order, timestamp, quote pembayaran, status dan settlement |
| `components.csv` | Komponen food/grocery, persiapan, handover, dan event pembatalan |
| `drivers.csv` | Posisi/status akhir, pekerjaan, perjalanan, utilisasi, revenue dan profit |
| `merchants.csv` | Posisi/status akhir, komponen, event pembatalan, KPI pembatalan PREPARING dan revenue merchant |
| `summary.json` | Rekap akhir/parsial termasuk agregat KPI pembatalan restaurant/store |
| `metadata.json` | Konfigurasi, seed, versi library, hash input/PDF, kebijakan dan jadwal demand |

Satu hari menghasilkan 1440 record KPI, dengan label tick 0 sampai 1439, serta
300 record order. Customer yang sudah `COMPLETED` atau `CANCELLED` dikeluarkan
dari grid, tetapi riwayat order dan komponen tetap tersimpan. Merchant permanen
tidak dihapus setelah pickup.

Produktivitas utama adalah `completed_service_units / (num_drivers *
elapsed_ticks / 60)`. Satu order food-only `COMPLETED` menghasilkan satu unit
food; satu order integrated `COMPLETED` menghasilkan satu unit food dan satu
unit grocery. Komponen `HANDED_OVER` belum dihitung selesai. KPI pendukung
`service_units_per_busy_driver_hour` dan `service_units_per_km` memakai definisi
unit layanan yang sama.

KPI emisi utama adalah `emission_intensity_per_service_unit`, yaitu total unit
emisi dibagi `completed_service_units`. Total emisi tetap tersedia sebagai
`emission_units`. Untuk setiap paired comparison, controller menghitung tambahan
emisi per completed grocery, tambahan emisi per tambahan output layanan bersih,
persentase perubahan output dan emisi, serta emission-output elasticity. Rasio
per tambahan output hanya diisi ketika output bersih meningkat. Seluruh angka
menggunakan gram CO2 berdasarkan asumsi skenario motor rendah karbon dengan
faktor emisi 60 g CO2/km; nilai ini bukan hasil pengukuran kendaraan penelitian.

Service level memakai dua dimensi. Food service protection mencakup seluruh
permintaan food dan menggunakan food completion rate serta mean completion time
sebagai KPI utama. Integrated customer experience menggunakan integrated
completion rate serta mean integrated completion time. Cancellation, unfinished,
mean pickup wait, dan mean food post-pickup delivery menjadi KPI pendukung. Controller
merangkum pengalaman integrated lintas replikasi dalam
`service_level_statistics.csv`. On-time completion rate tidak digunakan karena
tidak ada target waktu empiris atau SLA yang ditetapkan.

Pada keluaran eksperimen, seluruh KPI food memakai nama kanonis berawalan
`food_`: `food_completion_rate`, `food_cancellation_rate`,
`food_unfinished_rate`, `mean_food_completion_ticks`,
`mean_food_pickup_wait_ticks`, dan `mean_food_post_pickup_delivery_ticks`.
Nama generik lama `completion_rate`,
`cancellation_rate`, `active_rate`, dan `mean_completion_ticks` tidak lagi
ditulis ke tabel eksperimen. Perubahan ini menghilangkan alias yang sebelumnya
bernilai sama hanya karena semua order dalam model mengandung food.

- Assignment waiting: `assigned - created`.
- Delivery service: `delivered - created`.
- Acknowledgement delay: `completed - delivered`.
- Completion total: `completed - created`.
- Cancellation waiting: `cancelled - created`.
- Pickup waiting: jumlah aktivasi pickup yang belum menerima handover. Rata-rata
  menggunakan pickup yang sudah selesai; kolom `*_so_far` juga menyertakan
  penantian order yang masih berlangsung.
- Utilisasi: tick sibuk dibagi tick yang telah dijalankan. Tick berhasil claim,
  perjalanan, pickup, dan delivery termasuk sibuk.

Nilai waktu yang belum teramati adalah sel CSV kosong / JSON `null`, bukan nol.
Setiap rata-rata waktu disertai jumlah observasi. `delivered_orders` berarti
status DELIVERED saat ini, sedangkan `delivered_cumulative` mencakup semua event
delivery, termasuk order yang kemudian COMPLETED.

Rekonsiliasi status: generated = available + assigned + delivered + completed +
cancelled. Active = available + assigned + delivered. Revenue hanya diposting
sekali saat COMPLETED; `customer_payment` adalah quote, bukan bukti settlement.
`settled_customer_payment` hanya menjumlahkan order yang sudah COMPLETED.

Perhitungan uang menggunakan Decimal tanpa pembulatan antara. Uang dalam JSON
disimpan sebagai string desimal agar presisi terjaga; CSV menyimpan teks desimal
langsung. Dashboard membulatkan tampilan saja. Emisi dinyatakan dalam gram CO2
berdasarkan faktor 60 g CO2/km, yaitu 30 g CO2 per perpindahan aktual sejauh 0,5 km.

### KPI pembatalan saat persiapan

Setiap restaurant/store memiliki dua KPI kumulatif, tersedia pada snapshot
merchant dan `merchants.csv`:

- `preparing_cancelled_units`: jumlah komponen yang dibatalkan saat masih
  `PREPARING`. Satu komponen food/grocery dihitung sebagai satu unit.
- `preparing_cancelled_product_value`: jumlah nilai penuh `item_value` dari
  komponen tersebut, dalam rupiah sebelum pajak, ongkir, dan komisi. Produk
  bernilai Rp0 tetap menambah satu unit.

KPI dihitung dari event `PREPARATION_STOPPED` yang tersimpan pada komponen,
sehingga pembacaan atau ekspor berulang tidak menggandakan hitungan.
`components.csv` menyimpan merchant ID, nilai produk, dan event sebagai jejak
pencatatan. Pada order integrated, status food/grocery dievaluasi independen;
hanya merchant dengan komponen yang masih PREPARING yang dihitung. Pembatalan
READY tetap tercatat sebagai food disposal atau grocery return.

`kpi_ticks.csv` dan `summary.json` menyediakan agregat terpisah dengan awalan
`restaurant_` dan `store_` untuk kedua nama KPI tersebut. Tanpa pembatalan
PREPARING, jumlah unit dan nilai produk adalah nol. Snapshot/ekspor parsial
menggunakan seluruh event yang sudah terjadi sampai waktu tersebut.

Pembatalan customer mendahului pembaruan merchant: jika pembatalan bertepatan
dengan tick persiapan selesai dan komponen masih PREPARING, komponen dihitung.
Rata-rata persiapan baseline adalah sekitar 12,90 menit untuk food dan 31 menit
untuk grocery. Timeout assignment adalah 15 menit sejak order dibuat, sedangkan
timeout handover adalah 30 menit sejak Driver ditugaskan. KPI ini dapat bernilai
nol walaupun ada pembatalan READY.

Ini adalah pencatatan nilai produk yang dibatalkan sebagai KPI agen; tidak ada
mekanisme pembayar, pembayaran kompensasi, atau refund. KPI tidak mengubah
revenue/profit. Rumus pajak dan revenue tetap mengikuti asumsi model PDF.

Metadata standalone menggunakan `schema_version: 8`; run dengan shared experiment
scenario menggunakan skema 9. Kenaikan versi ini mengganti satu field
`timeout_ticks` dengan `assignment_timeout_ticks` dan
`handover_timeout_ticks`. Manifest controller menggunakan
`experiment_schema_version: 13`. Versi 13 mengganti seluruh metrik waktu P90
dengan mean aritmetika atas observasi order yang tersedia. Versi 12 menambahkan jumlah Store ke desain
faktorial penuh 3 x 3 x 3 dan menggunakan posisi Store bertingkat; versi 11
mengubah probabilitas integrasi dan kebijakan Store menjadi faktorial; versi 10 menambahkan taksonomi KPI berdasarkan
subjek pada manifest dan tabel statistik; versi 9 menghapus multiplier persiapan
dari grid sensitivitas; versi 8 menambahkan uji signifikansi paired,
p-value mentah, koreksi Holm, keputusan pada alpha 0,05, dan Cohen's `d_z`.
Versi 7 menstandarkan nama KPI food pada tabel eksperimen dan menghapus alias
service-level generik. Perubahan skema sebelumnya
mencatat KPI produktivitas, emisi, service level dua dimensi, serta parameter tarif
`delivery_minimum_fee`, `delivery_fee_per_km`, dan
`driver_delivery_fee_share`. Field compensation lama pada order,
komponen, merchant, dan rekap diganti dengan KPI serta kebijakan
`cancellation_accounting`; status UNRESOLVED untuk compensation tidak dipakai
lagi. Hasil simulasi lama tetap tersimpan dengan skema sebelumnya dan tidak
ditimpa. Konsumen ekspor perlu membaca versi skema dalam `metadata.json`.

Versi skema menyatakan struktur data, bukan baseline jarak. Baseline Manhattan
tetap menggunakan kolom tabel yang sama; periksa `policies.matching`,
`policies.billable_distance`, `policies.movement`, serta hash PDF untuk
membedakan model pada ekspor lama dan baru. Ekspor lama tidak dimigrasikan.

## Batas waktu dan reproduksibilitas

Urutan tick: generate, Customer, Store, Restaurant, Driver, resolve, KPI.
Driver diacak setiap tick oleh RNG model. Customer membatalkan order `AVAILABLE`
tepat pada usia 15 menit sebelum matching pada tick tersebut. Setelah Driver
ditugaskan, Customer membatalkan order jika pada usia assignment 30 menit masih
ada komponen yang belum `HANDED_OVER`. Pemeriksaan deadline terjadi sebelum
aktivasi Merchant dan Driver, sehingga deadline menang atas matching atau
handover yang baru mungkin terjadi pada tick batas. Setelah seluruh komponen
`HANDED_OVER`, timeout kedua tidak lagi berlaku. Persiapan siap pada
`created + duration`, termasuk durasi nol.

Hanya cabang state awal Driver yang dieksekusi per tick. Karena merchant aktif
sebelum Driver, kedatangan pada tick t baru bisa menerima handover pada tick
berikutnya. Customer mengakui delivery pada tick berikutnya juga.

Horizon berhenti setelah tick 1439 tanpa drain. Order yang baru DELIVERED pada
tick 1439 tetap belum COMPLETED dan belum memperoleh revenue. Biaya dan emisi
perjalanan yang sudah dilakukan tetap dihitung untuk seluruh order.

Menutup visualizer lebih awal mengekspor kondisi PARTIAL tanpa melanjutkan run.
Render, Pause, dan Speed tidak mengonsumsi RNG model atau mengubah mekanisme.
Konfigurasi, input, seed dan versi dependensi yang sama menghasilkan data
simulasi identik; timestamp ekspor dan nama direktori tidak termasuk data tersebut.

## Antarmuka Python dan pengujian

```python
from src.simulation_engine import SimulationConfig, IntegratedDeliveryModel

model = IntegratedDeliveryModel(SimulationConfig(p_integration=0.5, seed=42))
model.step()           # Jalankan tick domain 0.
view = model.snapshot()  # Salinan data; bukan referensi agen yang bisa diubah.
summary = model.run()  # Lanjutkan sampai tick 1439, tanpa drain.
path = model.export_results()  # Tidak memajukan waktu.
```

Jalankan seluruh pengujian mekanisme dan visualizer tanpa jendela:

```powershell
python -B -m unittest discover -s tests -p "test_simulation*.py" -v
```

Pengujian memakai standard library `unittest`, tidak memerlukan pytest. Stack
yang diperiksa: Python 3.12.14, Mesa 3.5.1, NumPy 2.5.2, SciPy 1.18.1, pandas 3.0.5,
Matplotlib 3.11.1. Daftar dependensi yang sama tersedia dalam
`requirements-simulation.txt`.

Validasi historis baseline Manhattan pada 13 September 2026: **39 pengujian lulus**
(32 engine dan 7 visualizer headless) dalam environment `abms` saat itu.
Pada 19 September 2026, pengujian engine, controller, dan visualizer berjumlah
**49 pengujian lulus** (32 engine, 10 controller, dan 7 visualizer headless).
Pemeriksaan seed 42 untuk `p_integration` 0, 0,5, dan 1 masing-masing menghasilkan
300 order dan 1440 record KPI serta lolos rekonsiliasi status, jarak, dan keuangan.
Hasil ilustratif lengkap dan bukti pengujian tercantum dalam PDF; bukan target
kinerja atau hasil eksperimen multi-replikasi.
