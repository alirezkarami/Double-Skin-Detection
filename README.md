# تشخیص دوپوستگی بیلت (Double-Skin Detection)

مدل یادگیری عمیق برای تشخیص **دوپوستگی** روی تصاویر کراپ‌شده‌ی بیلت در خط تولید (دوربین Basler).
خروجی مدل برای هر تصویر یک **احتمال** بین ۰ تا ۱ است (احتمال دوپوسته بودن). اگر این احتمال از آستانه بیشتر باشد، تصویر دوپوسته اعلام می‌شود.

| کلاس | پوشه | معنی |
|---|---|---|
| 0 | `dataset/0/` | سالم (دوپوسته نیست) |
| 1 | `dataset/1/` | دوپوسته |

---

## ساختار پروژه

```
khani/
├── train.py              # آموزش + ارزیابی cross-validation + خروجی مدل نهایی
├── predict.py            # استنتاج (CLI و کلاس DoubleSkinDetector برای استفاده در کد)
├── tools/extract_rar.py  # باز کردن فایل rar دیتاست با libarchive سیستم
├── requirements.txt
├── data/dataset/{0,1}/   # تصاویر (در git نیست)
└── runs/<نام اجرا>/       # خروجی‌های آموزش (در git نیست)
```

---

## نصب

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy pillow onnxruntime
```

> **نکته‌ی شبکه:** اگر `files.pythonhosted.org` در دسترس نیست، `download.pytorch.org` معمولاً باز است.
> torch و torchvision را با `--no-deps` از ایندکس PyTorch نصب کنید و بقیه‌ی وابستگی‌ها را از یک پوشه‌ی wheel محلی
> (`pip install --no-index --find-links wheelhouse ...`).

GPU لازم نیست. آموزش روی CPU انجام می‌شود (با ۱۲ هسته، حدود ۵۰ ثانیه برای هر epoch).

---

## داده

دیتاست فعلی **۲۶۹۶ تصویر** دارد: ۲۶۵۷ سالم و ۳۹ دوپوسته.
تصاویر مونوکروم هستند (به‌صورت RGB ذخیره شده‌اند)، ارتفاعشان ۱۲۱۶ پیکسل و عرضشان حدود ۵۵۰ پیکسل است.

چند نکته‌ی مهم که روی طراحی اثر گذاشتند:

- **عدم توازن شدید** (حدود ۱.۵٪ مثبت). هر epoch به‌صورت متوازن نمونه‌برداری می‌شود: نمونه‌های دوپوسته با تکرار، و نمونه‌های سالم به‌صورت تصادفی.
- **فریم‌های تکراری:** از یک بیلت چند فریم پشت‌سرهم گرفته شده است (برچسب زمانی در نام فایل، با فاصله‌ی چند ثانیه).
  این ۳۹ تصویر در عمل فقط **۲۹ رویداد مستقل** هستند. فریم‌هایی که کمتر از ۲۰ ثانیه با هم فاصله دارند یک **گروه** حساب می‌شوند،
  و یک گروه هیچ‌وقت هم در آموزش و هم در ارزیابی نمی‌آید. اگر این کار انجام نشود، دقت روی کاغذ بیش از حد خوش‌بینانه می‌شود.

باز کردن فایل rar (بدون نیاز به unrar):

```bash
python3 -I tools/extract_rar.py dataset.rar data
```

---

## روش

- **مدل:** EfficientNet-B0 پیش‌آموزش‌دیده روی ImageNet (transfer learning) با یک خروجی sigmoid.
- **ورودی:** تصویر خاکستری با اندازه‌ی ۴۴۸×۱۹۲. نسبت ابعاد بیلت حفظ می‌شود تا بافت پوسته از بین نرود.
- **Augmentation:** قرینه‌ی افقی، جابه‌جایی، چرخش و مقیاس کوچک، تغییر روشنایی و کنتراست، و blur. قرینه‌ی عمودی استفاده نمی‌شود، چون جهت نورپردازی را به‌هم می‌زند.
- **ارزیابی:** **Cross-validation پنج‌تایی گروهی و طبقه‌بندی‌شده.** هر تصویر را مدلی پیش‌بینی می‌کند که آن تصویر (و فریم‌های هم‌گروهش) را ندیده است (out-of-fold).
- **آستانه:** از روی احتمال‌های out-of-fold دو آستانه انتخاب می‌شود:
  - `threshold`: آستانه‌ای که F1 را بیشینه می‌کند (پیش‌فرض)
  - `threshold_high_recall`: بالاترین آستانه‌ای که هنوز دست‌کم ۹۵٪ دوپوسته‌ها را می‌گیرد (برای وقتی که از دست دادن عیب گران‌تر از هشدار اشتباه است)
- **مدل نهایی:** بعد از CV، یک مدل روی **کل داده** آموزش می‌بیند و همراه با آستانه‌ها ذخیره می‌شود.
- **TTA:** در زمان استنتاج، میانگین پیش‌بینی تصویر اصلی و قرینه‌ی افقی‌اش گرفته می‌شود.

---

## آموزش

```bash
.venv/bin/python train.py --data data/dataset --out runs/v1 \
    --height 448 --width 192 --epochs 12 --pos_per_epoch 128 --neg_per_epoch 384
```

| آرگومان | توضیح |
|---|---|
| `--folds 0` | CV انجام نشود و فقط مدل نهایی آموزش ببیند (سریع‌تر). در این حالت آستانه ۰.۵ می‌ماند |
| `--arch` | `efficientnet_b0` (پیش‌فرض)، `resnet18` (حدود ۲ برابر سریع‌تر)، `mobilenet_v3_large` |

خروجی‌ها در `runs/v1/`:

| فایل | توضیح |
|---|---|
| `model_ts.pt` | مدل TorchScript برای استقرار (پیش‌پردازش و TTA داخل خود مدل است) |
| `model.onnx` | همان مدل با فرمت ONNX (برای C#/C++ یا onnxruntime) |
| `model.pt` | وزن‌ها و تنظیمات (برای ادامه‌ی آموزش) |
| `model_meta.json` | اندازه‌ی ورودی و آستانه‌ها |
| `results.json` | متریک‌های cross-validation |
| `cv_predictions.csv` | احتمال out-of-fold برای هر تصویر، مرتب‌شده (برای بررسی خطاها و برچسب‌های اشتباه) |
| `train.log` | لاگ آموزش |

---

## استفاده (استنتاج)

### خط فرمان

```bash
# یک تصویر
.venv/bin/python predict.py runs/v1 path/to/billet.jpg

# یک پوشه، با ذخیره‌ی نتایج در CSV
.venv/bin/python predict.py runs/v1 path/to/folder --csv result.csv

# با آستانه‌ی دلخواه
.venv/bin/python predict.py runs/v1 path/to/folder --threshold 0.3
```

خروجی نمونه:

```
0.9731  DOUBLE_SKIN  path/to/a.jpg
0.0123  OK           path/to/b.jpg
```

### داخل کد (کنار حلقه‌ی گرفتن تصویر Basler)

```python
from predict import DoubleSkinDetector

det = DoubleSkinDetector("runs/v1")          # یک بار در شروع برنامه

# cropped: تصویر کراپ‌شده‌ی بیلت، به‌صورت numpy array (mono8 یا RGB) یا PIL.Image
prob = det.predict(cropped)                   # float بین 0 و 1
if prob >= det.threshold:                     # یا det.threshold_high_recall
    print(f"دوپوسته! احتمال = {prob:.2f}")
```

نمونه با pypylon:

```python
from pypylon import pylon
cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
while cam.IsGrabbing():
    r = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if r.GrabSucceeded():
        frame = r.Array                        # mono8
        cropped = crop_billet(frame)           # همان کراپی که برای دیتاست استفاده شده
        prob = det.predict(cropped)
    r.Release()
```

> **مهم:** کراپ زمان استنتاج باید **همانند کراپ تصاویر آموزشی** باشد (همان ناحیه، همان جهت و همان نوردهی دوربین).

### بدون PyTorch (فقط onnxruntime)

ورودی مدل ONNX یک تصویر خاکستری `uint8` با شکل `[batch, 1, 448, 192]` است. خروجی آن احتمال دوپوستگی است و normalization و TTA داخل خود مدل انجام می‌شود.

```python
import numpy as np, onnxruntime as ort
from PIL import Image
sess = ort.InferenceSession("runs/v1/model.onnx")
x = np.asarray(Image.open("a.jpg").convert("L").resize((192, 448)), np.uint8)[None, None]
prob = sess.run(None, {"image_u8": x})[0][0]
```

---

## نتایج

نتایج cross-validation (out-of-fold) در `runs/v1/results.json` ذخیره می‌شوند.

<!-- RESULTS -->

---

## محدودیت‌ها و پیشنهادها

1. **داده‌ی دوپوسته خیلی کم است** (۲۹ رویداد مستقل). مهم‌ترین راه بهتر شدن مدل، **جمع‌آوری تصاویر دوپوسته‌ی بیشتر** است.
   برای شروع، `predict.py` را روی تصاویر خط اجرا کنید، تصاویری را که احتمال بالایی گرفته‌اند بازبینی کنید،
   موارد تأییدشده را به پوشه‌ی `1` اضافه کنید و دوباره آموزش دهید.
2. **بررسی برچسب‌ها:** بالای فایل `cv_predictions.csv` (تصاویر سالمی که احتمال بالا گرفته‌اند) را ببینید.
   ممکن است بعضی از آن‌ها در واقع دوپوسته باشند و اشتباه برچسب خورده باشند.
3. بعضی تصاویر پوشه‌ی `0` از نمای دیگری هستند (انتهای بیلت با نور شدید). اگر در خط واقعی چنین فریمی به مدل نمی‌رسد،
   بهتر است از دیتاست حذف شوند.
4. **انتخاب آستانه** به هزینه‌ی دو نوع خطا بستگی دارد: از دست دادن یک دوپوسته در برابر توقف یا بازرسی بی‌مورد.
   برای خط تولید معمولاً `threshold_high_recall` امن‌تر است.
5. وقتی داده بیشتر شد، مدل بزرگ‌تر یا رزولوشن بالاتر (مثلاً `--height 576 --width 256`) را امتحان کنید.
