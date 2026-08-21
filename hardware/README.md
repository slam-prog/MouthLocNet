# 🔌 هاردوير MouthLocNet

## نظرة عامة

هذا المجلد يحتوي على تصميمات الهاردوير لـ MouthLocNet.

---

## المكونات

### 1. PCB Design (`pcb_design/`)

تصميم دائرة مطبوعة لـ:
- 4 ميكروفونات MEMS
- ADC 768 kHz
- FPGA للمعالجة
- USB-C للطاقة والبيانات

### 2. FPGA Code (`fpga_code/`)

كود VHDL/Verilog لـ:
- معالجة الصوت في الوقت الفعلي
- حساب TDOA
- تشغيل نموذج MouthLocNet

### 3. Enclosure Design (`enclosure_design/`)

تصميم غلاف لـ:
- سماعة هيدفون مدمجة
- 4 ميكروفونات في ترتيب مربع
- بطارية LiPo 500mAh
- زر تشغيل/إيقاف

---

## المواصفات

| المقياس | القيمة |
|---------|--------|
| **الحجم** | 50 × 50 × 20 ملم |
| **الوزن** | 30 جرام |
| **الطاقة** | 100 mW |
| **البطارية** | 4 ساعات |
| **ميكروفونات** | 4 × MEMS I2S |
| **ADC** | 768 kHz, 24-bit |
| **FPGA** | Xilinx Artix-7 |
| **USB** | USB-C 3.0 |

---

## التجميع

### 1. PCB

```bash
# طلب من JLCPCB
https://jlcpcb.com/quote?file=pcb_design/mouthloc_gerbers.zip
```

### 2. FPGA

```bash
# Synthesis
vivado -mode batch -source fpga_code/synth.tcl

# Program
vivado -mode batch -source fpga_code/program.tcl
```

### 3. Enclosure

```bash
# 3D Print
# STL files in enclosure_design/
```

---

## التوصيل
## التوصيل
USB-C → FPGA → ADC → ميكروفونات
↓
MouthLocNet

text

---

## التكلفة

| المكون | التكلفة |
|--------|---------|
| PCB | $20 |
| FPGA | $30 |
| ADC | $15 |
| ميكروفونات | $10 |
| غلاف | $10 |
| بطارية | $10 |
| **الإجمالي** | **$95** |

---

## الترخيص

نفس ترخيص المشروع (HEUL v2.0)

---

## التواصل

للأسئلة حول الهاردوير:
- Issues: https://github.com/slam-prog/MouthLocNet/issues
- Discussions: https://github.com/slam-prog/MouthLocNet/discussions
